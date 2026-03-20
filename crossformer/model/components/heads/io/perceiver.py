"""Perceiver IO encoder/decoder architecture (Flax port)."""

from __future__ import annotations

import abc

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from .attention import CrossAttention, make_cross_attention_mask, SelfAttention
from .position_encoding import build_position_encoding, TrainablePositionEncoding


class PerceiverEncoder(nn.Module):
    """Perceiver encoder: cross-attend inputs into latents, then self-attend.

    Total self-attend layers = num_self_attends_per_block * num_blocks.
    Weights are shared across blocks.
    """

    num_self_attends_per_block: int = 6
    num_blocks: int = 8
    z_index_dim: int = 512
    num_z_channels: int = 1024
    qk_channels: int | None = None
    v_channels: int | None = None
    num_cross_attend_heads: int = 1
    num_self_attend_heads: int = 8
    cross_attend_widening_factor: int = 1
    self_attend_widening_factor: int = 1
    dropout_prob: float = 0.0
    z_pos_enc_init_scale: float = 0.02
    cross_attention_shape_for_attn: str = "kv"
    use_query_residual: bool = True

    def setup(self):
        if self.num_z_channels % self.num_self_attend_heads != 0:
            raise ValueError(
                f"num_z_channels ({self.num_z_channels}) must be divisible by "
                f"num_self_attend_heads ({self.num_self_attend_heads})."
            )
        if self.num_z_channels % self.num_cross_attend_heads != 0:
            raise ValueError(
                f"num_z_channels ({self.num_z_channels}) must be divisible by "
                f"num_cross_attend_heads ({self.num_cross_attend_heads})."
            )

        self.z_pos_enc = TrainablePositionEncoding(
            index_dim=self.z_index_dim,
            num_channels=self.num_z_channels,
            init_scale=self.z_pos_enc_init_scale,
            name="z_pos_enc",
        )

        self.cross_attend = CrossAttention(
            dropout_prob=self.dropout_prob,
            num_heads=self.num_cross_attend_heads,
            widening_factor=self.cross_attend_widening_factor,
            shape_for_attn=self.cross_attention_shape_for_attn,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            use_query_residual=self.use_query_residual,
            name="cross_attend",
        )

        self.self_attends = [
            SelfAttention(
                num_heads=self.num_self_attend_heads,
                dropout_prob=self.dropout_prob,
                qk_channels=self.qk_channels,
                v_channels=self.v_channels,
                widening_factor=self.self_attend_widening_factor,
                name=f"self_attend_{i}",
            )
            for i in range(self.num_self_attends_per_block)
        ]

    def latents(self, inputs):
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def __call__(self, inputs, z, *, deterministic=True, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=jnp.ones(z.shape[:2], dtype=jnp.int32),
                kv_mask=input_mask,
            )
        z = self.cross_attend(
            z,
            inputs,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )
        for _ in range(self.num_blocks):
            for sa in self.self_attends:
                z = sa(z, deterministic=deterministic)
        return z


class AbstractPerceiverDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Abstract decoder interface."""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, query, z, *, deterministic=True, query_mask=None):
        raise NotImplementedError


class ProjectionDecoder(AbstractPerceiverDecoder):
    """Mean-pool latents then project (no cross-attention)."""

    num_classes: int

    @nn.compact
    def __call__(self, query, z, *, deterministic=True, query_mask=None):
        z = jnp.mean(z, axis=1, dtype=z.dtype)
        return nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros, name="logits")(z)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return None

    def output_shape(self, inputs):
        return (inputs.shape[0], self.num_classes), None


class BasicDecoder(AbstractPerceiverDecoder):
    """Cross-attention decoder with optional output position encoding."""

    output_num_channels: int
    position_encoding_type: str = "trainable"
    output_index_dims: tuple[int, ...] | None = None
    subsampled_index_dims: int | None = None
    num_z_channels: int = 1024
    qk_channels: int | None = None
    v_channels: int | None = None
    use_query_residual: bool = False
    concat_preprocessed_input: bool = False
    num_heads: int = 1
    final_project: bool = True
    position_encoding_kwargs: dict | None = None

    def setup(self):
        self._subsampled_index_dims = (
            self.subsampled_index_dims
            if self.subsampled_index_dims is not None
            else (int(np.prod(self.output_index_dims)) if self.output_index_dims else None)
        )

        self.output_pos_enc = None
        if self.position_encoding_type != "none":
            kwargs = self.position_encoding_kwargs or {}
            self.output_pos_enc = build_position_encoding(
                self.position_encoding_type,
                index_dims=self.output_index_dims,
                **kwargs,
            )

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        assert self.position_encoding_type != "none"
        if subsampled_points is not None:
            pos = jnp.stack(
                jnp.unravel_index(subsampled_points, self.output_index_dims),
                axis=1,
            )
            pos = -1 + 2 * pos / jnp.array(self.output_index_dims)[None, :]
            pos = jnp.broadcast_to(pos[None], [inputs.shape[0], pos.shape[0], pos.shape[1]])
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0], pos=pos)
            pos_emb = jnp.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0])

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("inputs_without_pos required when concat_preprocessed_input")
            pos_emb = jnp.concatenate([inputs_without_pos, pos_emb], axis=-1)
        return pos_emb

    def output_shape(self, inputs):
        return (
            (inputs.shape[0], self._subsampled_index_dims, self.output_num_channels),
            None,
        )

    @nn.compact
    def __call__(self, query, z, *, deterministic=True, query_mask=None):
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=query_mask,
                kv_mask=jnp.ones(z.shape[:2], dtype=jnp.int32),
            )

        output = CrossAttention(
            dropout_prob=0.0,
            num_heads=self.num_heads,
            widening_factor=1,
            shape_for_attn="kv",
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            use_query_residual=self.use_query_residual,
            name="decoding_cross_attn",
        )(query, z, attention_mask=attention_mask, deterministic=deterministic)

        if self.final_project:
            output = nn.Dense(self.output_num_channels, name="output")(output)
        return output


class Perceiver(nn.Module):
    """Full Perceiver IO: preprocess -> encode -> decode -> postprocess."""

    encoder: PerceiverEncoder
    decoder: AbstractPerceiverDecoder
    input_preprocessor: nn.Module | None = None
    output_postprocessor: nn.Module | None = None

    @nn.compact
    def __call__(
        self,
        inputs,
        *,
        deterministic=True,
        subsampled_output_points=None,
        pos=None,
        input_mask=None,
        query_mask=None,
    ):
        if self.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(
                inputs,
                pos=pos,
                deterministic=deterministic,
                network_input_is_1d=True,
            )
        else:
            modality_sizes = None
            inputs_without_pos = None

        z = self.encoder.latents(inputs)
        decoder_query = self.decoder.decoder_query(
            inputs,
            modality_sizes,
            inputs_without_pos,
            subsampled_points=subsampled_output_points,
        )

        z = self.encoder(
            inputs,
            z,
            deterministic=deterministic,
            input_mask=input_mask,
        )

        _, output_modality_sizes = self.decoder.output_shape(inputs)
        output_modality_sizes = output_modality_sizes or modality_sizes

        outputs = self.decoder(
            decoder_query,
            z,
            deterministic=deterministic,
            query_mask=query_mask,
        )

        if self.output_postprocessor is not None:
            outputs = self.output_postprocessor(
                outputs,
                deterministic=deterministic,
                modality_sizes=output_modality_sizes,
            )
        return outputs
