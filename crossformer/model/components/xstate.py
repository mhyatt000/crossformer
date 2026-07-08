from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.embody import MASK_ID, MAX_VIEWS, VOCAB_SIZE
from crossformer.model.components.base import TokenGroup
from crossformer.model.components.diffusion import FourierFeatures
from crossformer.model.components.heads.io.attention import CrossAttention, make_cross_attention_mask, SelfAttention


def _broadcast_slots(x: ArrayLike, base: Array, name: str) -> Array:
    arr = jnp.asarray(x)
    if arr.ndim == 2:
        return jnp.broadcast_to(arr[:, None, :], base.shape)
    if arr.ndim == 3:
        if arr.shape != base.shape:
            raise ValueError(f"{name} has shape {arr.shape}; expected {base.shape}")
        return arr
    raise ValueError(f"{name} must have shape (B, A) or (B, W, A); got {arr.shape}")


def _state_mask(observations: Mapping[str, Any], state: Mapping[str, Any], base: Array, dof_ids: Array) -> Array:
    mask = None
    obs_mask = observations.get("mask")
    if isinstance(obs_mask, Mapping):
        mask = obs_mask.get("states")
        if mask is None:
            state_mask = obs_mask.get("state")
            if isinstance(state_mask, Mapping):
                mask = state_mask.get("base")
    if mask is None:
        mask = state.get("mask")
    if mask is None:
        mask = jnp.ones(dof_ids.shape, dtype=bool)

    mask = _broadcast_slots(mask, base, "mask.states").astype(bool)
    return mask & (_broadcast_slots(dof_ids, base, "state.id") != MASK_ID)


class XStatePerceiver(nn.Module):
    """Perceiver encoder from state slots to a fixed latent set."""

    num_heads: int = 8
    num_blocks: int = 2
    num_self_attend_layers: int = 1
    widening_factor: int = 4
    dropout_prob: float = 0.0
    qk_channels: int | None = None
    v_channels: int | None = None

    @nn.compact
    def __call__(
        self,
        latents: Array,
        state_tokens: Array,
        *,
        attention_mask: ArrayLike,
        deterministic: bool = True,
    ) -> Array:
        x = latents
        for b in range(self.num_blocks):
            x = CrossAttention(
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
                dropout_prob=self.dropout_prob,
                qk_channels=self.qk_channels,
                v_channels=self.v_channels,
                use_query_residual=True,
                shape_for_attn="kv",
                name=f"cross_attend_{b}",
            )(x, state_tokens, attention_mask=attention_mask, deterministic=deterministic)

            for i in range(self.num_self_attend_layers):
                x = SelfAttention(
                    num_heads=self.num_heads,
                    widening_factor=self.widening_factor,
                    dropout_prob=self.dropout_prob,
                    qk_channels=self.qk_channels,
                    v_channels=self.v_channels,
                    name=f"self_attend_{b}_{i}",
                )(x, deterministic=deterministic)
        return x


class XStateEncoder(nn.Module):
    """Encode observation.state slots into fixed observation tokens.

    Shape contract:
        observation.state.base: (B, W, A) float state values.
        observation.state.id:   (B, A) or (B, W, A) DOF ids.
        observation.state.view: (B, A) or (B, W, A) view ids, optional.
        observation.mask.state.base: (B, A) or (B, W, A) slot validity, optional.
        output.tokens: (B, W, M, D), where M=num_latents and D=num_channels.
    """

    num_latents: int = 8
    num_channels: int = 256
    num_heads: int = 8
    num_blocks: int = 2
    num_self_attend_layers: int = 1
    widening_factor: int = 4
    dropout_prob: float = 0.0
    value_fourier_dim: int = 32
    slot_fourier_dim: int = 32
    dof_vocab_size: int = VOCAB_SIZE
    input_drop_prob: float = 0.25
    latent_drop_prob: float = 0.25
    skip_missing: bool = True

    @nn.compact
    def __call__(
        self,
        observations: Mapping[str, Any],
        tasks: Mapping[str, Any] | None = None,
        train: bool = True,
    ) -> TokenGroup | None:
        del tasks
        state = observations.get("state")
        if not isinstance(state, Mapping):
            if self.skip_missing:
                return None
            raise KeyError("XStateEncoder expects observations['state']")

        base = jnp.asarray(state["base"])
        if base.ndim != 3:
            raise ValueError(f"state.base must have shape (B, W, A); got {base.shape}")

        bsz, win, num_slots = base.shape
        # base: (B, W, A); dof_ids/view_ids/valid are broadcast to match.
        dof_ids = _broadcast_slots(state["id"], base, "state.id").astype(jnp.int32)
        view_ids = _broadcast_slots(
            state.get("view", jnp.zeros(dof_ids.shape[:1] + dof_ids.shape[2:])), base, "state.view"
        )
        view_ids = view_ids.astype(jnp.int32)
        if train and self.input_drop_prob > 0:
            keep = jax.random.bernoulli(self.make_rng("dropout"), 1.0 - self.input_drop_prob, dof_ids.shape)
            dof_ids = jnp.where(keep, dof_ids, MASK_ID)
            view_ids = jnp.where(keep, view_ids, 0)
        slot_pos = jnp.broadcast_to(jnp.arange(num_slots, dtype=base.dtype), base.shape)
        valid = _state_mask(observations, state, base, dof_ids)

        # state_tokens: (B, W, A, D)
        value = FourierFeatures(self.value_fourier_dim, learnable=True, name="value_ff")(base[..., None])
        value = nn.Dense(self.num_channels, name="value_proj")(value)
        dof = nn.Embed(self.dof_vocab_size, self.num_channels, name="dof_embed")(dof_ids)
        view = nn.Embed(MAX_VIEWS + 1, self.num_channels, name="view_embed")(view_ids)
        slot = FourierFeatures(self.slot_fourier_dim, learnable=True, name="slot_ff")(slot_pos[..., None])
        slot = nn.Dense(self.num_channels, name="slot_proj")(slot)
        state_tokens = value + dof + view + slot

        # Perceiver runs per sample timestep: (B*W, A, D) -> (B*W, M, D).
        flat_tokens = jnp.reshape(state_tokens, (bsz * win, num_slots, self.num_channels))
        flat_valid = jnp.reshape(valid, (bsz * win, num_slots))

        null_token = self.param(
            "null_state",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_channels),
        )
        null_tokens = jnp.broadcast_to(null_token[None, :, :], (bsz * win, 1, self.num_channels))
        flat_tokens = jnp.concatenate([flat_tokens, null_tokens], axis=1)
        flat_valid = jnp.concatenate([flat_valid, jnp.ones((bsz * win, 1), dtype=bool)], axis=1)

        latents = self.param(
            "latents",
            nn.initializers.normal(stddev=0.02),
            (self.num_latents, self.num_channels),
        )
        latents = jnp.broadcast_to(latents[None, :, :], (bsz * win, self.num_latents, self.num_channels))
        query_mask = jnp.ones((bsz * win, self.num_latents), dtype=bool)
        attn_mask = make_cross_attention_mask(query_mask, flat_valid)

        latents = XStatePerceiver(
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            num_self_attend_layers=self.num_self_attend_layers,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            name="encoder",
        )(latents, flat_tokens, attention_mask=attn_mask, deterministic=not train)
        latents = nn.LayerNorm(name="ln_out")(latents)

        tokens = jnp.reshape(latents, (bsz, win, self.num_latents, self.num_channels))
        mask = jnp.ones((bsz, win, self.num_latents), dtype=bool)
        if train and self.latent_drop_prob > 0:
            keep = jax.random.bernoulli(self.make_rng("dropout"), 1.0 - self.latent_drop_prob, (bsz, 1, 1))
            mask = mask & keep
        view = jnp.zeros((bsz, win, self.num_latents), dtype=jnp.int32)
        return TokenGroup(tokens, mask, view=view)
