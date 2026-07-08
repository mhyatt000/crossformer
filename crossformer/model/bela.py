"""BELA: Bottleneck Encoder Latent Architecture — Perceiver-style trunk.

Replaces the block transformer with a latent cross-attention encoder: obs
tokenizer outputs are projected, tagged with view embeddings, and read into a
small learned latent array per readout (obs_xattn), which a few self-attention
layers then refine. The latents are returned as the readout TokenGroups that
heads consume.

Because the trunk runs once per action prediction while a flow head's ODE loop
reruns only the head, putting obs_xattn here means the (potentially large)
observation token soup is encoded once per prediction — not once per Euler
step. Pairs with PerceiverIOHead's act_xattn/fuse/decode split.

BELAModel subclasses CrossFormerModel via the module_cls hook, inheriting all
housekeeping (from_config, load/save_pretrained, sample_actions).
"""

from __future__ import annotations

from einops import rearrange
from flax import struct
import flax.linen as nn
import jax.numpy as jnp

from crossformer.embody import MAX_VIEWS
from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.io.attention import (
    CrossAttention,
    make_cross_attention_mask,
    SelfAttention,
)
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.model.crossformer_module import CrossFormerModule
from crossformer.utils.spec import ModuleSpec


class BELATransformer(nn.Module):
    """Latent encoder trunk: tokenize -> project (+view embed) -> obs_xattn -> process.

    API-compatible with CrossFormerTransformer.__call__: returns
    {f"readout_{name}": TokenGroup (B, W, n, D)} for each entry in readouts,
    where n is that readout's latent count. Timesteps are encoded independently
    (window folded into batch); task tokenizers, if any, contribute tokens to
    the same soup broadcast across the window.
    """

    observation_tokenizers: dict[str, nn.Module]
    task_tokenizers: dict[str, nn.Module]
    readouts: dict[str, int]
    token_embedding_size: int = 256
    num_layers: int = 2
    num_heads: int = 8
    widening_factor: int = 4
    dropout_rate: float = 0.0
    max_horizon: int = 1  # accepted for config compat; unused

    @nn.compact
    def __call__(self, observations, tasks, timestep_pad_mask, train: bool = True, verbose: bool = False):
        D = self.token_embedding_size
        view_embed = nn.Embed(MAX_VIEWS + 1, D, name="view_embed")

        groups: list[TokenGroup] = []
        B, W = timestep_pad_mask.shape[:2]
        for name, tok in self.observation_tokenizers.items():
            out: TokenGroup | None = tok(observations, tasks, train=train)
            if out is None:
                continue
            tokens = nn.Dense(D, name=f"obs_{name}_projection")(out.tokens)  # (B, W, n, D)
            view = out.view if out.view is not None else jnp.zeros(tokens.shape[:-1], dtype=jnp.int32)
            tokens = tokens + view_embed(view)
            mask = jnp.logical_and(timestep_pad_mask[:, :, None], out.mask)
            groups.append(TokenGroup(tokens, mask, view=view))
        for name, tok in self.task_tokenizers.items():
            out = tok(observations, tasks, train=train)
            if out is None:
                continue
            tokens = nn.Dense(D, name=f"task_{name}_projection")(out.tokens)  # (B, n, D)
            tokens = jnp.broadcast_to(tokens[:, None], (B, W, *tokens.shape[1:]))
            mask = jnp.broadcast_to(out.mask[:, None], tokens.shape[:-1])
            groups.append(TokenGroup(tokens, mask))
        assert groups, "BELA needs at least one tokenizer producing tokens"

        soup = TokenGroup.concatenate(groups)  # (B, W, S, D)
        x = rearrange(soup.tokens, "b w s d -> (b w) s d")
        x_mask = rearrange(soup.mask, "b w s -> (b w) s").astype(jnp.int32)

        attn = {
            "num_heads": self.num_heads,
            "widening_factor": self.widening_factor,
            "dropout_prob": self.dropout_rate,
        }
        outputs: dict[str, TokenGroup] = {}
        for r_name, n_latents in self.readouts.items():
            z0 = self.param(f"{r_name}_latents", nn.initializers.normal(0.02), (n_latents, D))
            z = jnp.broadcast_to(z0[None], (B * W, n_latents, D))
            xattn_mask = make_cross_attention_mask(jnp.ones((B * W, n_latents), dtype=jnp.int32), x_mask)
            z = CrossAttention(use_query_residual=True, shape_for_attn="kv", name=f"{r_name}_xattn", **attn)(
                z, x, xattn_mask, None, not train
            )
            for i in range(self.num_layers):
                z = SelfAttention(name=f"{r_name}_process_{i}", **attn)(z, None, not train)

            z = rearrange(z, "(b w) n d -> b w n d", b=B, w=W)
            z_mask = jnp.broadcast_to(timestep_pad_mask[:, :, None], (B, W, n_latents))
            outputs[f"readout_{r_name}"] = TokenGroup(z, z_mask)
        return outputs


class BELAModule(CrossFormerModule):
    """CrossFormerModule with a BELATransformer trunk (same head wiring)."""

    @classmethod
    def create(
        cls,
        observation_tokenizers: dict[str, ModuleSpec],
        task_tokenizers: dict[str, ModuleSpec],
        heads: dict[str, ModuleSpec],
        readouts: dict[str, int],
        transformer_kwargs: dict,
        token_embedding_size: int,
        max_horizon: int,
        repeat_task_tokens: bool = False,
    ) -> "BELAModule":
        del repeat_task_tokens  # prefix/timestep distinction doesn't exist here
        observation_tokenizer_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in observation_tokenizers.items()}
        task_tokenizer_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in task_tokenizers.items()}
        head_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in heads.items()}

        trunk = BELATransformer(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            readouts=readouts,
            token_embedding_size=token_embedding_size,
            num_layers=transformer_kwargs.get("num_layers", 2),
            num_heads=transformer_kwargs.get("num_attention_heads", 8),
            dropout_rate=transformer_kwargs.get("dropout_rate", 0.0),
            max_horizon=max_horizon,
        )
        return cls(crossformer_transformer=trunk, heads=head_defs)


@struct.dataclass
class BELAModel(CrossFormerModel):
    """CrossFormerModel housekeeping over a BELA trunk.

    Re-decorated with @struct.dataclass: pytree registration is per-class, and
    TrainState flattens the model field.
    """

    module_cls = BELAModule
