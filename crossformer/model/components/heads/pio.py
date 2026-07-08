"""Perceiver IO flow-matching head: encode-process-decode with a latent bottleneck.

Same DOF-vocabulary queries, loss, and ODE sampling as XFlowHead — but instead of
running self-attention over all max_H*max_A query tokens, the per-scalar action
tokens (and the obs/guidance context) are cross-attended INTO a small learned
latent array, all depth runs on the latents, and the per-scalar output queries
cross-attend back out once. Activation memory is linear in query count with no
quadratic term, so large batches fit.

Padded slots are masked on the kv side of the encode cross-attention, so the
latents never read dead tokens; padded output queries produce values that the
loss mask discards.
"""

from __future__ import annotations

from einops import rearrange
import flax.linen as nn
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.model.components.base import TokenGroup

from .dof import build_query_mask
from .io.attention import CrossAttention, make_cross_attention_mask, SelfAttention
from .xflow import XFlowHead


class PerceiverIOHead(XFlowHead):
    """Latent-bottleneck variant of XFlowHead (new param tree).

    Inherits the full loss/predict_action API, the factored query vocabulary,
    and guidance handling. `num_blocks * num_self_attend_layers` sets the
    latent process depth; `factor_attn` is ignored.

    Two encode topologies, selected by ``num_act_latents``:

    0 (legacy): one latent array cross-attends the concatenated
        [action tokens | context] kv — obs tokens share a softmax with
        max_H*max_A action tokens.
    M > 0 (act_xattn/fuse): action tokens are first compressed into M action
        latents (act_xattn), then the context — already latent-encoded when the
        trunk is BELA — cross-attends [m_t | guidance] for ``num_fuse_layers``
        blocks (fuse), and decode reads the fused context. Obs never attends
        raw a_t; joint obs x a_t depth lives in the fuse blocks. With a BELA
        trunk the obs encoding happens once per prediction, outside the ODE
        loop.
    """

    num_latents: int = 128
    # act_xattn/fuse topology: number of action-summary latents (0 = legacy
    # single-encode over concatenated kv).
    num_act_latents: int = 0
    num_fuse_layers: int = 2

    def setup(self) -> None:
        # Query/action/time builders, output_proj, guidance. The parent's dense
        # decoder is defined but never called, so it materializes no params.
        super().setup()
        D = self.num_query_channels
        attn = {
            "num_heads": self.num_heads,
            "widening_factor": self.widening_factor,
            "dropout_prob": self.dropout_prob,
        }
        # static_argnums counts `self` as 0: deterministic is positional arg 5
        # (CrossAttention) / 3 (SelfAttention); remat requires static args be
        # passed positionally, hence the positional call style in __call__.
        CrossAttn, SelfAttn = CrossAttention, SelfAttention
        if self.remat:
            CrossAttn = nn.remat(CrossAttention, static_argnums=(5,))
            SelfAttn = nn.remat(SelfAttention, static_argnums=(3,))
        self.context_proj = nn.Dense(D)
        self.decode_xattn = CrossAttn(use_query_residual=False, shape_for_attn="kv", **attn)
        if self.num_act_latents > 0:
            self.act_latent_init = self.param(
                "act_latents", nn.initializers.normal(0.02), (self.num_act_latents, D)
            )
            self.act_xattn = CrossAttn(use_query_residual=True, shape_for_attn="kv", **attn)
            self.fuse = [CrossAttn(use_query_residual=True, shape_for_attn="kv", **attn) for _ in range(self.num_fuse_layers)]
        else:
            self.latent_init = self.param("latents", nn.initializers.normal(0.02), (self.num_latents, D))
            self.encode_xattn = CrossAttn(use_query_residual=True, shape_for_attn="kv", **attn)
            self.process = [SelfAttn(**attn) for _ in range(self.num_blocks * self.num_self_attend_layers)]

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        a_t: ArrayLike | None = None,
        dof_ids: ArrayLike | None = None,
        chunk_steps: ArrayLike | None = None,
        slot_pos: ArrayLike | None = None,
        view_ids: ArrayLike | None = None,
        train: bool = True,
        guide_input: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
    ) -> Array:
        """Predict action velocities. Same contract as XFlowHead.__call__."""
        max_H, max_A = self.max_horizon, self.max_dofs
        if self.pool_strategy != "pass":
            raise NotImplementedError("PerceiverIOHead supports pool_strategy='pass' only")
        if self.readout_only:
            group = transformer_outputs[self.readout_key]
        else:
            group = TokenGroup.concatenate(list(transformer_outputs.values()))
        assert group.tokens.ndim == 4
        embeddings = group.tokens  # (B, W, S, E)
        ctx_mask = group.mask  # (B, W, S)

        # During init provide zero dummies
        if self.is_initializing():
            B, W = embeddings.shape[:2]
            dof_ids = jnp.zeros((B, max_A), dtype=jnp.int32)
            chunk_steps = jnp.zeros((B, max_H), dtype=jnp.float32)
            slot_pos = jnp.zeros((B, max_A), dtype=jnp.float32)
            view_ids = jnp.zeros((B, max_A), dtype=jnp.int32)
            time = jnp.zeros((B, W, 1))
            a_t = jnp.zeros((B, W, max_H, max_A))
            if self.use_guidance and guide_input is None:
                guide_input = jnp.zeros((B, 1, self.guidance_input_dim), dtype=embeddings.dtype)
        elif time is None or a_t is None or dof_ids is None or chunk_steps is None:
            raise ValueError("Must provide time, a_t, dof_ids, chunk_steps")

        guidance_tokens = self._encode_guidance(guide_input, train=train)

        if slot_pos is None:
            slot_pos = jnp.broadcast_to(jnp.arange(max_A, dtype=jnp.float32), dof_ids.shape)
        if view_ids is None:
            view_ids = jnp.zeros_like(dof_ids)

        if a_t.ndim == 3:
            a_t = rearrange(a_t, "b w (h a) -> b w h a", h=max_H, a=max_A)

        B, W = time.shape[:2]

        # Merge batch and window: (B, W, ...) -> (BW, ...)
        embed_bw = rearrange(embeddings, "b w s e -> (b w) s e")
        ctx_mask_bw = rearrange(ctx_mask, "b w s -> (b w) s")
        time_bw = rearrange(time, "b w t -> (b w) t")
        a_t_bw = rearrange(a_t, "b w h a -> (b w) h a")

        # Tile per-sample specs across window: (B, ...) -> (BW, ...)
        dof_bw = jnp.repeat(dof_ids, W, axis=0)
        chunk_bw = jnp.repeat(chunk_steps, W, axis=0)
        slot_bw = jnp.repeat(slot_pos, W, axis=0)
        view_bw = jnp.repeat(view_ids, W, axis=0)

        # Per-scalar action tokens: vocab encoding + a_t + tau. Used both as the
        # encode-side kv array and as the decode-side output queries.
        action_tokens = self._build_queries(chunk_bw, dof_bw, slot_bw, view_bw, time_bw, a_t_bw)
        q_mask = build_query_mask(chunk_bw, dof_bw, slot_bw)  # (BW, max_H*max_A)

        # Context: obs/readout tokens projected to query width. Guidance tokens
        # (CFG-droppable via guidance_mask) are always kv-side conditioning.
        ctx = self.context_proj(embed_bw)  # (BW, S, D)
        ctx_mask_i = ctx_mask_bw.astype(jnp.int32)
        q_mask_i = q_mask.astype(jnp.int32)
        guide_kv = guide_mask_i = None
        if guidance_tokens is not None:
            G = guidance_tokens.shape[1]
            guide_bw = rearrange(jnp.tile(guidance_tokens[:, None], (1, W, 1, 1)), "b w g e -> (b w) g e")
            guide_kv = self.context_proj(guide_bw)
            if guidance_mask is not None:
                guide_mask_i = rearrange(jnp.tile(guidance_mask[:, None], (1, W, 1)), "b w g -> (b w) g")
            else:
                guide_mask_i = jnp.ones((B * W, G), dtype=jnp.int32)

        if self.num_act_latents > 0:
            # act_xattn: compress per-scalar action tokens into M action latents.
            m = jnp.broadcast_to(self.act_latent_init[None], (B * W, *self.act_latent_init.shape))
            act_mask = make_cross_attention_mask(
                jnp.ones((B * W, self.num_act_latents), dtype=jnp.int32), q_mask_i
            )
            m = self.act_xattn(m, action_tokens, act_mask, None, not train)

            fuse_kv, fuse_kv_mask = m, jnp.ones((B * W, self.num_act_latents), dtype=jnp.int32)
            if guide_kv is not None:
                fuse_kv = jnp.concatenate([fuse_kv, guide_kv], axis=1)
                fuse_kv_mask = jnp.concatenate([fuse_kv_mask, guide_mask_i], axis=1)

            # fuse: context queries read [m_t | guidance]; obs never attends raw
            # a_t, and joint obs x action-state depth lives here.
            z = ctx
            fuse_mask = make_cross_attention_mask(ctx_mask_i, fuse_kv_mask)
            for layer in self.fuse:
                z = layer(z, fuse_kv, fuse_mask, None, not train)

            # Decode: per-scalar queries read the fused context.
            dec_mask = make_cross_attention_mask(q_mask_i, ctx_mask_i)
            decoded = self.decode_xattn(action_tokens, z, dec_mask, None, not train)
        else:
            # Legacy single-encode: latents read [action tokens | context | guidance].
            kv = jnp.concatenate([action_tokens, ctx], axis=1)
            kv_mask = jnp.concatenate([q_mask_i, ctx_mask_i], axis=1)
            if guide_kv is not None:
                kv = jnp.concatenate([kv, guide_kv], axis=1)
                kv_mask = jnp.concatenate([kv_mask, guide_mask_i], axis=1)

            z = jnp.broadcast_to(self.latent_init[None], (B * W, *self.latent_init.shape))
            enc_mask = make_cross_attention_mask(jnp.ones((B * W, self.num_latents), dtype=jnp.int32), kv_mask)
            z = self.encode_xattn(z, kv, enc_mask, None, not train)

            for layer in self.process:
                z = layer(z, None, not train)

            # Decode: per-scalar queries read the latents once; no self-attention.
            decoded = self.decode_xattn(action_tokens, z, None, None, not train)

        output = self.output_proj(decoded).squeeze(-1)  # (BW, max_H*max_A)
        return rearrange(output, "(b w) q -> b w q", b=B, w=W)
