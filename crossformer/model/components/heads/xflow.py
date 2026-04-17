"""Cross-embodiment flow-matching action head using Perceiver IO decoding."""

from __future__ import annotations

from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.diffusion import FourierFeatures
from crossformer.model.components.guidance import TokenGuidance
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.mytyping import PRNGKey

from .base import ActionHead
from .dof import build_query_mask, FactoredQueryEncoding
from .io.attention import CrossAttention, make_cross_attention_mask, SelfAttention
from .losses import continuous_loss, sample_tau


class PerceiverDecoder(nn.Module):
    """Perceiver IO decoder: interleaved [cross-attn, self-attn x k] blocks.

    num_blocks=1 matches the legacy single-cross design (one cross-attn, then
    num_self_attend_layers self-attn). Larger num_blocks re-reads the context
    after self-refinement so later layers aren't blind to vision.

    Args:
        num_heads: heads for both cross- and self-attention.
        num_blocks: number of [cross, self x k] blocks.
        num_self_attend_layers: self-attn layers per block (k).
        widening_factor: MLP expansion ratio inside attention blocks.
        dropout_prob: dropout rate for attention and MLP.
        qk_channels: override QK projection dim (default: inferred from inputs).
        v_channels: override V projection dim.
    """

    num_heads: int = 8
    num_blocks: int = 4
    num_self_attend_layers: int = 1
    widening_factor: int = 4
    dropout_prob: float = 0.0
    qk_channels: int | None = None
    v_channels: int | None = None

    @nn.compact
    def __call__(self, queries, context, *, deterministic=True, attention_mask=None):
        """Interleave cross-attn and self-attn blocks.

        Args:
            queries: (batch, seq_q, d_q)
            context: (batch, seq_kv, d_kv)
            attention_mask: (batch, seq_q, seq_kv) bool mask for cross-attention.

        Returns:
            (batch, seq_q, d_q)
        """
        # Preserve legacy param names at num_blocks=1 so old checkpoints load.
        legacy = self.num_blocks == 1
        x = queries
        for b in range(self.num_blocks):
            x = CrossAttention(
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
                dropout_prob=self.dropout_prob,
                qk_channels=self.qk_channels,
                v_channels=self.v_channels,
                use_query_residual=True,
                shape_for_attn="kv",
                name="cross_attend" if legacy else f"cross_attend_{b}",
            )(x, context, attention_mask=attention_mask, deterministic=deterministic)

            for i in range(self.num_self_attend_layers):
                x = SelfAttention(
                    num_heads=self.num_heads,
                    widening_factor=self.widening_factor,
                    dropout_prob=self.dropout_prob,
                    qk_channels=self.qk_channels,
                    v_channels=self.v_channels,
                    name=f"self_attend_{i}" if legacy else f"self_attend_{b}_{i}",
                )(x, deterministic=deterministic)

        return x


class XFlowHead(nn.Module, ActionHead):
    """Perceiver IO flow-matching head for cross-embodied action prediction.

    Embodiment-agnostic: accepts per-sample dof_ids and chunk_steps at forward
    time. The DOF vocabulary embedding is shared across all embodiments.
    Padded positions (MASK DOF / CHUNK_PAD) are masked in attention.

    Single-kernel training: all embodiments in one batch, padded to max size.

    Shape contract:
        transformer tokens: (B, W, N, E)
        dof_ids:            (B, max_A) int — per-sample DOF vocab IDs, MASK-padded
        chunk_steps:        (B, max_H) float — per-sample temporal positions, padded
        actions:            (B, W, max_H, max_A) — padded with zeros
        guide_input:        (B, S, D) — optional raw guidance input
    """

    # Identity
    readout_key: str

    # Structural bounds (determines max query count = max_horizon * max_dofs)
    max_dofs: int = 50
    max_horizon: int = 20
    clip_pred: bool = True
    max_action: float = 5.0
    loss_type: str = "mse"
    loss_weight: float = 1.0

    # Perceiver decoder
    num_query_channels: int = 256
    num_heads: int = 8
    num_blocks: int = 4
    num_self_attend_layers: int = 1
    widening_factor: int = 4
    dropout_prob: float = 0.1

    # Flow matching
    time_dim: int = 32
    flow_steps: int = 25
    base_std: float = 1.0

    # Pooling strategy for transformer outputs
    pool_strategy: str = "pass"

    # If True, cross-attend only to transformer_outputs[readout_key].
    # If False, concat all transformer_outputs groups along the token axis.
    readout_only: bool = False

    # Guidance
    use_guidance: bool = False
    guidance_embed_dim: int | None = None
    guidance_input_dim: int | None = None
    compress_guidance: bool = False
    num_guidance_latents: int = 4

    def setup(self):
        D = self.num_query_channels

        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        # Factored queries: Fourier chunk position x learned DOF embedding
        self.query_pos_enc = FactoredQueryEncoding(
            num_channels=D,
            name="query_pos_enc",
        )

        # Per-query scalar action conditioning via Fourier features
        self.action_features = FourierFeatures(
            self.time_dim,
            learnable=True,
            name="action_ff",
        )
        self.action_proj = nn.Dense(D, name="action_proj")

        # Time conditioning
        self.time_features = FourierFeatures(self.time_dim, learnable=True, name="time_ff")
        self.time_proj = nn.Dense(D, name="time_proj")

        # Decoder and scalar output projection (one value per query)
        self.decoder = PerceiverDecoder(
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            num_self_attend_layers=self.num_self_attend_layers,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            name="decoder",
        )
        self.output_proj = nn.Dense(1, name="output_proj")
        if self.use_guidance:
            if self.guidance_embed_dim is None:
                raise ValueError("guidance_embed_dim must be set when use_guidance=True")
            if self.guidance_input_dim is None:
                raise ValueError("guidance_input_dim must be set when use_guidance=True")
            self.guidance = TokenGuidance(
                embed_dim=self.guidance_embed_dim,
                compress=self.compress_guidance,
                num_latents=self.num_guidance_latents,
                name="guidance",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, transformer_outputs: dict[str, TokenGroup], train: bool) -> Array:
        """Return context embeddings as (B, W, E) pooled or (B, W, N, E) unpooled.

        Selection:
          readout_only=True  -> transformer_outputs[readout_key] only
          readout_only=False -> concat all groups along the token axis

        Pooling:
          "pass"    -> (B, W, N, E) unpooled (K=N cross-attn tokens)
          "mean"    -> (B, W, E) mean-reduced (K=1)
          "use_map" -> (B, W, E) MAP-pooled (K=1)
        """
        if self.readout_only:
            token_group = transformer_outputs[self.readout_key]
        else:
            token_group = TokenGroup.concatenate(list(transformer_outputs.values()))
        assert token_group.tokens.ndim == 4
        if self.pool_strategy == "use_map":
            return self.map_head(token_group, train=train)[:, :, 0]
        if self.pool_strategy == "mean":
            return token_group.tokens.mean(axis=-2)
        if self.pool_strategy == "pass":
            return token_group.tokens
        raise ValueError(f"{self.pool_strategy} not implemented!")

    def _build_queries(
        self,
        chunk_steps: ArrayLike,
        dof_ids: ArrayLike,
        slot_pos: ArrayLike,
        time: ArrayLike,
        a_t: ArrayLike,
    ) -> Array:
        """Build factored conditioned queries: (chunk + dof + slot) + action + time.

        Args:
            chunk_steps: (BW, max_H) float.
            dof_ids: (BW, max_A) int.
            slot_pos: (BW, max_A) float.
            time: (BW, 1) flow timestep.
            a_t: (BW, max_H, max_A) noisy actions.

        Returns:
            (BW, max_H*max_A, D)
        """
        pos_q = self.query_pos_enc(chunk_steps, dof_ids, slot_pos)  # (BW, max_H*max_A, D)

        # Per-scalar action conditioning
        a_flat = rearrange(a_t, "bw h a -> bw (h a) 1")
        act_cond = self.action_proj(self.action_features(a_flat))  # (BW, max_H*max_A, D)

        # Time conditioning — broadcast across all queries
        t_cond = self.time_proj(self.time_features(time))[:, None, :]  # (BW, 1, D)

        return pos_q + act_cond + t_cond

    def _encode_guidance(self, guide_input: ArrayLike | None, train: bool) -> Array | None:
        """Encode raw guidance input to context-width tokens."""
        if not self.use_guidance or guide_input is None:
            return None
        return self.guidance(guide_input, deterministic=not train)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        a_t: ArrayLike | None = None,
        dof_ids: ArrayLike | None = None,
        chunk_steps: ArrayLike | None = None,
        slot_pos: ArrayLike | None = None,
        train: bool = True,
        guide_input: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
    ) -> Array:
        """Predict action velocities.

        Args:
            transformer_outputs: dict mapping readout_key -> TokenGroup (B, W, N, E).
            time: (B, W, 1) flow timestep in [0, 1].
            a_t: (B, W, max_H, max_A) or (B, W, max_H*max_A) noisy actions.
            dof_ids: (B, max_A) int — MASK-padded DOF vocab IDs.
            chunk_steps: (B, max_H) float — padded temporal positions.
            slot_pos: (B, max_A) float — ordinal position in action vector.
            guide_input: (B, S, D) optional raw guidance signal.
            guidance_mask: (B, G) int mask (1=keep, 0=drop for CFG).

        Returns:
            (B, W, max_H*max_A) predicted velocity field.
        """
        max_H, max_A = self.max_horizon, self.max_dofs
        embeddings = self._embed(transformer_outputs, train=train)

        # During init provide zero dummies
        if self.is_initializing():
            B = embeddings.shape[0]
            # embeddings is (B, W, E) for pooled or (B, W, N, E) for "pass".
            W = embeddings.shape[1] if embeddings.ndim >= 3 else 1
            dof_ids = jnp.zeros((B, max_A), dtype=jnp.int32)
            chunk_steps = jnp.zeros((B, max_H), dtype=jnp.float32)
            slot_pos = jnp.zeros((B, max_A), dtype=jnp.float32)
            time = jnp.zeros((B, W, 1))
            a_t = jnp.zeros((B, W, max_H, max_A))
            if self.use_guidance and guide_input is None:
                guide_input = jnp.zeros((B, 1, self.guidance_input_dim), dtype=embeddings.dtype)
        elif time is None or a_t is None or dof_ids is None or chunk_steps is None:
            raise ValueError("Must provide time, a_t, dof_ids, chunk_steps")

        guidance_tokens = self._encode_guidance(guide_input, train=train)

        if slot_pos is None:
            slot_pos = jnp.broadcast_to(
                jnp.arange(max_A, dtype=jnp.float32),
                dof_ids.shape,
            )

        if a_t.ndim == 3:
            a_t = rearrange(a_t, "b w (h a) -> b w h a", h=max_H, a=max_A)

        B, W = time.shape[:2]

        # Merge batch and window: (B, W, ...) -> (BW, ...)
        embed_bw = rearrange(embeddings, "b w ... -> (b w) ...")
        if embed_bw.ndim == 2:
            embed_bw = embed_bw[:, None, :]  # (BW, 1, E)
        time_bw = rearrange(time, "b w t -> (b w) t")
        a_t_bw = rearrange(a_t, "b w h a -> (b w) h a")

        # Tile per-sample specs across window: (B, ...) -> (BW, ...)
        dof_bw = jnp.repeat(dof_ids, W, axis=0)
        chunk_bw = jnp.repeat(chunk_steps, W, axis=0)
        slot_bw = jnp.repeat(slot_pos, W, axis=0)

        # Build conditioned queries
        queries = self._build_queries(chunk_bw, dof_bw, slot_bw, time_bw, a_t_bw)

        # Query mask from padding — zero out padded queries
        q_mask = build_query_mask(chunk_bw, dof_bw, slot_bw)  # (BW, max_H*max_A)
        queries = queries * q_mask[..., None]

        # Cross-attention mask (always applied — masks padded queries)
        S = embed_bw.shape[1]
        context = embed_bw
        kv_mask = jnp.ones((B * W, S), dtype=jnp.int32)

        if guidance_tokens is not None:
            G = guidance_tokens.shape[1]
            guide_bw = jnp.tile(guidance_tokens[:, None], (1, W, 1, 1))
            guide_bw = rearrange(guide_bw, "b w g e -> (b w) g e")
            context = jnp.concatenate([context, guide_bw], axis=1)

            if guidance_mask is not None:
                g_mask_bw = jnp.tile(guidance_mask[:, None], (1, W, 1))
                g_mask_bw = rearrange(g_mask_bw, "b w g -> (b w) g")
            else:
                g_mask_bw = jnp.ones((B * W, G), dtype=jnp.int32)
            kv_mask = jnp.concatenate([kv_mask, g_mask_bw], axis=1)

        attention_mask = make_cross_attention_mask(
            q_mask.astype(jnp.int32),
            kv_mask,
        )

        # Decode
        decoded = self.decoder(
            queries,
            context,
            deterministic=not train,
            attention_mask=attention_mask,
        )

        # Scalar output per query, flatten
        output = self.output_proj(decoded).squeeze(-1)  # (BW, max_H*max_A)
        return rearrange(output, "(b w) q -> b w q", b=B, w=W)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        dof_ids: ArrayLike,
        chunk_steps: ArrayLike,
        slot_pos: ArrayLike | None = None,
        train: bool = True,
        guide_input: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute flow matching loss.

        Args:
            transformer_outputs: dict with readout_key -> TokenGroup (B, W, N, E).
            actions: (B, W, max_H, max_A) padded ground-truth actions.
            dof_ids: (B, max_A) MASK-padded DOF vocab IDs.
            chunk_steps: (B, max_H) padded temporal positions.
            slot_pos: (B, max_A) float — ordinal position (optional, defaults to arange).
        """
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        rng = self.make_rng("dropout")
        base_key, time_key = jax.random.split(rng)
        base = self.base_std * jax.random.normal(base_key, actions_flat.shape)

        time = sample_tau(time_key, shape=(*actions_flat.shape[:2], 1), s=0.99)
        blended = time * actions_flat + (1.0 - time) * base
        target = actions_flat - base

        pred = self(
            transformer_outputs,
            time=time,
            a_t=blended,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            slot_pos=slot_pos,
            train=train,
            guide_input=guide_input,
            guidance_mask=guidance_mask,
        )

        # Loss mask from padding — broadcast across window dim
        q_mask = build_query_mask(chunk_steps, dof_ids, slot_pos)  # (B, max_H*max_A)
        mask = jnp.broadcast_to(q_mask[:, None, :], pred.shape)

        loss, metrics = continuous_loss(pred, target, mask, loss_type=self.loss_type)
        return loss * self.loss_weight, metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        rng: PRNGKey,
        dof_ids: ArrayLike,
        chunk_steps: ArrayLike,
        slot_pos: ArrayLike | None = None,
        train: bool = False,
        *args,
        sample_shape: tuple[int, ...] = (),
        guide_input: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
        cfg_scale: float | None = None,
        accumulate: bool = False,
        **kwargs,
    ) -> Array:
        """Predict actions by solving the flow ODE (Euler integration).

        Args:
            N: arbitrary leading sample dimensions.
            accumulate: If True, return full trajectory with shape
                (*N, F+1, B, W, max_H, max_A) where F = flow_steps.
                If False (default), return only final prediction
                (*N, B, W, max_H, max_A).
        """
        module, variables = self.unbind()
        max_H, max_A = self.max_horizon, self.max_dofs

        # Ensure guidance_mask is explicit so zeros_like works in CFG path
        if cfg_scale is not None and guidance_mask is None and guide_input is not None:
            G = self.num_guidance_latents if self.compress_guidance else guide_input.shape[1]
            guidance_mask = jnp.ones((guide_input.shape[0], G), dtype=jnp.int32)

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            tokens = transformer_outputs[self.readout_key].tokens
            batch_size, window_size = tokens.shape[:2]

            a_t = self.base_std * jax.random.normal(
                key,
                (batch_size, window_size, max_H * max_A),
            )
            dt = 1.0 / max(self.flow_steps, 1)
            a_0 = a_t

            def _velocity(a_t, time_val):
                t = jnp.full((*a_t.shape[:2], 1), time_val, dtype=a_t.dtype)
                return module.apply(
                    variables,
                    transformer_outputs,
                    t,
                    a_t,
                    dof_ids=dof_ids,
                    chunk_steps=chunk_steps,
                    slot_pos=slot_pos,
                    train=train,
                    guide_input=guide_input,
                    guidance_mask=guidance_mask,
                )

            def scan_fn(a_t, step):
                time_val = (step + 0.5) * dt

                if cfg_scale is not None:
                    v_cond = _velocity(a_t, time_val)
                    zero_mask = jnp.zeros_like(guidance_mask) if guidance_mask is not None else None
                    v_uncond = module.apply(
                        variables,
                        transformer_outputs,
                        jnp.full((*a_t.shape[:2], 1), time_val, dtype=a_t.dtype),
                        a_t,
                        dof_ids=dof_ids,
                        chunk_steps=chunk_steps,
                        slot_pos=slot_pos,
                        train=train,
                        guide_input=guide_input,
                        guidance_mask=zero_mask,
                    )
                    velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    velocity = _velocity(a_t, time_val)

                updated = a_t + dt * velocity
                if self.clip_pred:
                    updated = jnp.clip(updated, -self.max_action, self.max_action)
                return updated, updated if accumulate else ()

            steps = jnp.arange(self.flow_steps)
            a_t, history = jax.lax.scan(scan_fn, a_t, steps)

            if accumulate:
                source = jnp.concatenate([a_0[None, ...], history], axis=0)
                fmt = "f b w (h a) -> f b w h a"
            else:
                source = a_t
                fmt = "b w (h a) -> b w h a"

            actions = rearrange(source, fmt, h=max_H, a=max_A)
            if self.clip_pred:
                actions = jnp.clip(actions, -self.max_action, self.max_action)
            return actions

        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        samples = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        return samples.reshape(sample_shape + samples.shape[1:])
