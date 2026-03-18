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
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.mytyping import PRNGKey

from .base import ActionHead
from .dof import FactoredQueryEncoding
from .io.attention import CrossAttention, make_cross_attention_mask, SelfAttention
from .losses import continuous_loss, sample_tau


class PerceiverDecoder(nn.Module):
    """Perceiver IO decoder block: cross-attend queries against context, then self-attend.

    Args:
        num_heads: heads for both cross- and self-attention.
        num_self_attend_layers: self-attention depth after the cross-attention.
        widening_factor: MLP expansion ratio inside attention blocks.
        dropout_prob: dropout rate for attention and MLP.
        qk_channels: override QK projection dim (default: inferred from inputs).
        v_channels: override V projection dim.
    """

    num_heads: int = 8
    num_self_attend_layers: int = 2
    widening_factor: int = 4
    dropout_prob: float = 0.0
    qk_channels: int | None = None
    v_channels: int | None = None

    @nn.compact
    def __call__(self, queries, context, *, deterministic=True, attention_mask=None):
        """Cross-attend queries against context, then refine with self-attention.

        Args:
            queries: (batch, seq_q, d_q)
            context: (batch, seq_kv, d_kv)
            attention_mask: (batch, seq_q, seq_kv) bool mask for cross-attention.

        Returns:
            (batch, seq_q, d_q)
        """
        x = CrossAttention(
            num_heads=self.num_heads,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            use_query_residual=True,
            shape_for_attn="kv",
            name="cross_attend",
        )(queries, context, attention_mask=attention_mask, deterministic=deterministic)

        for i in range(self.num_self_attend_layers):
            x = SelfAttention(
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
                dropout_prob=self.dropout_prob,
                qk_channels=self.qk_channels,
                v_channels=self.v_channels,
                name=f"self_attend_{i}",
            )(x, deterministic=deterministic)

        return x


class XFlowHead(nn.Module, ActionHead):
    """Perceiver IO flow-matching head for cross-embodied action prediction.

    Uses factored output queries (Fourier chunk position x learned DOF embedding)
    that cross-attend to transformer embeddings (+ optional guidance tokens).
    Each query predicts one scalar: the velocity for a single (timestep, DOF) pair.

    Shape contract:
        transformer tokens: (B, W, N, E) — batch, window, num_tokens, embed_dim
        actions:            (B, W, H, A) — batch, window, action_horizon, action_dim
        guidance_tokens:    (B, G, E)    — optional extra conditioning for CFG
    """

    # Identity
    readout_key: str

    # Action space — defined by DOF vocabulary + temporal chunk steps
    dof_ids: tuple[int, ...]  # DOF vocab IDs, e.g. ids("j0",..."gripper")
    chunk_steps: tuple[float, ...]  # temporal positions, e.g. chunk_range(20)
    clip_pred: bool = True
    max_action: float = 5.0
    loss_type: str = "mse"
    loss_weight: float = 1.0
    constrain_loss_dims: bool = True

    # Perceiver decoder
    num_query_channels: int = 256
    num_heads: int = 8
    num_self_attend_layers: int = 2
    widening_factor: int = 4
    dropout_prob: float = 0.1

    # Flow matching
    time_dim: int = 32
    flow_steps: int = 10
    base_std: float = 1.0

    # Pooling strategy for transformer outputs
    pool_strategy: str = "mean"

    @property
    def action_dim(self) -> int:
        return len(self.dof_ids)

    @property
    def action_horizon(self) -> int:
        return len(self.chunk_steps)

    def setup(self):
        D = self.num_query_channels

        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        # Factored queries: Fourier chunk position x learned DOF embedding
        self.query_pos_enc = FactoredQueryEncoding(
            chunk_steps=self.chunk_steps,
            dof_ids=self.dof_ids,
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
            num_self_attend_layers=self.num_self_attend_layers,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            name="decoder",
        )
        self.output_proj = nn.Dense(1, name="output_proj")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, transformer_outputs: dict[str, TokenGroup], train: bool) -> Array:
        """Pool transformer output tokens to (B, W, E)."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4
        if self.pool_strategy == "use_map":
            return self.map_head(token_group, train=train)[:, :, 0]
        if self.pool_strategy == "mean":
            return token_group.tokens.mean(axis=-2)
        if self.pool_strategy == "pass":
            # Flatten tokens into the sequence dim: (B, W, N, E) -> (B, W*N, E)
            t = token_group.tokens
            return rearrange(t, "b w n e -> b (w n) e")
        raise ValueError(f"{self.pool_strategy} not implemented!")

    def _build_queries(self, batch_size: int, time: ArrayLike, a_t: ArrayLike) -> Array:
        """Build factored conditioned queries: (chunk ⊕ dof) + action + time.

        Args:
            batch_size: B*W (merged batch and window dims).
            time: (BW, 1) flow timestep.
            a_t: (BW, H, A) noisy actions.

        Returns:
            (BW, H*A, num_query_channels)
        """
        # Factored positional encoding
        pos_q = self.query_pos_enc(batch_size=batch_size)  # (BW, H*A, D)

        # Per-scalar action conditioning: (BW, H, A) → (BW, H*A, 1) → Fourier → Dense
        a_flat = rearrange(a_t, "bw h a -> bw (h a) 1")
        act_cond = self.action_proj(self.action_features(a_flat))  # (BW, H*A, D)

        # Time conditioning — broadcast across all queries
        t_cond = self.time_proj(self.time_features(time))[:, None, :]  # (BW, 1, D)

        return pos_q + act_cond + t_cond

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        a_t: ArrayLike | None = None,
        train: bool = True,
        guidance_tokens: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
    ) -> Array:
        """Predict action velocities.

        Args:
            transformer_outputs: dict mapping readout_key -> TokenGroup with
                tokens of shape (B, W, N, E).
            time: (B, W, 1) flow timestep in [0, 1].
            a_t: (B, W, H, A) or (B, W, H*A) current noisy actions.
            guidance_tokens: (B, G, E) optional extra conditioning tokens.
            guidance_mask: (B, G) int mask (1=keep, 0=drop for CFG).

        Returns:
            (B, W, H*A) predicted velocity field.
        """
        embeddings = self._embed(transformer_outputs, train=train)  # (B, W, E) or (B, S, E)

        # During init provide zero dummies
        if (time is None or a_t is None) and not self.is_initializing():
            raise ValueError("Must provide time and a_t when calling XFlowHead")
        if self.is_initializing():
            B = embeddings.shape[0]
            W = embeddings.shape[1] if embeddings.ndim == 3 else 1
            time = jnp.zeros((B, W, 1))
            a_t = jnp.zeros((B, W, self.action_horizon, self.action_dim))

        # Flatten actions to (B, W, H, A) if needed
        if a_t.ndim == 3:
            a_t = rearrange(a_t, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim)

        B, W = time.shape[:2]
        E = embeddings.shape[-1]

        # Merge batch and window: (B, W, ...) -> (BW, ...)
        embed_bw = rearrange(embeddings, "b w ... -> (b w) ...")  # (BW, ..., E)
        # If pool was "mean", embed_bw is (BW, E) — unsqueeze to (BW, 1, E) for cross-attn
        if embed_bw.ndim == 2:
            embed_bw = embed_bw[:, None, :]  # (BW, 1, E)
        time_bw = rearrange(time, "b w t -> (b w) t")  # (BW, 1)
        a_t_bw = rearrange(a_t, "b w h a -> (b w) h a")  # (BW, H, A)

        # Build conditioned queries
        queries = self._build_queries(B * W, time_bw, a_t_bw)  # (BW, H*A, D)

        # Build context: transformer embeddings + optional guidance tokens
        context = embed_bw  # (BW, S, E)
        attention_mask = None

        if guidance_tokens is not None:
            G = guidance_tokens.shape[1]
            # Repeat guidance for each window step: (B, G, E) -> (BW, G, E)
            guide_bw = jnp.repeat(guidance_tokens, W, axis=0) if W > 1 else guidance_tokens
            # Tile properly: (B, G, E) -> (B, 1, G, E) -> (B, W, G, E) -> (BW, G, E)
            guide_bw = jnp.tile(guidance_tokens[:, None], (1, W, 1, 1))
            guide_bw = rearrange(guide_bw, "b w g e -> (b w) g e")
            context = jnp.concatenate([context, guide_bw], axis=1)  # (BW, S+G, E)

            # Build cross-attention mask
            S = embed_bw.shape[1]
            ctx_mask = jnp.ones((B * W, S), dtype=jnp.int32)

            if guidance_mask is not None:
                # (B, G) -> (BW, G)
                g_mask_bw = jnp.tile(guidance_mask[:, None], (1, W, 1))
                g_mask_bw = rearrange(g_mask_bw, "b w g -> (b w) g")
            else:
                g_mask_bw = jnp.ones((B * W, G), dtype=jnp.int32)

            kv_mask = jnp.concatenate([ctx_mask, g_mask_bw], axis=1)  # (BW, S+G)
            query_mask = jnp.ones((B * W, self.action_horizon * self.action_dim), dtype=jnp.int32)
            attention_mask = make_cross_attention_mask(query_mask, kv_mask)

        # Decode
        decoded = self.decoder(
            queries,
            context,
            deterministic=not train,
            attention_mask=attention_mask,
        )  # (BW, H*A, D)

        # Scalar output per query, flatten
        output = self.output_proj(decoded).squeeze(-1)  # (BW, H*A)
        output = rearrange(output, "(b w) q -> b w q", b=B, w=W)  # (B, W, H*A)
        return output

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
        guidance_tokens: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute flow matching loss.

        Args:
            transformer_outputs: dict with readout_key -> TokenGroup (B, W, N, E).
            actions: (B, W, H, A) ground-truth actions.
            timestep_pad_mask: (B, W) bool mask for valid timesteps.
            action_pad_mask: (B, W, H, A) bool mask for valid action dims.
            action_head_mask: (B,) bool mask for embodiments using this head.
            guidance_tokens: (B, G, E) optional guidance for CFG.
            guidance_mask: (B, G) mask for guidance dropout.
        """
        if self.constrain_loss_dims:
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

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
            train=train,
            guidance_tokens=guidance_tokens,
            guidance_mask=guidance_mask,
        )

        if action_head_mask is None:
            action_head_mask = jnp.ones(pred.shape[0], dtype=bool)

        mask = rearrange(
            timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None],
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred, target, mask, loss_type=self.loss_type)
        return loss * self.loss_weight, metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = False,
        *args,
        sample_shape: tuple[int, ...] = (),
        guidance_tokens: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
        cfg_scale: float | None = None,
        embodiment_action_dim: int | None = None,
        **kwargs,
    ) -> Array:
        """Predict actions by solving the flow ODE (Euler integration).

        Args:
            cfg_scale: if set, run classifier-free guidance with this scale.
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
        """
        module, variables = self.unbind()

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            tokens = transformer_outputs[self.readout_key].tokens
            batch_size, window_size = tokens.shape[:2]

            a_t = self.base_std * jax.random.normal(
                key,
                (batch_size, window_size, self.action_horizon * self.action_dim),
            )
            dt = 1.0 / max(self.flow_steps, 1)

            def _velocity(a_t, time_val):
                t = jnp.full((*a_t.shape[:2], 1), time_val, dtype=a_t.dtype)
                return module.apply(
                    variables,
                    transformer_outputs,
                    t,
                    a_t,
                    train=train,
                    guidance_tokens=guidance_tokens,
                    guidance_mask=guidance_mask,
                )

            def scan_fn(a_t, step):
                time_val = (step + 0.5) * dt

                if cfg_scale is not None:
                    v_cond = _velocity(a_t, time_val)
                    # Unconditioned: zero out guidance mask
                    zero_mask = jnp.zeros_like(guidance_mask) if guidance_mask is not None else None
                    v_uncond = module.apply(
                        variables,
                        transformer_outputs,
                        jnp.full((*a_t.shape[:2], 1), time_val, dtype=a_t.dtype),
                        a_t,
                        train=train,
                        guidance_tokens=guidance_tokens,
                        guidance_mask=zero_mask,
                    )
                    velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    velocity = _velocity(a_t, time_val)

                updated = a_t + dt * velocity
                if self.clip_pred:
                    updated = jnp.clip(updated, -self.max_action, self.max_action)
                return updated, ()

            steps = jnp.arange(self.flow_steps)
            a_t, _ = jax.lax.scan(scan_fn, a_t, steps)

            actions = rearrange(
                a_t,
                "b w (h a) -> b w h a",
                h=self.action_horizon,
                a=self.action_dim,
            )
            if self.clip_pred:
                actions = jnp.clip(actions, -self.max_action, self.max_action)
            return actions

        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        samples = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        return samples.reshape(sample_shape + samples.shape[1:])
