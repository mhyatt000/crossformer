"""L1 and MSE action heads."""

from __future__ import annotations

from einops import rearrange
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.model.components.base import TokenGroup

from .base import ContinuousActionHead
from .dof import build_query_mask
from .losses import continuous_loss


class L1ActionHead(ContinuousActionHead):
    """Action head using L1 loss (mean absolute error)."""

    loss_type: str = "l1"


class MSEActionHead(ContinuousActionHead):
    """Action head using MSE loss with MAP pooling."""

    loss_type: str = "mse"
    pool_strategy: str = "use_map"


class BundledMSEHead(ContinuousActionHead):
    """ContinuousActionHead adapted to the bundled-action training contract.

    Same loss signature as XFlowHead.loss so make_train_step needs no changes:
    padding comes from dof_ids/chunk_steps (build_query_mask), per-slot
    supervision from mask_act, and per-DOF weights from dof_weights. Guidance
    args are accepted and ignored. Direct regression — no flow matching — which
    makes this the head-isolation baseline: loss ~1.0 = mean predictor,
    loss -> 0 on a pinned batch = trunk + data + optimizer are healthy.
    """

    loss_type: str = "mse"
    pool_strategy: str = "mean"

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        dof_ids: ArrayLike,
        chunk_steps: ArrayLike,
        slot_pos: ArrayLike | None = None,
        view_ids: ArrayLike | None = None,
        train: bool = True,
        guide_input: ArrayLike | None = None,
        guidance_mask: ArrayLike | None = None,
        mask_act: ArrayLike | None = None,
        dof_weights: ArrayLike | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        """Regression loss on bundled actions.

        Args:
            transformer_outputs: dict with readout_key -> TokenGroup (B, W, N, E).
            actions: (B, W, max_H, max_A) padded ground-truth actions.
            dof_ids: (B, max_A) MASK-padded DOF vocab IDs.
            chunk_steps: (B, max_H) padded temporal positions (CHUNK_PAD marks padding).
            slot_pos: (B, max_A) float ordinal slot position (optional).
            view_ids / guide_input / guidance_mask: accepted for signature parity; unused.
            mask_act: (B, max_A) bool per-slot supervision mask (optional).
            dof_weights: (VOCAB_SIZE,) per-DOF loss weights (optional, relative).
        """
        del view_ids, guide_input, guidance_mask
        actions_flat = rearrange(jnp.asarray(actions), "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        pred = self(transformer_outputs, train=train)  # (B, W, H, A)
        pred_flat = rearrange(pred, "b w h a -> b w (h a)")

        # Same mask construction as XFlowHead.loss — one variable differs: the head.
        q_mask = build_query_mask(chunk_steps, dof_ids, slot_pos)  # (B, max_H*max_A)
        if mask_act is not None:
            act_mask = jnp.broadcast_to(
                jnp.asarray(mask_act, dtype=bool)[:, None, :],
                (q_mask.shape[0], self.action_horizon, self.action_dim),
            )
            q_mask = q_mask & rearrange(act_mask, "b h a -> b (h a)")

        weights = q_mask.astype(pred_flat.dtype)
        if dof_weights is not None:
            w = jnp.asarray(dof_weights, dtype=pred_flat.dtype)[jnp.asarray(dof_ids)]  # (B, max_A)
            w = jnp.broadcast_to(w[:, None, :], (w.shape[0], self.action_horizon, self.action_dim))
            weights = weights * rearrange(w, "b h a -> b (h a)")
        mask = jnp.broadcast_to(weights[:, None, :], pred_flat.shape)

        loss, metrics = continuous_loss(pred_flat, actions_flat, mask, loss_type=self.loss_type)
        return loss * self.loss_weight, metrics
