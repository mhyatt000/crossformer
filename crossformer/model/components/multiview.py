"""Stacked multiview image tokenizers.

Consume the stacked ``observations["image"]`` array produced by the grain
pipeline (``fix_views`` orders views and pads to MAX_VIEWS) and emit a single
TokenGroup whose per-token ``view`` ids (1..V) travel with the tokens through
the train-time view permutation. 0 = NO_VIEW is reserved for global tokens.

Encoders:
  StackedTipsTokenizer — TIPS v2 ViT (default). The trunk lives in the linen
      param tree under the module name "tips" so pretrained weights load with
      ``tips_checkpoint.load_checkpoint`` (same pattern as DreamTIPS in
      crossformer/model/dream.py); ``freeze=True`` stops gradients without
      removing the params.
  StackedDinoTokenizer — frozen DINOv3 trunk via the load_dino closure
      (weights outside the param tree; see dino_encoder.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.dino_encoder import DinoV3Encoder, MODEL_ID_DEFAULT

TIPS_VARIANT_DEFAULT = "tips_v2_b14"


def _stack_views(
    view_tokens: list[Array],
    view_masks: list[Array],
    *,
    rng: Array | None,
) -> tuple[Array, Array, Array]:
    """Merge per-view token lists into one flattened sequence with view ids.

    Args:
        view_tokens: V arrays of (B, T, N, E).
        view_masks: V arrays of (B, T, N) bool.
        rng: when given, apply one random view permutation (per forward, shared
            across the batch) to tokens, masks, and view ids together — so the
            positional embedding added downstream cannot bind position to view,
            while view identity stays attached to its tokens.

    Returns:
        tokens (B, T, V*N, E), mask (B, T, V*N), view (B, T, V*N) int32 in 1..V.
    """
    V = len(view_tokens)
    B, T, N, E = view_tokens[0].shape

    tokens = jnp.stack(view_tokens, axis=0)  # (V, B, T, N, E)
    mask = jnp.stack(view_masks, axis=0)  # (V, B, T, N)
    view = jnp.broadcast_to(jnp.arange(1, V + 1, dtype=jnp.int32)[:, None, None, None], (V, B, T, N))

    if rng is not None and V > 1:
        perm = jax.random.permutation(rng, V)
        tokens, mask, view = tokens[perm], mask[perm], view[perm]

    tokens = jnp.transpose(tokens, (1, 2, 0, 3, 4)).reshape(B, T, V * N, E)
    mask = jnp.transpose(mask, (1, 2, 0, 3)).reshape(B, T, V * N)
    view = jnp.transpose(view, (1, 2, 0, 3)).reshape(B, T, V * N)
    return tokens, mask, view


def _split_stacked(observations: dict) -> tuple[Array, Array]:
    """Return (images (B, T, V, H, W, C), view_mask (B, T, V) bool).

    ``view_mask`` comes from ``observations["view_mask"]`` (data-side
    ``mask.view`` — which stacked slots hold a real camera vs zero padding);
    defaults to all-real when absent.
    """
    imgs = observations["image"]
    if imgs.ndim == 5:  # (B, V, H, W, C) — no window dim
        imgs = imgs[:, None]
    assert imgs.ndim == 6, f"expected stacked image (B, T, V, H, W, C), got {imgs.shape}"
    B, T, V = imgs.shape[:3]

    vm = observations.get("view_mask")
    if vm is None:
        view_mask = jnp.ones((B, T, V), dtype=bool)
    else:
        vm = jnp.asarray(vm).astype(bool).reshape(B, -1, V)
        view_mask = jnp.broadcast_to(vm[:, :1], (B, T, V)) if vm.shape[1] != T else vm
    return imgs, view_mask


class _StackedViewTokenizer(nn.Module):
    """Shared skeleton: per-view encode -> permute -> flatten -> mask augments."""

    permute_views: bool = True
    # Train-time augmentations (applied to the output mask, not the tokens).
    key_drop_prob: float = 0.1  # per-sample chance of masking out an entire view
    patch_drop_prob: float = 0.1  # per-token chance of masking a patch

    def make_encoder(self):
        """Return fn (N, H, W, C) -> (N, n_tokens, E). Called once per forward;
        the returned fn is invoked once per view (trunk shared across views)."""
        raise NotImplementedError

    @nn.compact
    def __call__(self, observations, tasks=None, train: bool = False) -> TokenGroup:
        imgs, view_mask = _split_stacked(observations)
        B, T, V = imgs.shape[:3]
        encode = self.make_encoder()

        view_tokens, view_masks = [], []
        for v in range(V):
            x = imgs[:, :, v]  # (B, T, H, W, C)
            tokens = encode(x.reshape(B * T, *x.shape[2:]), train)
            tokens = tokens.reshape(B, T, tokens.shape[-2], tokens.shape[-1])
            view_tokens.append(tokens)
            view_masks.append(jnp.broadcast_to(view_mask[:, :, v, None], tokens.shape[:-1]))

        rng = self.make_rng("dropout") if train and self.permute_views and V > 1 else None
        tokens, mask, view = _stack_views(view_tokens, view_masks, rng=rng)
        N = tokens.shape[-2] // V

        if train:
            if self.patch_drop_prob > 0.0:
                keep = jax.random.uniform(self.make_rng("dropout"), mask.shape) >= self.patch_drop_prob
                mask = mask & keep
            if self.key_drop_prob > 0.0:
                # Per-sample, per-view slot: (B, V). Broadcast across T, repeat across N.
                view_keep = jax.random.uniform(self.make_rng("dropout"), (B, V)) >= self.key_drop_prob
                view_keep = jnp.repeat(view_keep, N, axis=-1)  # (B, V*N)
                mask = mask & jnp.broadcast_to(view_keep[:, None, :], mask.shape)

        return TokenGroup(tokens, mask, view=view)


class StackedTipsTokenizer(_StackedViewTokenizer):
    """TIPS v2 encoder over stacked views — the default multiview tokenizer.

    The trunk is a linen submodule named "tips": params live in the tree (load
    pretrained weights post-init via load_tips_params) and freeze=True applies
    stop_gradient so they never train.
    """

    variant: str = TIPS_VARIANT_DEFAULT
    freeze: bool = True

    def make_encoder(self):
        from tips.scenic.configs import tips_model_config
        from tips.scenic.models import tips

        cfg = tips_model_config.get_config(self.variant)
        enc = tips.VisionEncoder(
            variant=cfg.variant,
            pooling=cfg.pooling,
            num_cls_tokens=cfg.num_cls_tokens,
            posembs=tuple(cfg.positional_embedding.shape),
            name="tips",
        )

        def encode(frames: Array, train: bool) -> Array:
            x = frames.astype(jnp.float32)
            if jnp.issubdtype(frames.dtype, jnp.integer):
                x = x / 255.0
            spatial, _cls = enc(x, train=False)  # (N, fh, fw, E)
            if self.freeze:
                spatial = jax.lax.stop_gradient(spatial)
            return spatial.reshape(spatial.shape[0], -1, spatial.shape[-1])

        return encode


class StackedDinoTokenizer(_StackedViewTokenizer):
    """Frozen DINOv3 encoder over stacked views (weights outside the param tree)."""

    model_id: str = MODEL_ID_DEFAULT
    patch_only: bool = False
    num_prefix_tokens: int = 5
    target_size: tuple[int, int] | None = (240, 320)

    def make_encoder(self):
        encoder = DinoV3Encoder(
            model_id=self.model_id,
            patch_only=self.patch_only,
            num_prefix_tokens=self.num_prefix_tokens,
            target_size=self.target_size,
        )
        return lambda frames, train: encoder(frames, train=train)


def load_tips_params(params: dict, variant: str = TIPS_VARIANT_DEFAULT, checkpoint_path: str | Path | None = None):
    """Load pretrained TIPS weights into every "tips" subtree of a param tree.

    Call after model init (params must be unfrozen). Mirrors
    scripts/train/dream.py:load_tips_params but finds the subtree wherever the
    tokenizer landed in the module hierarchy.
    """
    from tips.scenic.utils import checkpoint as tips_checkpoint

    from crossformer.model.load import resolve_checkpoint_path

    path = resolve_checkpoint_path(variant, checkpoint_path)

    def walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        return {k: tips_checkpoint.load_checkpoint(path, v) if k == "tips" else walk(v) for k, v in node.items()}

    return walk(params)
