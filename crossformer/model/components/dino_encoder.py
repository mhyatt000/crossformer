"""DINOv3 vision encoder — linen adapter over a frozen bonsai nnx model.

Weights stay outside the linen param tree: load_dino() caches the nnx
(graphdef, state), and the encoder applies jax.lax.stop_gradient to the state
before every forward. This means zero grads, zero optimizer entries, and zero
checkpoint bloat for the pretrained trunk.

Why not flax.nnx.bridge.ToLinen:
    ToLinen registers every nnx parameter as a linen param. For a frozen
    pretrained ViT that's pure overhead — every checkpoint ships a copy of
    weights we never train, and the optimizer allocates state for them unless
    we build an explicit mask. The closure pattern below avoids all of that.
"""

from __future__ import annotations

from typing import Any, Sequence

from bonsai.models.dinov3 import modeling
from flax import linen as nn
from flax import nnx
import jax
from jax import Array
import jax.numpy as jnp

from crossformer.model.components.base import TokenGroup

MODEL_ID_DEFAULT = "facebook/dinov3-vits16-pretrain-lvd1689m"

_IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
_IMAGENET_STD = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)


def _preprocess(images: Array, target_size: tuple[int, int] | None = None) -> Array:
    """(N, H, W, C) uint8 or float[0,1] -> (N, C, H', W') imagenet-normalized.

    Resize (bilinear + antialias) happens after [0,1] conversion and before
    imagenet normalization, so aliasing is controlled in pixel space.
    """
    x = images.astype(jnp.float32)
    if jnp.issubdtype(images.dtype, jnp.integer):
        x = x / 255.0
    if target_size is not None and (x.shape[1], x.shape[2]) != target_size:
        h, w = target_size
        x = jax.image.resize(x, (x.shape[0], h, w, x.shape[-1]), method="bilinear", antialias=True)
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return jnp.transpose(x, (0, 3, 1, 2))


_DINO_CACHE: dict[str, tuple[Any, Any]] = {}


def load_dino(model_id: str = MODEL_ID_DEFAULT) -> tuple[Any, Any]:
    """Load DINOv3 once; return (static graphdef, array state) on the default
    JAX device.

    Bonsai's from_pretrained returns numpy arrays; without the device_put every
    forward would pay a CPU->GPU transfer (or fall back to CPU execution).
    Prime before jit so the HF download doesn't happen inside a trace.
    """
    if model_id not in _DINO_CACHE:
        dino = modeling.Dinov3ViTModel.from_pretrained(model_id)
        graphdef, state = nnx.split(dino)
        state = jax.tree.map(jnp.asarray, state)
        _DINO_CACHE[model_id] = (graphdef, state)
    return _DINO_CACHE[model_id]


def shard_dino(sharding, model_id: str = MODEL_ID_DEFAULT) -> None:
    """Replicate DINOv3 state across a jax.sharding.

    DINOv3 weights live outside the linen param tree (by design — zero
    checkpoint/optimizer overhead for a frozen trunk). That also means the
    existing `jax.device_put(model.params, replicated)` call does NOT touch
    them. Call this once after creating the mesh to place the state on every
    device, otherwise forwards on non-zero devices cross-device-copy every call.
    """
    graphdef, state = load_dino(model_id)
    state = jax.tree.map(lambda x: jax.device_put(x, sharding), state)
    _DINO_CACHE[model_id] = (graphdef, state)


class DinoV3Encoder(nn.Module):
    """Frozen DINOv3 ViT as a linen image encoder.

    Output: (N, 1 + num_register_tokens + P, E).
    For dinov3-vits16 at the default 240x320 target: 1 + 4 + 15*20 = 305 tokens,
    E = 384.

    The encoder owns its preferred input shape via `target_size`. Any (H, W)
    divisible by patch_size=16 works; bilinear+antialias resize is applied
    inside _preprocess. Set `target_size=None` to disable and trust the caller.

    Call load_dino(model_id) once at startup to prime the cache so the HF
    download does not happen inside a jit trace.
    """

    model_id: str = MODEL_ID_DEFAULT
    patch_only: bool = False
    num_prefix_tokens: int = 5
    target_size: tuple[int, int] | None = (240, 320)

    def __call__(self, images: Array, train: bool = False) -> Array:
        graphdef, state = load_dino(self.model_id)
        state = jax.tree.map(jax.lax.stop_gradient, state)
        model = nnx.merge(graphdef, state)
        out = model(_preprocess(images, self.target_size))
        tokens = out["last_hidden_state"]
        if self.patch_only:
            tokens = tokens[:, self.num_prefix_tokens :, :]
        return tokens


class DinoTokenizer(nn.Module):
    """Per-view DINOv3 tokenizer — one encode per camera, tokens concatenated.

    For each key in image_keys:
        observations[key]: (B, T, H, W, C) -> DINOv3 -> (B, T, N, E)
    Concatenate along the token axis: (B, T, len(image_keys) * N, E).

    Per-view padding comes from observations["pad_mask_dict"][key] and is
    broadcast across the view's N tokens before concatenation — so the mask
    still tags which specific view each token came from.

    DINOv3 weights are shared across views (one frozen trunk regardless of
    how many cameras).
    """

    image_keys: Sequence[str]
    model_id: str = MODEL_ID_DEFAULT
    patch_only: bool = False
    num_prefix_tokens: int = 5
    # Train-time augmentations (applied to the output mask, not the tokens).
    key_drop_prob: float = 0.1  # per-sample chance of masking out an entire view
    patch_drop_prob: float = 0.1  # per-token chance of masking a patch

    @nn.compact
    def __call__(self, observations, tasks=None, train: bool = False) -> TokenGroup:
        encoder = DinoV3Encoder(
            model_id=self.model_id,
            patch_only=self.patch_only,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        pad_mask_dict = observations.get("pad_mask_dict", None)

        view_tokens, view_masks = [], []
        for key in self.image_keys:
            imgs = observations[key]
            b, t, h, w, c = imgs.shape
            tokens = encoder(imgs.reshape(b * t, h, w, c), train=train)
            tokens = tokens.reshape(b, t, tokens.shape[-2], tokens.shape[-1])
            view_tokens.append(tokens)

            if pad_mask_dict is not None and key in pad_mask_dict:
                view_mask = pad_mask_dict[key]
            else:
                view_mask = jnp.ones((b, t), dtype=bool)
            view_masks.append(jnp.broadcast_to(view_mask[..., None], tokens.shape[:-1]))

        V = len(view_tokens)
        B, T = view_tokens[0].shape[:2]
        N, E = view_tokens[0].shape[-2], view_tokens[0].shape[-1]

        # Permute view order along the keys axis only — NOT along B, T, or within a
        # view's N patches. One permutation per forward; all batch samples get the
        # same view ordering this step.
        if train and V > 1:
            perm = jax.random.permutation(self.make_rng("dropout"), V)
            stacked_t = jnp.stack(view_tokens, axis=0)[perm]  # (V, B, T, N, E)
            stacked_m = jnp.stack(view_masks, axis=0)[perm]  # (V, B, T, N)
            tokens = jnp.transpose(stacked_t, (1, 2, 0, 3, 4)).reshape(B, T, V * N, E)
            mask = jnp.transpose(stacked_m, (1, 2, 0, 3)).reshape(B, T, V * N)
        else:
            tokens = jnp.concatenate(view_tokens, axis=-2)
            mask = jnp.concatenate(view_masks, axis=-1)

        # Train-time mask augmentations.
        if train:
            if self.patch_drop_prob > 0.0:
                rng = self.make_rng("dropout")
                keep = jax.random.uniform(rng, mask.shape) >= self.patch_drop_prob
                mask = mask & keep
            if self.key_drop_prob > 0.0:
                rng = self.make_rng("dropout")
                # Per-sample, per-view: (B, V). Broadcast across T, repeat across N.
                view_keep = jax.random.uniform(rng, (B, V)) >= self.key_drop_prob
                view_keep = jnp.repeat(view_keep, N, axis=-1)  # (B, V*N)
                view_keep = jnp.broadcast_to(view_keep[:, None, :], (B, T, V * N))
                mask = mask & view_keep

        return TokenGroup(tokens, mask)
