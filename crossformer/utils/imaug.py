from __future__ import annotations

import augmax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def rotate_image_np(image: np.ndarray, angle_deg: float, resample: int, fill: int = 0) -> np.ndarray:
    out = Image.fromarray(image).rotate(angle_deg, resample=resample, expand=False, fillcolor=fill)
    return np.asarray(out)


def rotate_keypoints_np(kp2d: np.ndarray, angle_deg: float, h: int, w: int) -> np.ndarray:
    a = np.deg2rad(np.float32(angle_deg))
    c, s = np.cos(a), np.sin(a)
    cx = np.float32(w) * 0.5
    cy = np.float32(h) * 0.5
    x = kp2d[..., 0] - cx
    y = kp2d[..., 1] - cy
    # PIL rotates CCW in screen / y-down coords: c*x + s*y, y' = -s*x + c*y
    return np.stack([c * x + s * y + cx, -s * x + c * y + cy], axis=-1).astype(np.float32)


augmax_color_chain = augmax.Chain(
    augmax.ChannelShuffle(p=0.5),
    augmax.RandomGrayscale(p=0.5),
    augmax.ChannelDrop(),
    augmax.Blur(),
    augmax.RandomBrightness((-1.0, 1.0), p=0.5),
    augmax.RandomContrast(),
    augmax.RandomGamma(),
    augmax.RandomChannelGamma(),
    augmax.ColorJitter(),
    augmax.Solarization(),
)


def apply_augmax_color(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply augmax color transforms to a single HWC uint8 image, returns uint8"""
    key = jax.random.key(rng.integers(2**32 - 1, dtype=np.uint32))
    img_f = jnp.asarray(image, dtype=jnp.float32) / 255.0
    img_f = _augmax_color_chain(key, img_f)
    return np.clip(np.asarray(img_f) * 255.0, 0, 255).astype(np.uint8)


@jax.jit
def apply_augmax_color_batch(keys: jax.Array, images: jax.Array) -> jax.Array:
    """Apply augmax color transforms to a NHWC float32 batch, returns float32."""
    return jax.vmap(_augmax_color_chain)(keys, images)


def translate_image_np(
    image: np.ndarray, tx: float, ty: float, resample: int = Image.BILINEAR, fill: int = 0
) -> np.ndarray:
    pil = Image.fromarray(image)
    pil = pil.transform(pil.size, Image.AFFINE, (1, 0, -tx, 0, 1, -ty), resample=resample, fillcolor=fill)
    return np.asarray(pil)


def zoom_image_np(image: np.ndarray, scale: float, resample: int = Image.BILINEAR, fill: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    inv_s = 1.0 / scale
    pil = Image.fromarray(image)
    pil = pil.transform(
        pil.size,
        Image.AFFINE,
        (inv_s, 0, cx * (1 - inv_s), 0, inv_s, cy * (1 - inv_s)),
        resample=resample,
        fillcolor=fill,
    )
    return np.asarray(pil)


def kp_in_bounds(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    return (kp2d[:, 0] >= 0.0) & (kp2d[:, 0] < np.float32(w)) & (kp2d[:, 1] >= 0.0) & (kp2d[:, 1] < np.float32(h))


def kp_render_mask(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    kp2d = np.asarray(kp2d)
    return np.isfinite(kp2d).all(axis=-1) & _kp_in_bounds(kp2d, h, w)


def plot_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
        return arr
    if arr.ndim == 3:
        return np.zeros((1, 1, arr.shape[2]), dtype=arr.dtype)
    return np.zeros((1, 1), dtype=arr.dtype)


def _maybe_apply_grain_imaug(ds, cfg: Config):
    if not (cfg.imaug or cfg.rotate):
        return ds

    def aug(batch: dict, rng):
        state = _make_aug_state(batch)

        if cfg.imaug:
            state.image = _apply_color_aug_np(state.image, rng)

        if cfg.rotate:
            _apply_geometric_aug_batch_inplace(state, rng, cfg.min_visible_kp)

        return _state_to_batch(batch, state)

    return ds.random_map(aug)


@dataclass
class AugState:
    image: np.ndarray
    mask: np.ndarray
    kp: np.ndarray
    vis: np.ndarray
    K: np.ndarray
    w2c: np.ndarray
    h: int
    w: int


def _make_aug_state(batch: dict) -> AugState:
    image = np.asarray(batch["image"]).copy()
    mask = np.asarray(batch["mask"]).copy()
    kp = np.asarray(batch["keypoints_2d_netin"], dtype=np.float32).copy()
    vis = np.asarray(batch["keypoints_visible"], dtype=bool).copy()
    K = np.asarray(batch["K"], dtype=np.float32).copy()
    w2c = np.asarray(batch["w2c"], dtype=np.float32).copy()
    h, w = image.shape[1:3]

    return AugState(image, mask, kp, vis, K, w2c, h, w)


def _state_to_batch(batch: dict, state: AugState) -> dict:
    return {
        **batch,
        "image": state.image,
        "mask": state.mask,
        "keypoints_2d_netin": state.kp,
        "keypoints_2d_norm": _normalize_kp2d_np(state.kp, h=state.h, w=state.w),
        "keypoints_visible": state.vis,
        "K": state.K,
        "w2c": state.w2c,
    }


def _commit_if_enough_visible(
    state: AugState,
    i: int,
    kp_i: np.ndarray,
    vis_i: np.ndarray,
    min_visible_kp: int,
    commit_fn,
) -> None:
    if int(vis_i.sum()) < min_visible_kp:
        return

    commit_fn()
    state.kp[i] = kp_i
    state.vis[i] = vis_i


def _maybe_rotate_sample(state: AugState, i: int, rng, min_visible_kp: int) -> None:
    if rng.random() >= 0.3:
        return

    angle = float(rng.uniform(-15.0, 15.0))
    kp_i = _rotate_keypoints_np(state.kp[i], angle, h=state.h, w=state.w)
    vis_i = state.vis[i] & _kp_in_bounds(kp_i, state.h, state.w)

    def commit():
        state.image[i] = _rotate_image_np(state.image[i], angle, Image.BILINEAR, fill=0)
        state.mask[i] = _rotate_image_np(
            state.mask[i].astype(np.uint8),
            angle,
            Image.NEAREST,
            fill=0,
        ).astype(state.mask.dtype)

        a = np.deg2rad(np.float32(-angle))
        ca, sa = np.cos(a), np.sin(a)

        Rz = np.eye(4, dtype=np.float32)
        Rz[0, 0] = ca
        Rz[0, 1] = -sa
        Rz[1, 0] = sa
        Rz[1, 1] = ca

        state.w2c[i] = Rz @ state.w2c[i]

    _commit_if_enough_visible(state, i, kp_i, vis_i, min_visible_kp, commit)


def _maybe_translate_sample(state: AugState, i: int, rng, min_visible_kp: int) -> None:
    if rng.random() >= 0.5:
        return

    tx = float(rng.uniform(-0.1 * state.w, 0.1 * state.w))
    ty = float(rng.uniform(-0.1 * state.h, 0.1 * state.h))

    kp_i = state.kp[i].copy()
    kp_i[:, 0] += tx
    kp_i[:, 1] += ty

    vis_i = state.vis[i] & _kp_in_bounds(kp_i, state.h, state.w)

    def commit():
        state.image[i] = _translate_image_np(
            state.image[i],
            tx,
            ty,
            resample=Image.BILINEAR,
            fill=0,
        )
        state.mask[i] = _translate_image_np(
            state.mask[i].astype(np.uint8),
            tx,
            ty,
            resample=Image.NEAREST,
            fill=0,
        ).astype(state.mask.dtype)

        state.K[i, 0, 2] += np.float32(tx)
        state.K[i, 1, 2] += np.float32(ty)

    _commit_if_enough_visible(state, i, kp_i, vis_i, min_visible_kp, commit)


def _maybe_zoom_sample(state: AugState, i: int, rng, min_visible_kp: int) -> None:
    if rng.random() >= 0.5:
        return

    scale = float(rng.uniform(0.85, 1.15))
    cx_img = np.float32(state.w) * 0.5
    cy_img = np.float32(state.h) * 0.5

    kp_i = state.kp[i].copy()
    kp_i[:, 0] = (kp_i[:, 0] - cx_img) * scale + cx_img
    kp_i[:, 1] = (kp_i[:, 1] - cy_img) * scale + cy_img

    vis_i = state.vis[i] & _kp_in_bounds(kp_i, state.h, state.w)

    def commit():
        state.image[i] = _zoom_image_np(
            state.image[i],
            scale,
            resample=Image.BILINEAR,
            fill=0,
        )
        state.mask[i] = _zoom_image_np(
            state.mask[i].astype(np.uint8),
            scale,
            resample=Image.NEAREST,
            fill=0,
        ).astype(state.mask.dtype)

        s = np.float32(scale)
        state.K[i, 0, 0] *= s
        state.K[i, 1, 1] *= s
        state.K[i, 0, 2] = (state.K[i, 0, 2] - cx_img) * s + cx_img
        state.K[i, 1, 2] = (state.K[i, 1, 2] - cy_img) * s + cy_img

    _commit_if_enough_visible(state, i, kp_i, vis_i, min_visible_kp, commit)


def _apply_geometric_aug_batch_inplace(
    state: AugState,
    rng,
    min_visible_kp: int,
) -> None:
    for i in range(state.image.shape[0]):
        _maybe_rotate_sample(state, i, rng, min_visible_kp)
        _maybe_translate_sample(state, i, rng, min_visible_kp)
        _maybe_zoom_sample(state, i, rng, min_visible_kp)


def _apply_color_aug_np(image: np.ndarray, rng) -> np.ndarray:
    base_key = jax.random.key(rng.integers(2**32 - 1, dtype=np.uint32))
    keys = jax.random.split(base_key, image.shape[0])
    imgs_f = jnp.asarray(image, dtype=jnp.float32) / 255.0

    return np.clip(
        np.asarray(_apply_augmax_color_batch(keys, imgs_f)) * 255.0,
        0,
        255,
    ).astype(np.uint8)
