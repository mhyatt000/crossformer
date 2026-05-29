from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def normalize_kp2d(kp2d: jax.Array, h: int, w: int) -> jax.Array:
    return kp2d / jnp.array([w, h], dtype=jnp.float32)


def denormalize_kp2d(kp2d_norm: jax.Array, h: int, w: int) -> jax.Array:
    return kp2d_norm * jnp.array([w, h], dtype=jnp.float32)


def normalize_kp2d_np(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    return kp2d / np.array([w, h], dtype=np.float32)


def denormalize_kp2d_np(kp2d_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    return kp2d_norm * np.array([w, h], dtype=np.float32)


def shrink_crop_resolution(raw_h: int, raw_w: int, net_h: int, net_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    scale_w = raw_w / net_w
    ref_h_from_w = int(scale_w * net_h)
    scale_h = raw_h / net_h
    ref_w_from_h = int(scale_h * net_w)

    if raw_w >= ref_w_from_h:
        crop_w, crop_h = ref_w_from_h, raw_h
    else:
        crop_w, crop_h = raw_w, ref_h_from_w

    left = (raw_w - crop_w) // 2
    top = (raw_h - crop_h) // 2
    return (crop_h, crop_w), (top, left)


def shrink_crop_keypoints_np(kp2d: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    (crop_h, crop_w), (top, left) = shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    out = np.asarray(kp2d, dtype=np.float32).copy()
    out[:, 0] = (out[:, 0] - left) / crop_w * net_w
    out[:, 1] = (out[:, 1] - top) / crop_h * net_h
    return out


def shrink_crop_intrinsics_np(K: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    (crop_h, crop_w), (top, left) = shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 2] -= left
    out[1, 2] -= top
    out[0] *= net_w / crop_w
    out[1] *= net_h / crop_h
    return out


def default_intrinsics_np(h: int, w: int) -> np.ndarray:
    f = np.float32(600.0)
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]], dtype=np.float32)


def opencv_w2c_np(w2c_raw: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c_raw, dtype=np.float32)
    flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = flip @ w2c[:3, :3]
    out[:3, 3] = flip @ w2c[:3, 3]
    return out


def shrink_crop_image_np(image: np.ndarray, net_h: int, net_w: int, resample: int) -> np.ndarray:
    raw_h, raw_w = image.shape[:2]
    (crop_h, crop_w), (top, left) = shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    pil = Image.fromarray(image)
    pil = pil.crop((left, top, left + crop_w, top + crop_h))
    return np.asarray(pil.resize((net_w, net_h), resample))


_normalize_kp2d = normalize_kp2d
_denormalize_kp2d = denormalize_kp2d
_normalize_kp2d_np = normalize_kp2d_np
_denormalize_kp2d_np = denormalize_kp2d_np
_shrink_crop_resolution = shrink_crop_resolution
_shrink_crop_keypoints_np = shrink_crop_keypoints_np
_shrink_crop_intrinsics_np = shrink_crop_intrinsics_np
_default_intrinsics_np = default_intrinsics_np
_opencv_w2c_np = opencv_w2c_np
_shrink_crop_image_np = shrink_crop_image_np
