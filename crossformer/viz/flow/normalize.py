from __future__ import annotations

from typing import Any, Mapping

import jax
import numpy as np


def get_intrinsics(batch: Mapping[str, Any], fallback: np.ndarray) -> np.ndarray:
    for path in [
        ("observation", "camera_intrinsics"),
        ("observation", "K"),
        ("task", "camera_intrinsics"),
    ]:
        cur: Any = batch
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                break
            cur = cur[key]
        else:
            K = np.asarray(jax.device_get(cur))
            return K[0] if K.ndim == 3 else K
    return fallback


def get_extrinsics(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    obs = batch.get("observation", {})
    if not isinstance(obs, Mapping):
        return None
    if "camera_extrinsics_R" in obs and "camera_extrinsics_t" in obs:
        R = np.asarray(jax.device_get(obs["camera_extrinsics_R"]))
        t = np.asarray(jax.device_get(obs["camera_extrinsics_t"]))
        return (R[0] if R.ndim == 3 else R), (t[0] if t.ndim == 2 else t)
    return None


def first_vis_array(vis: Mapping[str, np.ndarray], keys: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        arr = vis.get(key)
        if arr is not None:
            return np.asarray(arr), key
    return None, None


def flow_steps_first_sample_any(steps: np.ndarray) -> np.ndarray | None:
    if steps.ndim < 3:
        return None
    if steps.ndim == 3:
        return steps
    bsf = steps if steps.shape[0] <= steps.shape[1] else np.swapaxes(steps, 0, 1)
    return bsf[0]


def normalize_q_flow_steps(q_steps: np.ndarray) -> np.ndarray | None:
    q_sfd = flow_steps_first_sample_any(q_steps)
    if q_sfd is None:
        return None
    if q_sfd.ndim == 2:
        return q_sfd[:, None, :]
    if q_sfd.ndim == 3:
        return q_sfd
    return None


def normalize_xyz_flow_steps(x_steps: np.ndarray) -> np.ndarray | None:
    x_s = flow_steps_first_sample_any(x_steps)
    if x_s is None or x_s.shape[-1] != 3:
        return None
    if x_s.ndim == 3:
        return x_s[:, :, None, :]
    if x_s.ndim == 4:
        return x_s
    return None
