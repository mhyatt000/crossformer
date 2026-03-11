from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax
import numpy as np

from crossformer.utils.train_callbacks import ValidationCallback
import wandb

_DEFAULT_K = np.array(
    [[515.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_INTRINSICS_DIR = "~/intrinsics/cam"


def load_camera_extrinsics(view: str, base_dir: str = _INTRINSICS_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load R, t from HT.npz for a given camera view (low/over/side)."""
    import os

    path = os.path.expanduser(f"{base_dir}/{view}/HT.npz")
    HT = np.load(path)["HT"]  # (4,4)
    return HT[:3, :3].astype(np.float32), HT[:3, 3].astype(np.float32)


def _get_intrinsics(batch: Mapping[str, Any], fallback: np.ndarray) -> np.ndarray:
    for path in [
        ("observation", "camera_intrinsics"),
        ("observation", "K"),
        ("task", "camera_intrinsics"),
    ]:
        cur: Any = batch
        for k in path:
            if not isinstance(cur, Mapping) or k not in cur:
                break
            cur = cur[k]
        else:
            K = np.asarray(jax.device_get(cur))
            return K[0] if K.ndim == 3 else K
    return fallback


def _get_extrinsics(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    obs = batch.get("observation", {})
    if not isinstance(obs, Mapping):
        return None
    if "camera_extrinsics_R" in obs and "camera_extrinsics_t" in obs:
        R = np.asarray(jax.device_get(obs["camera_extrinsics_R"]))
        t = np.asarray(jax.device_get(obs["camera_extrinsics_t"]))
        return (R[0] if R.ndim == 3 else R), (t[0] if t.ndim == 2 else t)
    return None


def world_to_uv(X_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """X_world: (...,3) → uv: (...,2)"""
    X_cam = (X_world @ R.T) + t
    z = np.maximum(X_cam[..., 2], eps)
    u = K[0, 0] * (X_cam[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (X_cam[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def _mpl_scatter_frame_uv(uv: np.ndarray, title: str) -> np.ndarray:
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor("white")
    ax.scatter(uv[:, 0], uv[:, 1], s=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _mpl_scatter3d_frame(
    xyz: np.ndarray,
    title: str,
    lims: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """xyz: (N,3); lims: (lo (3,), hi (3,)) fixed across frames for stable animation."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    lo, hi = lims
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=8)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _make_3d_video(data_ft: np.ndarray, title_prefix: str) -> np.ndarray:
    """data_ft: (ft, N, 3) → (T,C,H,W) video with fixed axis limits."""
    lo = data_ft.reshape(-1, 3).min(axis=0)
    hi = data_ft.reshape(-1, 3).max(axis=0)
    pad = np.maximum((hi - lo) * 0.1, 1e-3)
    lims = (lo - pad, hi + pad)
    frames = [
        _mpl_scatter3d_frame(data_ft[k], title=f"{title_prefix} ft={k}", lims=lims) for k in range(data_ft.shape[0])
    ]
    return _frames_to_video(frames)


def _frames_to_video(frames_rgb: list[np.ndarray]) -> np.ndarray:
    return np.stack(frames_rgb).transpose(0, 3, 1, 2)  # (T,C,H,W)


@dataclass
class FlowVisCallback(ValidationCallback):
    """Projects joints_ft (B, ft, J, 3) to UV and logs wandb.Video per refinement step.

    Set camera_view to auto-load extrinsics from ~/intrinsics/cam/{view}/HT.npz.
    """

    fps: int = 10
    max_videos: int = 8
    camera_view: str = "low"  # one of: low, over, side
    camera_intrinsics: np.ndarray = field(default_factory=lambda: _DEFAULT_K.copy())
    camera_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    camera_t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self):
        super().__post_init__()
        self.camera_R, self.camera_t = load_camera_extrinsics(self.camera_view)

    def __call__(self, train_state, step: int) -> dict[str, Any]:
        if jax.process_index() != 0:
            return {}

        out: dict[str, Any] = {}
        for ds_name, val_data_iter in self.val_iterators.items():
            uv_videos: list[wandb.Video] = []
            joints_videos: list[wandb.Video] = []
            xyz_videos: list[wandb.Video] = []

            for _ in range(self.num_val_batches):
                batch = next(val_data_iter)
                metric_by_mode = self.eval_step(train_state, batch)

                K = _get_intrinsics(batch, self.camera_intrinsics)
                extr = load_camera_extrinsics(batch)
                R, t = extr if extr is not None else (self.camera_R, self.camera_t)

                for mode, metric in metric_by_mode.items():
                    vis = metric.get("vis")
                    if vis is None:
                        continue
                    vis = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), vis)
                    pfx = f"{ds_name}/{mode}"

                    joints_ft = vis.get("joints_ft")
                    if joints_ft is not None:
                        joints_ft0 = joints_ft[0]  # (ft, J, 3)
                        uv_frames = [
                            _mpl_scatter_frame_uv(
                                world_to_uv(joints_ft0[k], R=R, t=t, K=K),
                                title=f"{pfx} uv ft={k}",
                            )
                            for k in range(joints_ft0.shape[0])
                        ]
                        uv_videos.append(wandb.Video(_frames_to_video(uv_frames), fps=self.fps))
                        joints_videos.append(wandb.Video(_make_3d_video(joints_ft0, f"{pfx} joints"), fps=self.fps))

                    xyz_ft = vis.get("xyz_ft")
                    if xyz_ft is not None:
                        xyz_ft0 = xyz_ft[0]  # (ft, N, 3) or (ft, 3)
                        if xyz_ft0.ndim == 2:
                            xyz_ft0 = xyz_ft0[:, None, :]  # (ft, 1, 3)
                        xyz_videos.append(wandb.Video(_make_3d_video(xyz_ft0, f"{pfx} xyz"), fps=self.fps))

                n = max(len(uv_videos), len(joints_videos), len(xyz_videos))
                if n >= self.max_videos:
                    break

            out[f"flow_vis/{ds_name}/uv"] = uv_videos[: self.max_videos]
            out[f"flow_vis/{ds_name}/joints_3d"] = joints_videos[: self.max_videos]
            out[f"flow_vis/{ds_name}/xyz_3d"] = xyz_videos[: self.max_videos]

        return out
