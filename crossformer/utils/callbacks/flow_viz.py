from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Mapping
import warnings

import jax
import numpy as np

from crossformer.utils.train_callbacks import ValidationCallback
import wandb

_DEFAULT_K = np.array(
    [[515.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_INTRINSICS_DIR = "~/intrinsics/cam"

# ROS REP-103 (X=fwd, Y=left, Z=up) → OpenCV (X=right, Y=down, Z=fwd)
_R_ROS2CV = np.array(
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    dtype=np.float32,
)

# ---- new module wiring (mode 2, URDF robot renderer) ----
from xgym import calibrate
from xgym.calibrate.urdf.robot import urdf

from scripts.flow_viz.flow_viz_overlay import render_xyz_overlay_video
from scripts.flow_viz.flow_viz_pca import _ensure_snd, _plot_two_panel_video
from scripts.flow_viz.flow_viz_robot import make_robot_from_urdf, render_robot_q_flow_video


def load_camera_extrinsics(view: str, base_dir: str = _INTRINSICS_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load world-to-camera (R_w2c, t_w2c) for a given camera view (low/over/side)."""
    path = f"/home/nhogg/crossformer_data/extr/cam/{view}/HT.npz"
    HT = np.load(path)["HT"]  # (4,4) world-to-camera
    R_w2c = HT[:3, :3].astype(np.float32)
    t_w2c = HT[:3, 3].astype(np.float32)
    return R_w2c, t_w2c


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


def world_to_uv(
    X_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    ros_to_opencv: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """X_world: (...,3) → uv: (...,2)"""
    if ros_to_opencv:
        X_world = X_world @ _R_ROS2CV.T
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


def _mpl_scatter3d_denoise_frame(
    pred_xyz: np.ndarray,
    title: str,
    lims: tuple[np.ndarray, np.ndarray],
    target_xyz: np.ndarray | None = None,
) -> np.ndarray:
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    lo, hi = lims
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], s=8, alpha=0.9, label="pred")
    if target_xyz is not None:
        ax.scatter(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], s=8, alpha=0.3, label="target")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    if target_xyz is not None:
        ax.legend(loc="best")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _make_3d_video(data_ft: np.ndarray, title_prefix: str) -> np.ndarray:
    lo = data_ft.reshape(-1, 3).min(axis=0)
    hi = data_ft.reshape(-1, 3).max(axis=0)
    pad = np.maximum((hi - lo) * 0.1, 1e-3)
    lims = (lo - pad, hi + pad)
    frames = [
        _mpl_scatter3d_frame(data_ft[k], title=f"{title_prefix} ft={k}", lims=lims) for k in range(data_ft.shape[0])
    ]
    return _frames_to_video(frames)


def _make_denoise_3d_video(pred_sftj3: np.ndarray, target_ftj3: np.ndarray | None, title_prefix: str) -> np.ndarray:
    pred_snp3 = pred_sftj3.reshape(pred_sftj3.shape[0], -1, 3)
    target_np3 = None if target_ftj3 is None else target_ftj3.reshape(-1, 3)

    all_pts = pred_snp3.reshape(-1, 3)
    if target_np3 is not None:
        all_pts = np.concatenate([all_pts, target_np3], axis=0)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = np.maximum((hi - lo) * 0.1, 1e-3)
    lims = (lo - pad, hi + pad)

    frames = [
        _mpl_scatter3d_denoise_frame(
            pred_xyz=pred_snp3[s],
            target_xyz=target_np3,
            title=f"{title_prefix} flow_step={s}",
            lims=lims,
        )
        for s in range(pred_snp3.shape[0])
    ]
    return _frames_to_video(frames)


def _frames_to_video(frames_rgb: list[np.ndarray]) -> np.ndarray:
    return np.stack(frames_rgb).transpose(0, 3, 1, 2)  # (T,C,H,W)


def _first_vis_array(vis: Mapping[str, np.ndarray], keys: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        arr = vis.get(key)
        if arr is not None:
            return np.asarray(arr), key
    return None, None


def _flow_steps_first_sample(pred_steps: np.ndarray) -> np.ndarray | None:
    """Normalize flow-denoise trajectories to (S, ft, J, 3) for sample 0."""
    if pred_steps.ndim != 5 or pred_steps.shape[-1] != 3:
        return None
    bsf = pred_steps if pred_steps.shape[0] <= pred_steps.shape[1] else np.swapaxes(pred_steps, 0, 1)
    return bsf[0]


def _compute_denoise_curves(pred_sftj3: np.ndarray, target_ftj3: np.ndarray | None) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}

    final_ftj3 = pred_sftj3[-1]
    curves["l2_to_final"] = np.linalg.norm(pred_sftj3 - final_ftj3[None], axis=-1).mean(axis=(1, 2))

    if target_ftj3 is None or target_ftj3.ndim != 3 or target_ftj3.shape[-1] != 3:
        return curves

    ft = min(pred_sftj3.shape[1], target_ftj3.shape[0])
    j = min(pred_sftj3.shape[2], target_ftj3.shape[1])
    diff = pred_sftj3[:, :ft, :j] - target_ftj3[None, :ft, :j]
    curves["l2_to_target"] = np.linalg.norm(diff, axis=-1).mean(axis=(1, 2))
    return curves


def _mpl_denoise_curve_plot(curves: Mapping[str, np.ndarray], title: str) -> np.ndarray:
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(5, 3))
    for name, y in curves.items():
        ax.plot(np.arange(len(y)), y, label=name)
    ax.set_title(title)
    ax.set_xlabel("flow_step")
    ax.set_ylabel("mean L2")
    ax.grid(alpha=0.3)
    if curves:
        ax.legend(loc="best")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _show_image_opencv(window_name: str, image_rgb: np.ndarray, wait_ms: int) -> None:
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.imshow(window_name, image_bgr)
        cv2.waitKey(wait_ms)
    except cv2.error as exc:
        msg = str(exc)
        if "The function is not implemented" in msg or "cvShowImage" in msg:
            raise RuntimeError(
                "OpenCV GUI display is unavailable (likely opencv-python-headless). "
                "Disable popout or install GUI-enabled OpenCV."
            ) from exc
        raise


def _flow_steps_first_sample_any(steps: np.ndarray) -> np.ndarray | None:
    if steps.ndim < 3:
        return None
    if steps.ndim == 3:
        return steps
    bsf = steps if steps.shape[0] <= steps.shape[1] else np.swapaxes(steps, 0, 1)
    return bsf[0]


def _normalize_q_flow_steps(q_steps: np.ndarray) -> np.ndarray | None:
    q_sfd = _flow_steps_first_sample_any(q_steps)
    if q_sfd is None:
        return None
    if q_sfd.ndim == 2:
        return q_sfd[:, None, :]
    if q_sfd.ndim == 3:
        return q_sfd
    return None


def _normalize_xyz_flow_steps(x_steps: np.ndarray) -> np.ndarray | None:
    x_s = _flow_steps_first_sample_any(x_steps)
    if x_s is None or x_s.shape[-1] != 3:
        return None
    if x_s.ndim == 3:  # (S, ft, 3)
        return x_s[:, :, None, :]
    if x_s.ndim == 4:  # (S, ft, J, 3)
        return x_s
    return None


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H,W,3); got {x.shape}")
    if x.dtype == np.uint8:
        return x
    x = np.nan_to_num(x)
    if x.max() <= 1.0:
        x = x * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def _step_rgb(step: int, n_steps: int) -> tuple[int, int, int]:
    import matplotlib.pyplot as plt

    denom = max(n_steps - 1, 1)
    r, g, b, _ = plt.cm.coolwarm(step / denom)
    return int(255 * r), int(255 * g), int(255 * b)


def _draw_uv_step_and_flow(
    image: np.ndarray,
    uv_t: np.ndarray,
    uv_next: np.ndarray | None,
    step: int,
    n_steps: int,
) -> np.ndarray:
    import cv2

    out = image.copy()
    color_rgb = _step_rgb(step, n_steps)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    valid_t = np.isfinite(uv_t).all(axis=1)
    uv_ti = np.round(uv_t[valid_t]).astype(np.int32)

    if len(uv_ti) >= 2:
        cv2.polylines(out, [uv_ti.reshape(-1, 1, 2)], isClosed=False, color=(180, 180, 180), thickness=1)

    for p in uv_ti:
        cv2.circle(out, tuple(p), 3, color_bgr, -1)

    if uv_next is not None:
        valid_n = np.isfinite(uv_next).all(axis=1)
        valid = valid_t & valid_n
        uv_ti2 = np.round(uv_t[valid]).astype(np.int32)
        uv_ni = np.round(uv_next[valid]).astype(np.int32)
        for p0, p1 in zip(uv_ti2, uv_ni, strict=False):
            cv2.arrowedLine(out, tuple(p0), tuple(p1), color_bgr, 1, tipLength=0.25)
    return out


def _make_xyz_image_overlay_video(
    xyz_sftj3: np.ndarray,
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    ros_to_opencv: bool,
    ft_idx: int,
) -> np.ndarray:
    ft_idx = min(max(ft_idx, 0), xyz_sftj3.shape[1] - 1)
    base = _as_uint8_rgb(image)
    frames: list[np.ndarray] = []
    for s in range(xyz_sftj3.shape[0]):
        uv_t = world_to_uv(xyz_sftj3[s, ft_idx], R=R, t=t, K=K, ros_to_opencv=ros_to_opencv)
        uv_next = None
        if s + 1 < xyz_sftj3.shape[0]:
            uv_next = world_to_uv(xyz_sftj3[s + 1, ft_idx], R=R, t=t, K=K, ros_to_opencv=ros_to_opencv)
        frames.append(_draw_uv_step_and_flow(base, uv_t, uv_next, step=s, n_steps=xyz_sftj3.shape[0]))
    return _frames_to_video(frames)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (N,D), got {x.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = x / np.maximum(std, 1e-6)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    basis = vh[:2].T
    z = x @ basis
    if z.shape[1] < 2:
        z = np.pad(z, [(0, 0), (0, 2 - z.shape[1])])
    return z


def _direction_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    da = np.diff(a, axis=0)
    db = np.diff(b, axis=0)
    n = min(da.shape[0], db.shape[0])
    da = da[:n]
    db = db[:n]
    na = np.linalg.norm(da, axis=1)
    nb = np.linalg.norm(db, axis=1)
    denom = np.maximum(na * nb, 1e-8)
    cos = np.sum(da * db, axis=1) / denom
    return float(np.mean(cos))


def _mpl_joint_xyz_pca_flow_plot(zq: np.ndarray, zx: np.ndarray, title: str, cos_mean: float) -> np.ndarray:
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))
    ax0.plot(zq[:, 0], zq[:, 1], color="blue", alpha=0.45)
    ax0.scatter(zq[:, 0], zq[:, 1], c=np.arange(len(zq)), cmap="Blues", s=30)
    for i in range(len(zq) - 1):
        ax0.annotate("", xy=zq[i + 1], xytext=zq[i], arrowprops={"arrowstyle": "->", "color": "blue", "alpha": 0.5})
    ax0.set_title("Joint-space PCA flow")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.grid(alpha=0.2)

    ax1.plot(zx[:, 0], zx[:, 1], color="red", alpha=0.45)
    ax1.scatter(zx[:, 0], zx[:, 1], c=np.arange(len(zx)), cmap="Reds", s=30)
    for i in range(len(zx) - 1):
        ax1.annotate("", xy=zx[i + 1], xytext=zx[i], arrowprops={"arrowstyle": "->", "color": "red", "alpha": 0.5})
    ax1.set_title("FK-XYZ PCA flow")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(alpha=0.2)

    fig.suptitle(f"{title} | mean direction cosine={cos_mean:.3f}")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


@dataclass
class FlowVisCallback(ValidationCallback):
    """Original callback + A/B/C module wiring with W&B logging."""

    fps: int = 10
    max_videos: int = 8
    camera_view: str = "low"
    camera_intrinsics: np.ndarray = field(default_factory=lambda: _DEFAULT_K.copy())
    camera_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    camera_t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    ros_to_opencv: bool = False

    # existing features
    enable_denoise_plots: bool = False
    denoise_pred_keys: tuple[str, ...] = ("joints_flow_steps", "flow_joints_steps", "denoise_joints_steps")
    denoise_target_keys: tuple[str, ...] = ("joints_ft", "joints_target_ft")
    show_denoise_plot_window: bool = False
    denoise_plot_window_wait_ms: int = 1
    enable_xyz_image_flow: bool = False
    flow_overlay_ft_index: int = 0
    enable_joint_xyz_pca_flow: bool = False
    flow_q_keys: tuple[str, ...] = ("q_flow_steps", "joint_flow_steps")
    flow_xyz_keys: tuple[str, ...] = ("fk_xyz_flow_steps", "xyz_flow_steps", "joints_flow_steps")
    compute_fk_from_q_steps: bool = False
    q_pca_dims: int = 7

    # new explicit A/B/C toggles
    enable_part_a: bool = True
    enable_part_b: bool = True
    enable_part_c: bool = True
    robot_pad_gripper: bool = True
    run_every_steps: int = 1
    enabled_renderers: tuple[str, ...] = ("xyz_overlay", "joint_fk_pca", "robot_flow")
    max_renderers_per_item: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.camera_R, self.camera_t = load_camera_extrinsics(self.camera_view)
        self._fk_fn = None
        self._robot = None

    def _get_fk_fn(self):
        if self._fk_fn is not None:
            return self._fk_fn
        from crossformer.model.components.adj.cart import get_fwd_kin_fn, make_robot

        robot = make_robot()
        self._fk_fn = get_fwd_kin_fn(robot, pad_gripper=True)
        return self._fk_fn

    def _get_robot(self):
        if self._robot is None:
            self._robot = make_robot_from_urdf(
                urdf_path=urdf,
                mesh_dir=calibrate.urdf.robot.DNAME / "assets",
            )
        return self._robot

    def _overlay_image(self, imgs: np.ndarray, K: np.ndarray) -> np.ndarray:
        if imgs.ndim == 5:
            return imgs[0, min(self.flow_overlay_ft_index, imgs.shape[1] - 1)]
        if imgs.ndim == 4:
            return imgs[0]
        return np.full((int(K[1, 2] * 2), int(K[0, 2] * 2), 3), 255, dtype=np.uint8)

    def _compute_flow_context(
        self,
        vis: dict[str, np.ndarray],
        imgs: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None, np.ndarray | None]:
        q_steps, _ = _first_vis_array(vis, self.flow_q_keys)
        q_sfd = _normalize_q_flow_steps(q_steps) if q_steps is not None else None

        xyz_steps, xyz_key = _first_vis_array(vis, self.flow_xyz_keys)
        xyz_sftj3 = _normalize_xyz_flow_steps(xyz_steps) if xyz_steps is not None else None

        if xyz_sftj3 is None and q_sfd is not None and self.compute_fk_from_q_steps:
            fk = self._get_fk_fn()
            q_for_fk = q_sfd[..., : self.q_pca_dims]
            xyz_sft3 = np.asarray(fk(q_for_fk))
            xyz_sftj3 = xyz_sft3[:, :, None, :]
            xyz_key = "fk(q_flow_steps)"

        return q_sfd, xyz_sftj3, xyz_key, None if xyz_sftj3 is None else self._overlay_image(imgs, K)

    def _render_xyz_overlay(
        self, xyz_sftj3: np.ndarray, img_for_overlay: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> wandb.Video:
        ft_idx = min(self.flow_overlay_ft_index, xyz_sftj3.shape[1] - 1)
        xyz_sj3 = xyz_sftj3[:, ft_idx]
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_mp4,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png,
        ):
            out_mp4 = Path(f_mp4.name)
            out_png = Path(f_png.name)
        render_xyz_overlay_video(
            image=img_for_overlay,
            xyz_steps=xyz_sj3,
            out_mp4=out_mp4,
            out_png=out_png,
            K=K,
            R=R,
            t=t,
            fps=self.fps,
        )
        return wandb.Video(str(out_mp4), fps=self.fps, format="mp4")

    def _render_joint_fk_pca(
        self, q_sfd: np.ndarray, xyz_sftj3: np.ndarray, xyz_key: str | None, pfx: str
    ) -> wandb.Image:
        ft_idx = min(self.flow_overlay_ft_index, q_sfd.shape[1] - 1, xyz_sftj3.shape[1] - 1)
        q_sd = q_sfd[:, ft_idx, : self.q_pca_dims]
        x_sd = xyz_sftj3[:, ft_idx].reshape(xyz_sftj3.shape[0], -1)
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png,
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_mp4,
        ):
            out_png = Path(f_png.name)
            out_mp4 = Path(f_mp4.name)
        _plot_two_panel_video(
            q_flow=_ensure_snd(q_sd, "q_flow"),
            x_flow=_ensure_snd(x_sd, "x_flow"),
            out_png=out_png,
            out_mp4=out_mp4,
            fps=self.fps,
        )
        return wandb.Image(str(out_png), caption=f"{pfx} joint_fk_pca (xyz={xyz_key or 'n/a'})")

    def _render_robot_flow(self, q_sfd: np.ndarray) -> wandb.Video:
        ft_idx = min(self.flow_overlay_ft_index, q_sfd.shape[1] - 1)
        q_sd = q_sfd[:, ft_idx, :]
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_mp4,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png,
        ):
            out_mp4 = Path(f_mp4.name)
            out_png = Path(f_png.name)
        robot = self._get_robot()
        render_robot_q_flow_video(
            robot=robot,
            q_steps=q_sd,
            out_mp4=out_mp4,
            out_png=out_png,
            fps=self.fps,
            pad_gripper=self.robot_pad_gripper,
        )
        return wandb.Video(str(out_mp4), fps=self.fps, format="mp4")

    def __call__(self, train_state, step: int) -> dict[str, Any]:
        if jax.process_index() != 0:
            return {}
        if self.run_every_steps > 1 and (step % self.run_every_steps != 0):
            return {}

        out: dict[str, Any] = {}

        for ds_name, val_data_iter in self.val_iterators.items():
            uv_videos: list[wandb.Video] = []
            joints_videos: list[wandb.Video] = []
            xyz_videos: list[wandb.Video] = []
            denoise_plots: list[wandb.Image] = []
            denoise_points_videos: list[wandb.Video] = []
            xyz_image_flow_videos: list[wandb.Video] = []
            joint_xyz_pca_plots: list[wandb.Image] = []
            denoise_last_l2_to_target: list[float] = []
            joint_xyz_direction_cos: list[float] = []

            xyz_overlay_videos: list[wandb.Video] = []
            joint_fk_pca_images: list[wandb.Image] = []
            robot_flow_videos: list[wandb.Video] = []

            for _ in range(self.num_val_batches):
                batch = next(val_data_iter)
                metric_by_mode = self.eval_step(train_state, batch)

                K = _get_intrinsics(batch, self.camera_intrinsics)
                extr = _get_extrinsics(batch)
                R, t = extr if extr is not None else (self.camera_R, self.camera_t)

                imgs = np.asarray(jax.device_get(batch.get("observation", {}).get("image_primary", np.array([]))))

                for mode, metric in metric_by_mode.items():
                    vis = metric.get("vis")
                    if vis is None:
                        continue
                    vis = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), vis)
                    pfx = f"{ds_name}/{mode}"
                    pred_sftj3: np.ndarray | None = None
                    target_ftj3: np.ndarray | None = None

                    # ---------- original logging ----------
                    joints_ft = vis.get("joints_ft")
                    if joints_ft is not None:
                        joints_ft0 = joints_ft[0]  # (ft, J, 3)
                        uv_frames = [
                            _mpl_scatter_frame_uv(
                                world_to_uv(joints_ft0[k], R=R, t=t, K=K, ros_to_opencv=self.ros_to_opencv),
                                title=f"{pfx} uv ft={k}",
                            )
                            for k in range(joints_ft0.shape[0])
                        ]
                        uv_videos.append(wandb.Video(_frames_to_video(uv_frames), fps=self.fps))
                        joints_videos.append(wandb.Video(_make_3d_video(joints_ft0, f"{pfx} joints"), fps=self.fps))

                    xyz_ft = vis.get("xyz_ft")
                    if xyz_ft is not None:
                        xyz_ft0 = xyz_ft[0]
                        if xyz_ft0.ndim == 2:
                            xyz_ft0 = xyz_ft0[:, None, :]
                        xyz_videos.append(wandb.Video(_make_3d_video(xyz_ft0, f"{pfx} xyz"), fps=self.fps))

                    if self.enable_denoise_plots:
                        pred_steps, pred_key = _first_vis_array(vis, self.denoise_pred_keys)
                        target_ft, _ = _first_vis_array(vis, self.denoise_target_keys)
                        if pred_steps is not None and pred_key is not None:
                            pred_sftj3 = _flow_steps_first_sample(pred_steps)
                            if target_ft is not None and target_ft.ndim == 4 and target_ft.shape[-1] == 3:
                                target_ftj3 = target_ft[0]
                            if pred_sftj3 is not None:
                                curves = _compute_denoise_curves(pred_sftj3, target_ftj3)
                                plot_rgb = _mpl_denoise_curve_plot(curves, f"{pfx} denoise ({pred_key})")
                                denoise_plots.append(wandb.Image(plot_rgb, caption=f"{pfx} denoise"))
                                denoise_points_videos.append(
                                    wandb.Video(
                                        _make_denoise_3d_video(
                                            pred_sftj3=pred_sftj3,
                                            target_ftj3=target_ftj3,
                                            title_prefix=f"{pfx} denoise_points",
                                        ),
                                        fps=self.fps,
                                    )
                                )
                                if self.show_denoise_plot_window:
                                    try:
                                        _show_image_opencv(
                                            window_name=f"denoise::{pfx}",
                                            image_rgb=plot_rgb,
                                            wait_ms=self.denoise_plot_window_wait_ms,
                                        )
                                    except RuntimeError as exc:
                                        warnings.warn(f"{exc} Disabling denoise popout.", stacklevel=2)
                                        self.show_denoise_plot_window = False
                                if "l2_to_target" in curves:
                                    denoise_last_l2_to_target.append(float(curves["l2_to_target"][-1]))

                    if self.enable_xyz_image_flow:
                        xyz_sftj3 = pred_sftj3
                        if xyz_sftj3 is None:
                            xyz_steps, _ = _first_vis_array(vis, self.flow_xyz_keys)
                            if xyz_steps is not None:
                                xyz_sftj3 = _normalize_xyz_flow_steps(xyz_steps)
                        if xyz_sftj3 is not None:
                            if imgs.ndim == 5:
                                img_for_overlay = imgs[0, min(self.flow_overlay_ft_index, imgs.shape[1] - 1)]
                            elif imgs.ndim == 4:
                                img_for_overlay = imgs[0]
                            else:
                                img_for_overlay = np.full((int(K[1, 2] * 2), int(K[0, 2] * 2), 3), 255, dtype=np.uint8)
                            xyz_image_flow_videos.append(
                                wandb.Video(
                                    _make_xyz_image_overlay_video(
                                        xyz_sftj3=xyz_sftj3,
                                        image=img_for_overlay,
                                        R=R,
                                        t=t,
                                        K=K,
                                        ros_to_opencv=self.ros_to_opencv,
                                        ft_idx=self.flow_overlay_ft_index,
                                    ),
                                    fps=self.fps,
                                )
                            )

                    if self.enable_joint_xyz_pca_flow:
                        q_steps, q_key = _first_vis_array(vis, self.flow_q_keys)
                        q_sfd = _normalize_q_flow_steps(q_steps) if q_steps is not None else None

                        xyz_steps, xyz_key = _first_vis_array(vis, self.flow_xyz_keys)
                        xyz_sftj3 = _normalize_xyz_flow_steps(xyz_steps) if xyz_steps is not None else None
                        if xyz_sftj3 is None and pred_sftj3 is not None:
                            xyz_sftj3 = pred_sftj3
                            xyz_key = "denoise_pred"

                        if xyz_sftj3 is None and q_sfd is not None and self.compute_fk_from_q_steps:
                            fk = self._get_fk_fn()
                            q_for_fk = q_sfd[..., : self.q_pca_dims]
                            xyz_sft3 = np.asarray(fk(q_for_fk))
                            xyz_sftj3 = xyz_sft3[:, :, None, :]
                            xyz_key = "fk(q_flow_steps)"

                        if q_sfd is not None and xyz_sftj3 is not None:
                            ft_idx = min(self.flow_overlay_ft_index, q_sfd.shape[1] - 1, xyz_sftj3.shape[1] - 1)
                            q_sd = q_sfd[:, ft_idx, : self.q_pca_dims]
                            x_sd = xyz_sftj3[:, ft_idx].reshape(xyz_sftj3.shape[0], -1)
                            zq = _pca_2d(q_sd)
                            zx = _pca_2d(x_sd)
                            cos_mean = _direction_cosine_mean(zq, zx)
                            joint_xyz_pca_plots.append(
                                wandb.Image(
                                    _mpl_joint_xyz_pca_flow_plot(
                                        zq,
                                        zx,
                                        title=f"{pfx} q={q_key or 'n/a'} x={xyz_key or 'n/a'}",
                                        cos_mean=cos_mean,
                                    ),
                                    caption=f"{pfx} joints-vs-fk pca flow",
                                )
                            )
                            if np.isfinite(cos_mean):
                                joint_xyz_direction_cos.append(cos_mean)

                    # ---------- lightweight extensible renderers ----------
                    q_sfd, xyz_sftj3, xyz_key, img_for_overlay = self._compute_flow_context(vis, imgs, K)
                    rendered = 0
                    for renderer in self.enabled_renderers:
                        if rendered >= self.max_renderers_per_item:
                            break
                        try:
                            if (
                                renderer == "xyz_overlay"
                                and self.enable_part_a
                                and xyz_sftj3 is not None
                                and img_for_overlay is not None
                            ):
                                xyz_overlay_videos.append(self._render_xyz_overlay(xyz_sftj3, img_for_overlay, K, R, t))
                                rendered += 1
                            elif (
                                renderer == "joint_fk_pca"
                                and self.enable_part_b
                                and q_sfd is not None
                                and xyz_sftj3 is not None
                            ):
                                joint_fk_pca_images.append(self._render_joint_fk_pca(q_sfd, xyz_sftj3, xyz_key, pfx))
                                rendered += 1
                            elif renderer == "robot_flow" and self.enable_part_c and q_sfd is not None:
                                robot_flow_videos.append(self._render_robot_flow(q_sfd))
                                rendered += 1
                        except Exception as exc:
                            warnings.warn(f"[{pfx}] {renderer} failed: {exc}", stacklevel=2)

                n = max(
                    len(uv_videos),
                    len(joints_videos),
                    len(xyz_videos),
                    len(denoise_plots),
                    len(denoise_points_videos),
                    len(xyz_image_flow_videos),
                    len(joint_xyz_pca_plots),
                    len(xyz_overlay_videos),
                    len(joint_fk_pca_images),
                    len(robot_flow_videos),
                )
                if n >= self.max_videos:
                    break

            # existing keys
            out[f"flow_vis/{ds_name}/uv"] = uv_videos[: self.max_videos]
            out[f"flow_vis/{ds_name}/joints_3d"] = joints_videos[: self.max_videos]
            out[f"flow_vis/{ds_name}/xyz_3d"] = xyz_videos[: self.max_videos]
            if self.enable_denoise_plots:
                out[f"flow_vis/{ds_name}/denoise_plots"] = denoise_plots[: self.max_videos]
                out[f"flow_vis/{ds_name}/denoise_points_3d"] = denoise_points_videos[: self.max_videos]
                if denoise_last_l2_to_target:
                    out[f"flow_vis/{ds_name}/denoise_last_l2_to_target"] = float(np.mean(denoise_last_l2_to_target))
            if self.enable_xyz_image_flow:
                out[f"flow_vis/{ds_name}/xyz_image_flow"] = xyz_image_flow_videos[: self.max_videos]
            if self.enable_joint_xyz_pca_flow:
                out[f"flow_vis/{ds_name}/joint_xyz_pca_flow"] = joint_xyz_pca_plots[: self.max_videos]
                if joint_xyz_direction_cos:
                    out[f"flow_vis/{ds_name}/joint_xyz_direction_cosine"] = float(np.mean(joint_xyz_direction_cos))

            if self.enable_part_a:
                out[f"flow_vis/{ds_name}/xyz_overlay"] = xyz_overlay_videos[: self.max_videos]
            if self.enable_part_b:
                out[f"flow_vis/{ds_name}/joint_fk_pca"] = joint_fk_pca_images[: self.max_videos]
            if self.enable_part_c:
                out[f"flow_vis/{ds_name}/robot_flow"] = robot_flow_videos[: self.max_videos]

        return out
