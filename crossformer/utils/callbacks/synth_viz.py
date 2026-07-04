"""Callback for visualizing predicted 2D keypoints + translucent robot overlay.

Pipeline per sample:
  1. Extract joints, kp2d, cam_intr, cam_extr from bundled prediction
  2. Denormalize joints (z-score → degrees → radians) and cam_extr translation
  3. Reconstruct w2c from cam_extr (Gram-Schmidt on 6D rotation + translation)
  4. Unscale cam_intr → build K matrix
  5. Rasterize translucent robot mesh with (joints, w2c, K)
  6. Composite onto upscaled image, then overlay GT + pred kp2d points
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import jax.numpy as jnp
import numpy as np

from crossformer.data.grain.metadata import DatasetStatistics
from crossformer.embody import DOF, KP2D_NAMES
from crossformer.utils.callbacks.base import EvalContext
import wandb

# ---------------------------------------------------------------------------
# DOF ID lookups
# ---------------------------------------------------------------------------

KP2D_UV_DOF_NAMES = tuple(f"kp2d_{n}_{ax}" for n in KP2D_NAMES for ax in ("u", "v"))
KP2D_VIS_DOF_NAMES = tuple(f"kp2d_{n}_vis" for n in KP2D_NAMES)
KP2D_UV_IDS = tuple(DOF[name] for name in KP2D_UV_DOF_NAMES)  # (20,)
KP2D_VIS_IDS = tuple(DOF[name] for name in KP2D_VIS_DOF_NAMES)  # (10,)
JOINT_IDS = tuple(DOF[f"j{i}"] for i in range(7))
GRIPPER_ID = DOF["gripper"]
CAM_INTR_IDS = tuple(DOF[n] for n in ("cam_fx", "cam_fy", "cam_cx", "cam_cy"))
CAM_EXTR_T_IDS = tuple(DOF[n] for n in ("t_x", "t_y", "t_z"))  # translation
CAM_EXTR_R6D_IDS = tuple(DOF[f"r6d_{i}"] for i in range(6))  # Zhou 6D rotation
CAM_EXTR_IDS = CAM_EXTR_T_IDS + CAM_EXTR_R6D_IDS  # (9,)

# landmark → URDF link name (order matches KP2D_NAMES)
LANDMARK_LINKS = (
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "link_eef",
    "link_tcp",
)

# cam_intr scaling constants (must match restructure_xarm_dream)
FX_MIN, FX_MAX = 450.0, 900.0
IMG_W, IMG_H = 640, 480

GT_COLOR = (0, 255, 0)  # green (BGR)
PRED_COLOR = (0, 0, 255)  # red
ROBOT_COLOR = np.array([0.2, 0.4, 0.9], dtype=np.float32)
ROBOT_ALPHA = 0.6


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_slots(act_base: np.ndarray, act_id: np.ndarray, dof_ids: tuple[int, ...]) -> np.ndarray | None:
    """Extract specific DOF slots from bundled action. Returns (..., len(dof_ids))."""
    id_arr = np.asarray(act_id)
    if id_arr.ndim == 1:
        id_arr = id_arr[None]
    sample_ids = id_arr[0]
    indices = []
    for dof_id in dof_ids:
        idx = np.where(sample_ids == dof_id)[0]
        if len(idx) == 0:
            return None
        indices.append(idx[0])
    return np.asarray(act_base)[..., indices]


def extract_kp2d_uv(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract kp2d uv coordinates as (B, ..., 10, 2)."""
    flat = _extract_slots(act_base, act_id, KP2D_UV_IDS)
    if flat is None:
        return None
    return flat.reshape(*flat.shape[:-1], 10, 2)


def extract_kp2d_vis(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract kp2d per-joint visibility as (B, ..., 10)."""
    return _extract_slots(act_base, act_id, KP2D_VIS_IDS)


def extract_joints_from_bundled(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract 7-DOF joints from bundled action."""
    return _extract_slots(act_base, act_id, JOINT_IDS)


def extract_cam_intr_from_bundled(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract cam_intr (fx, fy, cx, cy) from bundled action."""
    return _extract_slots(act_base, act_id, CAM_INTR_IDS)


def extract_cam_extr_from_bundled(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract cam_extr (tx,ty,tz, r6d_0..r6d_5) from bundled action as (..., 9)."""
    return _extract_slots(act_base, act_id, CAM_EXTR_IDS)


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------


def _pick_stats(stats, key: str):
    """Fetch ArrayStatistics for `key` from DatasetStatistics or mix dict."""
    if isinstance(stats, Mapping):
        stats = next(iter(stats.values()), None)
    if stats is None:
        return None
    return stats.action.get(key) or stats.proprio.get(key)


def denorm_joints(joints: np.ndarray, stats) -> np.ndarray:
    """Denormalize joints from z-score back to degrees.

    TODO: for multi-dataset mixes, resolve per-sample via batch dataset_name
    instead of picking the first entry.
    """
    s = _pick_stats(stats, "joints")
    return joints if s is None else s.unnormalize(joints)


def denorm_cam_extr(cam_extr: np.ndarray, stats) -> np.ndarray:
    """Denormalize cam_extr translation; 6D rotation was never normalized.

    Input/output: (..., 9) = (tx,ty,tz, r6d_0..r6d_5)
    """
    s = _pick_stats(stats, "cam_extr")
    if s is None:
        return cam_extr
    out = np.asarray(cam_extr, dtype=np.float32).copy()
    mean = np.asarray(s.mean, dtype=np.float32)[:3]
    std = np.asarray(s.std, dtype=np.float32)[:3]
    out[..., :3] = out[..., :3] * np.maximum(std, 1e-8) + mean
    return out


def reconstruct_w2c(cam_extr_physical: np.ndarray) -> np.ndarray:
    """Build a 4x4 world-to-camera matrix from (t, r6d) using Gram-Schmidt.

    cam_extr_physical: (9,) denormalized = (tx,ty,tz, r6d_0..r6d_5) where
    r6d = [R[:,0], R[:,1]]. Recovers the third column via cross product
    (Zhou et al. 2019).
    """
    x = np.asarray(cam_extr_physical, dtype=np.float64)
    t = x[:3]
    r1, r2 = x[3:6], x[6:9]
    b1 = r1 / max(np.linalg.norm(r1), 1e-8)
    b2 = r2 - np.dot(b1, r2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)  # columns = [b1, b2, b3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def unscale_cam_intr(cam: np.ndarray) -> np.ndarray:
    """Reverse the manual [0,1] scaling applied in restructure_xarm_dream.

    Input:  (fx_s, fy_s, cx_s, cy_s) in [0, 1]
    Output: (fx, fy, cx, cy) in pixel units
    """
    fx = cam[..., 0] * (FX_MAX - FX_MIN) + FX_MIN
    fy = cam[..., 1] * (FX_MAX - FX_MIN) + FX_MIN
    cx = cam[..., 2] * IMG_W
    cy = cam[..., 3] * IMG_H
    return np.stack([fx, fy, cx, cy], axis=-1)


def build_K(cam: np.ndarray) -> np.ndarray:
    """Build 3x3 intrinsic matrix from (fx, fy, cx, cy)."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = cam[0]
    K[1, 1] = cam[1]
    K[0, 2] = cam[2]
    K[1, 2] = cam[3]
    return K


# ---------------------------------------------------------------------------
# FK + PnP
# ---------------------------------------------------------------------------


def _get_robot_mesh():
    """Lazy-load shared _RobotMesh instance."""
    from crossformer.utils.callbacks.rast import _RobotMesh

    if not hasattr(_get_robot_mesh, "_cache"):
        _get_robot_mesh._cache = _RobotMesh(Path("xarm7_standalone.urdf"), Path("assets"))
    return _get_robot_mesh._cache


def fk_keypoints(joints_rad: np.ndarray, robot=None) -> np.ndarray:
    """Run FK and extract 3D positions for the 10 landmarks.

    Args:
        joints_rad: (7,) joint angles in radians
    Returns:
        (10, 3) world-frame 3D positions
    """
    from crossformer.utils.callbacks.rast import _poses_to_mats

    if robot is None:
        robot = _get_robot_mesh()
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    poses = robot._fk(jnp.asarray(q))
    mats = np.asarray(_poses_to_mats(poses))[0]  # (num_links, 4, 4)
    pts = []
    for link_name in LANDMARK_LINKS:
        idx = robot.link_index[link_name]
        pts.append(mats[idx, :3, 3])
    return np.stack(pts)  # (10, 3)


def solve_pnp(
    pts_3d: np.ndarray,
    pts_2d_px: np.ndarray,
    K: np.ndarray,
    vis_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Solve PnP → 4x4 world-to-camera matrix, or None on failure.

    Args:
        pts_3d: (N, 3) world frame
        pts_2d_px: (N, 2) pixel coordinates
        K: (3, 3) intrinsic matrix
        vis_mask: (N,) bool. Only visible points are fed to solvePnP. PnP needs
            ≥4 correspondences; returns None if fewer are visible.
    """
    mask = np.ones(len(pts_2d_px), dtype=bool) if vis_mask is None else np.asarray(vis_mask, dtype=bool)
    if mask.sum() < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        pts_3d[mask].astype(np.float64),
        pts_2d_px[mask].astype(np.float64),
        K,
        None,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    return w2c


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_robot(
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    robot=None,
) -> np.ndarray:
    """Render robot silhouette mask (H, W) using nvdiffrast.

    Returns float32 mask in [0, 1].
    """
    from crossformer.utils.callbacks.rast import _GpuRasterizer

    if robot is None:
        robot = _get_robot_mesh()

    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    verts = robot.posed_verts(q)  # (1, V, 4) homogeneous world-frame

    # build clip-space MVP from intrinsics + extrinsics
    # OpenCV intrinsics → OpenGL-style projection
    znear, zfar = 0.01, 10.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0

    # OpenCV → OpenGL camera convention (flip Y and Z)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    view = flip @ w2c

    mvp = (P @ view).astype(np.float32)
    clip = np.einsum("bvi,ji->bvj", verts, mvp)

    rasterizer = _GpuRasterizer(robot.faces)
    mask = rasterizer.render_masks(clip, width, height)  # (1, H, W)
    return mask[0]


def composite_robot(img: np.ndarray, mask: np.ndarray, alpha: float = ROBOT_ALPHA) -> np.ndarray:
    """Blend translucent robot color onto image using mask."""
    if img.shape[-1] == 4:
        img = img[..., :3]  # strip alpha
    img = img.astype(np.float32) / 255.0
    w = mask[:, :, None] * alpha
    color = ROBOT_COLOR[None, None, :]
    blended = img * (1.0 - w) + color * w
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_kp2d(
    img: np.ndarray,
    kp2d: np.ndarray,
    render_h: int,
    render_w: int,
    colors: list[tuple[int, int, int]],
    radius: int = 4,
    thickness: int = 2,
    label: bool = True,
    skeleton: bool = True,
    vis: np.ndarray | None = None,
    vis_thresh: float = 0.5,
) -> np.ndarray:
    def _ok(i: int) -> bool:
        return True if vis is None else bool(vis[i] >= vis_thresh)

    for i, (u, v) in enumerate(kp2d):
        if not _ok(i):
            continue
        px, py = int(u * render_w), int(v * render_h)
        color = colors[i % len(colors)]
        cv2.circle(img, (px, py), radius, color, thickness)
        if label:
            cv2.putText(img, KP2D_NAMES[i], (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    if skeleton:
        for i in range(len(kp2d) - 1):
            if not (_ok(i) and _ok(i + 1)):
                continue
            u0, v0 = kp2d[i]
            u1, v1 = kp2d[i + 1]
            p0 = (int(u0 * render_w), int(v0 * render_h))
            p1 = (int(u1 * render_w), int(v1 * render_h))
            cv2.line(img, p0, p1, colors[0], 1)
    return img


def overlay_kp2d(
    img: np.ndarray,
    kp2d: np.ndarray,
    render_h: int = 480,
    render_w: int = 640,
    pred_kp2d: np.ndarray | None = None,
    radius: int = 4,
    thickness: int = 2,
    vis: np.ndarray | None = None,
    pred_vis: np.ndarray | None = None,
    vis_thresh: float = 0.5,
) -> np.ndarray:
    img = np.asarray(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = cv2.resize(img, (render_w, render_h))
    img = img.copy()
    img = _draw_kp2d(
        img,
        kp2d,
        render_h,
        render_w,
        [GT_COLOR] * 10,
        radius,
        thickness,
        label=True,
        skeleton=True,
        vis=vis,
        vis_thresh=vis_thresh,
    )
    if pred_kp2d is not None:
        img = _draw_kp2d(
            img,
            pred_kp2d,
            render_h,
            render_w,
            [PRED_COLOR] * 10,
            radius + 1,
            1,
            label=False,
            skeleton=True,
            vis=pred_vis,
            vis_thresh=vis_thresh,
        )
    return img


# ---------------------------------------------------------------------------
# Full render pipeline for one sample
# ---------------------------------------------------------------------------


def render_sample(
    img: np.ndarray,
    gt_kp2d: np.ndarray,
    pred_kp2d: np.ndarray,
    pred_joints_deg: np.ndarray,
    pred_cam_scaled: np.ndarray,
    pred_cam_extr: np.ndarray,
    render_h: int = 480,
    render_w: int = 640,
    pred_vis: np.ndarray | None = None,
    gt_vis: np.ndarray | None = None,
    vis_thresh: float = 0.5,
) -> np.ndarray:
    """Full pipeline: rasterize robot from (joints, extr) + overlay kp2d.

    Args:
        img: (H, W, 3) uint8 source image (any size)
        gt_kp2d: (10, 2) normalized [0,1]
        pred_kp2d: (10, 2) normalized [0,1]
        pred_joints_deg: (7,) joint angles in degrees
        pred_cam_scaled: (4,) scaled cam_intr [0,1]
        pred_cam_extr: (9,) physical cam_extr = (tx,ty,tz, r6d_0..r6d_5),
            translation already denormalized; 6D rotation raw.
        pred_vis: (10,) predicted per-joint visibility in [0,1]. Used only to
            gate the pred kp2d overlay.
        gt_vis: (10,) ground-truth per-joint visibility. Gates GT overlay drawing.
    """
    img = np.asarray(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = cv2.resize(img, (render_w, render_h)).copy()

    # recover camera intrinsics + extrinsics directly from prediction
    cam_px = unscale_cam_intr(pred_cam_scaled)
    K = build_K(cam_px)
    joints_rad = np.deg2rad(pred_joints_deg.astype(np.float64))
    w2c = reconstruct_w2c(pred_cam_extr)

    try:
        robot = _get_robot_mesh()
        mask = rasterize_robot(joints_rad, w2c, K, render_w, render_h, robot)
        img = composite_robot(img, mask)
    except Exception:
        pass  # fall through to kp2d-only overlay

    # overlay keypoints on top — GT gated by gt_vis, pred gated by pred_vis
    img = _draw_kp2d(
        img,
        gt_kp2d,
        render_h,
        render_w,
        [GT_COLOR] * 10,
        4,
        2,
        label=True,
        skeleton=True,
        vis=gt_vis,
        vis_thresh=vis_thresh,
    )
    img = _draw_kp2d(
        img,
        pred_kp2d,
        render_h,
        render_w,
        [PRED_COLOR] * 10,
        5,
        1,
        label=False,
        skeleton=True,
        vis=pred_vis,
        vis_thresh=vis_thresh,
    )
    return img


@dataclass
class _SynthVizParts:
    images: np.ndarray
    gt_uv: np.ndarray
    pred_uv: np.ndarray
    gt_vis: np.ndarray | None
    pred_vis: np.ndarray | None
    pred_joints: np.ndarray | None
    pred_cam: np.ndarray | None
    pred_extr: np.ndarray | None


def _pick_uv(arr, idx: int):
    return arr[idx, 0, 0] if arr.ndim == 5 else arr[idx, 0]


def _pick_vis(arr, idx: int):
    if arr is None:
        return None
    return arr[idx, 0, 0] if arr.ndim == 4 else arr[idx, 0]


def _squeeze_slots(arr, idx: int):
    """Pick sample `idx` from a (B, [horizon, [H,]] D) DOF-slot tensor."""
    if arr.ndim >= 4:
        return arr[idx, 0, 0]
    if arr.ndim == 3:
        return arr[idx, 0]
    return arr[idx]


def _synth_viz_parts(batch: dict, pred: np.ndarray, stats: DatasetStatistics | None) -> _SynthVizParts | None:
    act_id = np.asarray(batch["act"]["id"])
    act_base = np.asarray(batch["act"]["base"])

    pred_uv = extract_kp2d_uv(pred, act_id)
    if pred_uv is None:
        return None
    gt_uv = extract_kp2d_uv(act_base, act_id)
    pred_vis = extract_kp2d_vis(pred, act_id)
    gt_vis = extract_kp2d_vis(act_base, act_id)

    pred_joints = extract_joints_from_bundled(pred, act_id)
    pred_cam = extract_cam_intr_from_bundled(pred, act_id)
    pred_extr = extract_cam_extr_from_bundled(pred, act_id)

    if stats is not None:
        if pred_joints is not None:
            pred_joints = denorm_joints(pred_joints, stats)
        if pred_extr is not None:
            pred_extr = denorm_cam_extr(pred_extr, stats)

    return _SynthVizParts(
        images=np.asarray(batch["observation"]["image_primary"]),
        gt_uv=gt_uv,
        pred_uv=pred_uv,
        gt_vis=gt_vis,
        pred_vis=pred_vis,
        pred_joints=pred_joints,
        pred_cam=pred_cam,
        pred_extr=pred_extr,
    )


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@dataclass
class SynthVizCallback:
    """Overlay predicted 2D keypoints + translucent robot on images."""

    name: str = "synth_kp2d"
    every: int = 0
    render_h: int = 480
    render_w: int = 640
    max_samples: int = 8

    def _render_panel(self, parts: _SynthVizParts, idx: int) -> np.ndarray:
        img = parts.images[idx, 0] if parts.images.ndim == 5 else parts.images[idx]
        gt_uv = _pick_uv(parts.gt_uv, idx)
        pred_uv = _pick_uv(parts.pred_uv, idx)
        gt_vis = _pick_vis(parts.gt_vis, idx)
        pred_vis = _pick_vis(parts.pred_vis, idx)

        if parts.pred_joints is not None and parts.pred_cam is not None and parts.pred_extr is not None:
            return render_sample(
                img,
                gt_uv,
                pred_uv,
                _squeeze_slots(parts.pred_joints, idx),
                _squeeze_slots(parts.pred_cam, idx),
                _squeeze_slots(parts.pred_extr, idx),
                self.render_h,
                self.render_w,
                pred_vis=pred_vis,
                gt_vis=gt_vis,
            )
        return overlay_kp2d(
            img,
            gt_uv,
            self.render_h,
            self.render_w,
            pred_kp2d=pred_uv,
            vis=gt_vis,
            pred_vis=pred_vis,
        )

    def __call__(self, ctx: EvalContext) -> dict:
        parts = _synth_viz_parts(dict(ctx.batch), ctx.pred, ctx.stats)
        if parts is None:
            return {}

        B = min(parts.images.shape[0], self.max_samples)
        panels = []
        for i in range(B):
            panel = self._render_panel(parts, i)
            panels.append(wandb.Image(panel, caption=f"sample {i} | GT=green, Pred=red"))

        return {"panels": panels}
