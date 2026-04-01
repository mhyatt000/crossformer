"""Pure functions for flow-matching PCA visualization.

Data prep, vectorised FK, PCA fitting, and Matplotlib rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pcax
from pcax.pca import PCAState
import yourdfpy

# (x_min, x_max), (y_min, y_max)
AxisLim = tuple[tuple[float, float], tuple[float, float]]


# ---------------------------------------------------------------------------
# FK factory
# ---------------------------------------------------------------------------


def make_fk_fn(link: str = "link_eef") -> Callable[[np.ndarray], np.ndarray]:
    urdf_path = Path.cwd() / "xarm7_standalone.urdf"
    mesh_dir = Path.cwd() / "assets"
    if not urdf_path.exists():
        raise FileNotFoundError(f"Expected URDF at {urdf_path}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Expected mesh dir at {mesh_dir}")
    robot = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))

    def fk_fn(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2:
            raise ValueError(f"Expected q shape [N,A], got {q.shape}")
        if q.shape[-1] < 7:
            raise ValueError(f"Expected joint dim >= 7, got {q.shape[-1]}")
        xyz = np.zeros((q.shape[0], 3), dtype=np.float32)
        for i, qi in enumerate(q):
            qi_padded = np.pad(qi[:7], (0, 1))
            robot.update_cfg(qi_padded)
            xyz[i] = robot.get_transform(link)[:3, 3]
        return xyz

    return fk_fn


# ---------------------------------------------------------------------------
# Phase 1 — data extraction, debug stats, subsampling
# ---------------------------------------------------------------------------


def prep_data(
    base_joints: np.ndarray,
    flow: np.ndarray,
    max_pts: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten spatial dims, print convergence stats, subsample.

    Args:
        base_joints: Ground-truth joints [B, W, H, A].
        flow:        Predicted trajectory  [F, B, W, H, A].
        max_pts:     Maximum subsampled points S (default 1000).

    Returns:
        base_sub [S, A], flow_sub [F, S, A].
    """
    f = flow.shape[0]
    a = flow.shape[-1]

    base_flat = base_joints.reshape(-1, a)

    flow_flat = flow.reshape(f, -1, a)

    n = base_flat.shape[0]

    mse = float(np.mean((flow_flat[-1] - base_flat) ** 2))
    print(f"[Debug] Target Manifold   | Mean: {base_flat.mean():.4f}, Std: {base_flat.std():.4f}")
    print(f"[Debug] Flow t=0 (Noise)  | Mean: {flow_flat[0].mean():.4f}, Std: {flow_flat[0].std():.4f}")
    print(f"[Debug] Flow t=F (Pred)   | Mean: {flow_flat[-1].mean():.4f}, Std: {flow_flat[-1].std():.4f}")
    print(f"[Debug] MSE(Flow_Final, Target): {mse:.6f}")

    s = min(max_pts, n)
    idx = np.random.choice(n, size=s, replace=False)
    return base_flat[idx], flow_flat[:, idx]


# ---------------------------------------------------------------------------
# Phase 2 — vectorised FK
# ---------------------------------------------------------------------------


def compute_fk(
    base_sub: np.ndarray,
    flow_sub: np.ndarray,
    fk_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FK to subsampled base and every flow step in one batch.

    Args:
        base_sub: [S, A]
        flow_sub: [F, S, A]
        fk_fn:    Callable [N, A] -> [N, 3].

    Returns:
        base_fk_xyz [S, 3], flow_fk_xyz [F, S, 3].
    """
    f, s, a = flow_sub.shape
    base_fk_xyz: np.ndarray = fk_fn(base_sub)
    flow_fk_xyz: np.ndarray = fk_fn(flow_sub.reshape(f * s, a)).reshape(f, s, 3)
    return base_fk_xyz, flow_fk_xyz


# ---------------------------------------------------------------------------
# Phase 3a — PCA fitting (static coordinate frame)
# ---------------------------------------------------------------------------


def fit_pca(
    base_sub: np.ndarray,
    base_fk_xyz: np.ndarray,
) -> tuple[PCAState, PCAState, np.ndarray, np.ndarray, AxisLim, AxisLim]:
    """Fit PCA exclusively on ground-truth data to anchor the coordinate frame.

    Args:
        base_sub:    [S, A] — subsampled joint targets.
        base_fk_xyz: [S, 3] — FK of subsampled joint targets.

    Returns:
        joint_state, fk_state,
        base_joint_2d [S, 2], base_fk_2d [S, 2],
        joint_lim, fk_lim  (each a pair of (min, max) per axis with ±0.5 pad).
    """
    joint_state: PCAState = pcax.fit(base_sub, n_components=2)
    fk_state: PCAState = pcax.fit(base_fk_xyz, n_components=2)

    base_joint_2d = np.asarray(pcax.transform(joint_state, base_sub), dtype=np.float32)
    base_fk_2d = np.asarray(pcax.transform(fk_state, base_fk_xyz), dtype=np.float32)

    def _lim(pts: np.ndarray) -> AxisLim:
        return (
            (float(pts[:, 0].min()) - 0.5, float(pts[:, 0].max()) + 0.5),
            (float(pts[:, 1].min()) - 0.5, float(pts[:, 1].max()) + 0.5),
        )

    return joint_state, fk_state, base_joint_2d, base_fk_2d, _lim(base_joint_2d), _lim(base_fk_2d)


# ---------------------------------------------------------------------------
# Phase 3b — Matplotlib rendering loop
# ---------------------------------------------------------------------------


def render_frames(
    flow_sub: np.ndarray,
    flow_fk_xyz: np.ndarray,
    joint_state: PCAState,
    fk_state: PCAState,
    base_joint_2d: np.ndarray,
    base_fk_2d: np.ndarray,
    joint_lim: AxisLim,
    fk_lim: AxisLim,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> np.ndarray:
    """Render joint-PCA and FK-PCA flow animation; capture frames via canvas.

    Args:
        flow_sub:     [F, S, A] — subsampled flow predictions.
        flow_fk_xyz:  [F, S, 3] — FK of subsampled flow predictions.
        joint_state:  Static PCA state for joint space.
        fk_state:     Static PCA state for FK XYZ space.
        base_joint_2d: [S, 2] — ground-truth joint manifold in 2-D.
        base_fk_2d:   [S, 2] — ground-truth FK manifold in 2-D.
        joint_lim:    Locked x/y limits for left subplot.
        fk_lim:       Locked x/y limits for right subplot.
        figsize:      Figure size in inches (width, height).

    Returns:
        frames [F, H, W, 3] uint8.
    """
    f = flow_sub.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120)

    # Static background — ground-truth manifold
    axes[0].scatter(base_joint_2d[:, 0], base_joint_2d[:, 1], c="grey", s=1, alpha=0.15, zorder=1)
    axes[1].scatter(base_fk_2d[:, 0], base_fk_2d[:, 1], c="grey", s=1, alpha=0.15, zorder=1)

    # Foreground animated scatters — initialised empty
    joint_sc = axes[0].scatter([], [], c="blue", s=2, alpha=0.2, zorder=2)
    fk_sc = axes[1].scatter([], [], c="red", s=2, alpha=0.2, zorder=2)

    # Lock axes permanently — no dynamic rescaling
    axes[0].set_xlim(*joint_lim[0])
    axes[0].set_ylim(*joint_lim[1])
    axes[0].set_xlabel("pc 1")
    axes[0].set_ylabel("pc 2")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlim(*fk_lim[0])
    axes[1].set_ylim(*fk_lim[1])
    axes[1].set_xlabel("pc 1")
    axes[1].set_ylabel("pc 2")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    frames: list[np.ndarray] = []
    for t in range(f):
        joint_2d = np.asarray(pcax.transform(joint_state, flow_sub[t]), dtype=np.float32)
        fk_2d = np.asarray(pcax.transform(fk_state, flow_fk_xyz[t]), dtype=np.float32)

        joint_sc.set_offsets(np.c_[joint_2d[:, 0], joint_2d[:, 1]])
        fk_sc.set_offsets(np.c_[fk_2d[:, 0], fk_2d[:, 1]])

        t_norm = t / max(f - 1, 1)
        axes[0].set_title(f"Joint PCA  t={t_norm:.2f}")
        axes[1].set_title(f"FK XYZ PCA  t={t_norm:.2f}")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())

    plt.close(fig)
    return np.stack(frames, axis=0)
