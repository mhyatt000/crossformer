from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import pcax
from pcax.pca import PCAState
from PIL import Image

from crossformer.utils.mytyping import Data

# (x_min, x_max), (y_min, y_max)
_AxisLim = tuple[tuple[float, float], tuple[float, float]]


# ---------------------------------------------------------------------------
# FK factory
# ---------------------------------------------------------------------------


def make_fk_fn(link: str = "link_eef") -> Callable[[np.ndarray], np.ndarray]:
    try:
        from xgym import calibrate
        from xgym.calibrate.urdf.robot import urdf as urdf_path
        import yourdfpy
    except ModuleNotFoundError as e:
        raise RuntimeError("Viz FK requires `yourdfpy` and `xgym` in the runtime environment.") from e

    robot = yourdfpy.URDF.load(urdf_path, mesh_dir=calibrate.urdf.robot.DNAME / "assets")

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


def _prep_data(
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


def _compute_fk(
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


def _fit_pca(
    base_sub: np.ndarray,
    base_fk_xyz: np.ndarray,
) -> tuple[PCAState, PCAState, np.ndarray, np.ndarray, _AxisLim, _AxisLim]:
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

    def _lim(pts: np.ndarray) -> _AxisLim:
        return (
            (float(pts[:, 0].min()) - 0.5, float(pts[:, 0].max()) + 0.5),
            (float(pts[:, 1].min()) - 0.5, float(pts[:, 1].max()) + 0.5),
        )

    return joint_state, fk_state, base_joint_2d, base_fk_2d, _lim(base_joint_2d), _lim(base_fk_2d)


# ---------------------------------------------------------------------------
# Phase 3b — Matplotlib rendering loop
# ---------------------------------------------------------------------------


def _render_frames(
    flow_sub: np.ndarray,
    flow_fk_xyz: np.ndarray,
    joint_state: PCAState,
    fk_state: PCAState,
    base_joint_2d: np.ndarray,
    base_fk_2d: np.ndarray,
    joint_lim: _AxisLim,
    fk_lim: _AxisLim,
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


# ---------------------------------------------------------------------------
# Phase 4 — VizCallback dataclass
# ---------------------------------------------------------------------------


@dataclass
class VizCallback:
    """Render flow trajectories in joint and URDF FK PCA space."""

    flow_key: tuple[str, ...] = ("action",)
    base_key: tuple[str, ...] = ("observation", "proprio", "joints")
    sample_idx: int = 0
    figsize: tuple[float, float] = (12.0, 6.0)
    fps: int = 12
    dpi: int = 120
    joint_dim: int = 7
    fk_link: str = "link_eef"
    _fk_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def every(self, batch: Data, step: int, log_every: int) -> np.ndarray | None:
        if (step + 1) % log_every != 0:
            return None
        return self(batch)

    def __call__(self, batch: dict) -> np.ndarray:
        base = self._select_base(self._get(batch, self.base_key))
        flow = self._select_flow(self._get(batch, self.flow_key))

        t0 = perf_counter()
        base_sub, flow_sub = _prep_data(base, flow)
        if self._fk_fn is None:
            self._fk_fn = make_fk_fn(link=self.fk_link)
        base_fk_xyz, flow_fk_xyz = _compute_fk(base_sub, flow_sub, self._fk_fn)
        t1 = perf_counter()
        print(f"[VizCallback] Data Prep & FK Time: {t1 - t0:.3f}s")

        joint_state, fk_state, base_joint_2d, base_fk_2d, joint_lim, fk_lim = _fit_pca(base_sub, base_fk_xyz)
        frames = _render_frames(
            flow_sub,
            flow_fk_xyz,
            joint_state,
            fk_state,
            base_joint_2d,
            base_fk_2d,
            joint_lim,
            fk_lim,
            self.figsize,
        )
        t2 = perf_counter()
        print(f"[VizCallback] Render Loop Time: {t2 - t1:.3f}s")

        return frames

    def save(self, frames: np.ndarray, path: str | Path, fps: int | None = None) -> Path:
        path = Path(path)
        fps = self.fps if fps is None else fps
        if path.suffix.lower() == ".gif":
            return self._save_gif(frames, path, fps)
        if path.suffix.lower() == ".mp4":
            self._save_mp4(frames, path, fps)
            return path
        raise ValueError(f"Expected '.gif' or '.mp4', got {path}")

    def _select_base(self, arr: Any) -> np.ndarray:
        base = np.asarray(arr, dtype=np.float32)
        if base.ndim < 2:
            raise ValueError(f"Expected base joints ndim >= 2, got {base.shape}")
        if base.shape[-1] < self.joint_dim:
            raise ValueError(f"Expected base joint dim >= {self.joint_dim}, got {base.shape[-1]}")
        return base[..., : self.joint_dim]

    def _select_flow(self, arr: Any) -> np.ndarray:
        flow = np.asarray(arr, dtype=np.float32)
        if flow.ndim < 2:
            raise ValueError(f"Expected flow ndim >= 2, got {flow.shape}")
        if flow.shape[-1] < self.joint_dim:
            raise ValueError(f"Expected flow dim >= {self.joint_dim}, got {flow.shape[-1]}")
        return flow[..., : self.joint_dim]

    def _get(self, batch: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        cur: Any = batch
        for key in path:
            cur = cur[key]
        return cur

    def _save_gif(self, frames: np.ndarray, path: Path, fps: int) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        dur = max(1, round(1000 / fps))
        imgs = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        imgs[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            duration=[dur] * len(imgs),
            loop=0,
            disposal=2,
            optimize=False,
        )
        return path

    def _save_mp4(self, frames: np.ndarray, path: Path, fps: int) -> None:
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("FFMpegWriter is unavailable; cannot write mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames.shape[1:3]
        fig = plt.figure(figsize=(w / self.dpi, h / self.dpi), dpi=self.dpi)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        im = ax.imshow(frames[0])
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, str(path), self.dpi):
            for frame in frames:
                im.set_data(frame)
                writer.grab_frame()
        plt.close(fig)
