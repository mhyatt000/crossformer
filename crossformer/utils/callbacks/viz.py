from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from pcax import pca
from PIL import Image

from crossformer.utils.mytyping import Data


def dummy_fk(joints: np.ndarray) -> np.ndarray:
    xyz = np.sin(joints[:, :3])
    return xyz.astype(np.float32)


def render_joint_fk_pca_flow(
    base_joints: np.ndarray,
    flow: np.ndarray,
    figsize: tuple[float, float] = (10.0, 4.5),
) -> np.ndarray:
    """Render joint and FK PCA over flow steps as RGB frames."""
    del base_joints

    last_step = flow[-1]
    last_joints = last_step.reshape(-1, last_step.shape[-1])
    last_xyz = dummy_fk(last_joints)

    print(f"[render] last_step shape={last_step.shape}")
    print(f"[render] last_joints before fit shape={last_joints.shape}")
    print(f"[render] last_xyz before fit shape={last_xyz.shape}")

    joint_state = pca.fit(last_joints, n_components=2)
    fk_state = pca.fit(last_xyz, n_components=2)
    joint_last_2d = np.asarray(pca.transform(joint_state, last_joints), dtype=np.float32)
    fk_last_2d = np.asarray(pca.transform(fk_state, last_xyz), dtype=np.float32)

    print(f"[render] joint_last_2d after transform shape={joint_last_2d.shape}")
    print(f"[render] fk_last_2d after transform shape={fk_last_2d.shape}")

    joint_lim = (
        (float(joint_last_2d[:, 0].min()) - 0.5, float(joint_last_2d[:, 0].max()) + 0.5),
        (float(joint_last_2d[:, 1].min()) - 0.5, float(joint_last_2d[:, 1].max()) + 0.5),
    )
    fk_lim = (
        (float(fk_last_2d[:, 0].min()) - 0.5, float(fk_last_2d[:, 0].max()) + 0.5),
        (float(fk_last_2d[:, 1].min()) - 0.5, float(fk_last_2d[:, 1].max()) + 0.5),
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120)
    joint_scatter = axes[0].scatter([], [], c="blue", alpha=0.5)
    fk_scatter = axes[1].scatter([], [], c="red", alpha=0.5)

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
    n_steps = flow.shape[0]
    for frame_idx in range(n_steps):
        step = flow[frame_idx]
        step_joints = step.reshape(-1, step.shape[-1])
        step_xyz = dummy_fk(step_joints)
        joint_2d = np.asarray(pca.transform(joint_state, step_joints), dtype=np.float32)
        fk_2d = np.asarray(pca.transform(fk_state, step_xyz), dtype=np.float32)

        joint_scatter.set_offsets(np.c_[joint_2d[:, 0], joint_2d[:, 1]])
        fk_scatter.set_offsets(np.c_[fk_2d[:, 0], fk_2d[:, 1]])

        t = frame_idx / max(n_steps - 1, 1)
        axes[0].set_title(f"Joint PCA  t={t:.2f}")
        axes[1].set_title(f"FK XYZ PCA  t={t:.2f}")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    return np.stack(frames, axis=0)


@dataclass
class VizCallback:
    """Render flow trajectories in joint and dummy-FK PCA space."""

    flow_key: tuple[str, ...] = ("action",)
    base_key: tuple[str, ...] = ("observation", "proprio", "joints")
    sample_idx: int = 0
    figsize: tuple[float, float] = (10.0, 4.5)
    fps: int = 12
    dpi: int = 120
    joint_dim: int = 7

    def every(self, batch: Data, step: int, log_every: int) -> np.ndarray | None:
        if (step + 1) % log_every != 0:
            return None
        return self(batch)

    def __call__(self, batch: dict) -> np.ndarray:
        t0 = perf_counter()
        base = self._select_base(self._get(batch, self.base_key))
        flow = self._select_flow(self._get(batch, self.flow_key))
        prep_s = perf_counter() - t0

        t1 = perf_counter()
        frames = render_joint_fk_pca_flow(base, flow, figsize=self.figsize)
        render_s = perf_counter() - t1

        print(f"[VizCallback] Data Prep: {prep_s:.3f}s | Rendering: {render_s:.3f}s")
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
        if base.ndim == 5:
            base = base[self.sample_idx]
        if base.ndim != 4:
            raise ValueError(f"Expected base joints shape [N,B,W,H,A] or [B,W,H,A], got {base.shape}")
        if base.shape[-1] < self.joint_dim:
            raise ValueError(f"Expected base joint dim >= {self.joint_dim}, got {base.shape[-1]}")
        return base[..., : self.joint_dim]

    def _select_flow(self, arr: Any) -> np.ndarray:
        flow = np.asarray(arr, dtype=np.float32)
        if flow.ndim == 6:
            flow = flow[self.sample_idx]
        if flow.ndim != 5:
            raise ValueError(f"Expected flow shape [N,F,B,W,H,A] or [F,B,W,H,A], got {flow.shape}")
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
        dur = max(1, int(round(1000 / fps)))
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
