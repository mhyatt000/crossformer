from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from crossformer.utils.mytyping import Data
from crossformer.viz.flow_pca import compute_fk, fit_pca, make_fk_fn, prep_data, render_frames


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
        base_sub, flow_sub = prep_data(base, flow)
        if self._fk_fn is None:
            self._fk_fn = make_fk_fn(link=self.fk_link)
        base_fk_xyz, flow_fk_xyz = compute_fk(base_sub, flow_sub, self._fk_fn)
        t1 = perf_counter()
        print(f"[VizCallback] Data Prep & FK Time: {t1 - t0:.3f}s")

        joint_state, fk_state, base_joint_2d, base_fk_2d, joint_lim, fk_lim = fit_pca(base_sub, base_fk_xyz)
        frames = render_frames(
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
