"""Flow-PCA eval callback: render flow trajectories in joint and FK PCA space."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from PIL import Image

from crossformer.utils.callbacks.adapt import adapt_viz_batch
from crossformer.utils.callbacks.base import EvalContext, maybe_getpath
from crossformer.viz.flow_pca import compute_fk, fit_pca, make_fk_fn, prep_data, render_frames
import wandb


@dataclass
class FlowPCACallback:
    """Render flow trajectories in joint and URDF FK PCA space."""

    name: str = "flow_pca"
    every: int = 5000
    flow_key: tuple[str, ...] = ("predict",)
    base_key: tuple[str, ...] = ("act", "base")
    robot_xyz_flow_key: tuple[str, ...] = ("robot_xyz", "predict")
    robot_xyz_base_key: tuple[str, ...] = ("robot_xyz", "base")
    human_xyz_flow_key: tuple[str, ...] = ("human_xyz", "predict")
    human_xyz_base_key: tuple[str, ...] = ("human_xyz", "base")
    sample_idx: int = 0
    figsize: tuple[float, float] = (12.0, 6.0)
    fps: int = 12
    dpi: int = 120
    joint_dim: int = 7
    fk_link: str = "link_eef"
    max_pts: int | None = None
    _fk_fn: Callable[[np.ndarray], np.ndarray] | None = field(default=None, init=False, repr=False)

    def __call__(self, ctx: EvalContext) -> dict[str, Any]:
        viz_batch, _ = adapt_viz_batch(ctx.batch, ctx.pred_flow)
        if viz_batch is None:
            return {}
        frames = self.render(viz_batch)
        return {"video": wandb.Video(np.moveaxis(frames, -1, 1), fps=self.fps)}

    def render(self, batch: dict) -> np.ndarray:
        """Render PCA flow frames from an already-adapted viz batch."""
        base = self._maybe_select(batch, self.base_key, self._select_base)
        flow = self._maybe_select(batch, self.flow_key, self._select_flow)
        robot_xyz_base = self._maybe_select(batch, self.robot_xyz_base_key, self._select_xyz)
        robot_xyz_flow = self._maybe_select(batch, self.robot_xyz_flow_key, self._select_xyz)
        human_xyz_base = self._maybe_select(batch, self.human_xyz_base_key, self._select_xyz)
        human_xyz_flow = self._maybe_select(batch, self.human_xyz_flow_key, self._select_xyz)

        t0 = perf_counter()
        base_sub = flow_sub = None
        base_fk_xyz = flow_fk_xyz = None
        if base is not None and flow is not None:
            base_sub, flow_sub = prep_data(base, flow, max_pts=self.max_pts)
            if self._fk_fn is None:
                self._fk_fn = make_fk_fn(link=self.fk_link)
            base_fk_xyz, flow_fk_xyz = compute_fk(base_sub, flow_sub, self._fk_fn)

        robot_xyz_base_sub = robot_xyz_flow_sub = None
        if robot_xyz_base is not None and robot_xyz_flow is not None:
            robot_xyz_base_sub, robot_xyz_flow_sub = prep_data(robot_xyz_base, robot_xyz_flow)

        human_xyz_base_sub = human_xyz_flow_sub = None
        if human_xyz_base is not None and human_xyz_flow is not None:
            human_xyz_base_sub, human_xyz_flow_sub = prep_data(human_xyz_base, human_xyz_flow)

        if base_fk_xyz is None and robot_xyz_base_sub is None and human_xyz_base_sub is None:
            raise ValueError("Expected robot joints or xyz inputs for FlowPCACallback")

        base_xyz = [x for x in (base_fk_xyz, robot_xyz_base_sub, human_xyz_base_sub) if x is not None]
        base_xyz = np.concatenate(base_xyz, axis=0)
        t1 = perf_counter()
        print(f"[FlowPCACallback] Data Prep & FK Time: {t1 - t0:.3f}s")

        joint_state, fk_state, base_joint_2d, base_fk_2d, joint_lim, fk_lim = fit_pca(base_sub, base_xyz)
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
            robot_xyz_flow=robot_xyz_flow_sub,
            human_flow_xyz=human_xyz_flow_sub,
        )
        t2 = perf_counter()
        print(f"[FlowPCACallback] Render Loop Time: {t2 - t1:.3f}s")

        return frames

    def save(self, frames: np.ndarray, path: str | Path, fps: int | None = None) -> Path:
        path = Path(path)
        fps = self.fps if fps is None else fps
        if path.suffix.lower() == ".gif":
            return self._save_gif(frames, path, fps)
        raise ValueError(f"Expected '.gif', got {path}")

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

    def _select_xyz(self, arr: Any) -> np.ndarray:
        xyz = np.asarray(arr, dtype=np.float32)
        if xyz.ndim < 2:
            raise ValueError(f"Expected xyz ndim >= 2, got {xyz.shape}")
        if xyz.shape[-1] < 3:
            raise ValueError(f"Expected xyz dim >= 3, got {xyz.shape[-1]}")
        return xyz[..., :3]

    def _maybe_select(
        self,
        batch: dict,
        path: tuple[str, ...],
        fn: Callable[[Any], np.ndarray],
    ) -> np.ndarray | None:
        arr = maybe_getpath(batch, path)
        if arr is None:
            return None
        return fn(arr)

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
