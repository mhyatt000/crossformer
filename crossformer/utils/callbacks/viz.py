from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from crossformer.data.grain.metadata import DatasetStatistics
from crossformer.embody import DOF, MASK_ID
from crossformer.utils.jax_utils import jax2str
from crossformer.utils.mytyping import Data
from crossformer.viz.flow_pca import compute_fk, fit_pca, make_fk_fn, prep_data, render_frames
import wandb


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


@dataclass
class HistVizCallback(VizCallback):
    """Log action histograms by DOF for data and predictions."""

    stats: Mapping[str, DatasetStatistics | Mapping[str, Any]] | None = None
    data_key: tuple[str, ...] = ("act", "base")
    dof_key: tuple[str, ...] = ("act", "id")
    ds_key: tuple[str, ...] = ("info", "dataset_name")
    pred_key: tuple[str, ...] = ("predict",)

    def __call__(
        self, batch: Mapping[str, Any], predict: Mapping[str, Any] | np.ndarray | None = None
    ) -> dict[str, Any]:
        data = self._get(batch, self.data_key)
        dof_ids = self._get(batch, self.dof_key)
        ds_names = self._decode_dataset_names(self._get(batch, self.ds_key))
        predict = self._get(predict, self.pred_key) if isinstance(predict, Mapping) else predict

        out = {
            "data": self._hist_tree(data, dof_ids, ds_names),
        }
        if predict is not None:
            out["predict"] = self._hist_tree(predict, dof_ids, ds_names, horizon=data.shape[-2])
        return out

    def _hist_tree(
        self,
        arr: Any,
        dof_ids: Any,
        ds_names: list[str],
        horizon: int | None = None,
    ) -> dict[str, wandb.Histogram]:
        vals = self._unnorm(arr, dof_ids, ds_names, horizon=horizon)
        out = {}
        for dof_name, xs in vals.items():
            if xs.size == 0:
                continue
            out[dof_name] = wandb.Histogram(xs)
        return out

    def _unnorm(
        self,
        arr: Any,
        dof_ids: Any,
        ds_names: list[str],
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        arr = np.asarray(arr, dtype=np.float32)
        dof_ids = np.asarray(dof_ids)
        arr = self._reshape_actions(arr, dof_ids.shape[-1], horizon=horizon)
        dof_ids = self._reshape_dof_ids(dof_ids, arr.shape[0])
        if len(ds_names) == 1 and arr.shape[0] > 1:
            ds_names = ds_names * arr.shape[0]
        if len(ds_names) != arr.shape[0]:
            raise ValueError(f"Expected {arr.shape[0]} dataset names, got {len(ds_names)}")
        vals: dict[str, list[np.ndarray]] = {}

        for b, ds_name in enumerate(ds_names):
            stats = self._action_stats(ds_name)
            for a, dof_id in enumerate(dof_ids[b]):
                dof_id = int(dof_id)
                if dof_id == MASK_ID:
                    continue
                dof_name = self._dof_name(dof_id)
                stat = self._dof_stat(stats, dof_name)
                xs = arr[b, :, :, a]
                if stat is not None:
                    mean, std = stat
                    xs = xs * std + mean
                vals.setdefault(dof_name, []).append(xs.reshape(-1))

        return {k: np.concatenate(v).astype(np.float32) for k, v in vals.items()}

    def _reshape_actions(self, arr: np.ndarray, max_a: int, horizon: int | None = None) -> np.ndarray:
        if arr.ndim == 4:
            return arr
        if arr.ndim != 3:
            raise ValueError(f"Expected action ndim 3 or 4, got {arr.shape}")
        if horizon is None:
            if arr.shape[-1] % max_a != 0:
                raise ValueError(f"Could not infer horizon from {arr.shape} and max_a={max_a}")
            horizon = arr.shape[-1] // max_a
        if horizon * max_a != arr.shape[-1]:
            raise ValueError(f"Expected last dim {horizon * max_a}, got {arr.shape[-1]}")
        return arr.reshape(arr.shape[0], arr.shape[1], horizon, max_a)

    def _reshape_dof_ids(self, dof_ids: np.ndarray, batch_size: int) -> np.ndarray:
        if dof_ids.ndim == 1:
            return np.broadcast_to(dof_ids[None, :], (batch_size, dof_ids.shape[0]))
        if dof_ids.ndim != 2:
            raise ValueError(f"Expected dof_ids ndim 1 or 2, got {dof_ids.shape}")
        return dof_ids

    def _decode_dataset_names(self, arr: Any) -> list[str]:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return [jax2str(arr).rstrip("\x00")]
        names = [jax2str(x).rstrip("\x00") for x in arr]
        known = sorted({x for x in names if x})
        if len(known) == 1:
            return [x or known[0] for x in names]
        return names

    def _action_stats(self, ds_name: str) -> Mapping[str, Any]:
        if self.stats is None:
            raise ValueError("HistVizCallback.stats is required")
        if not ds_name:
            raise KeyError("dataset name is empty")
        stats = self.stats[ds_name]
        if isinstance(stats, DatasetStatistics):
            return stats.action
        return stats["action"]

    def _dof_stat(self, stats: Mapping[str, Any], dof_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        part, idx = self._dof_part_idx(stats, dof_name)
        if part is None:
            return None
        stat = stats[part]
        mean = np.asarray(stat.mean if hasattr(stat, "mean") else stat["mean"], dtype=np.float32).reshape(-1)[idx]
        std = np.asarray(stat.std if hasattr(stat, "std") else stat["std"], dtype=np.float32).reshape(-1)[idx]
        mask = stat.mask if hasattr(stat, "mask") else stat.get("mask")
        if mask is not None and not bool(np.asarray(mask).reshape(-1)[idx]):
            return None
        return mean, std

    def _dof_part_idx(self, stats: Mapping[str, Any], dof_name: str) -> tuple[str | None, int]:
        if dof_name.startswith("j") and dof_name[1:].isdigit():
            return ("joints", int(dof_name[1:])) if "joints" in stats else (None, 0)
        if dof_name == "gripper":
            return ("gripper", 0) if "gripper" in stats else (None, 0)

        pos_idx = {"ee_x": 0, "ee_y": 1, "ee_z": 2}
        ori_idx = {"ee_rx": 0, "ee_ry": 1, "ee_rz": 2}
        if dof_name in pos_idx:
            if "pose" in stats:
                return "pose", pos_idx[dof_name]
            if "position" in stats:
                return "position", pos_idx[dof_name]
        if dof_name in ori_idx:
            if "pose" in stats:
                return "pose", 3 + ori_idx[dof_name]
            if "orientation" in stats:
                return "orientation", ori_idx[dof_name]
        return None, 0

    def _dof_name(self, dof_id: int) -> str:
        for name, idx in DOF.items():
            if idx == dof_id:
                return name
        return f"dof_{dof_id}"
