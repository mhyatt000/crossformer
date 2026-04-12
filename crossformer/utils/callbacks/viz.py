from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from crossformer.cn.base import default
from crossformer.data.grain.metadata import ArrayStatistics, DatasetStatistics
from crossformer.embody import DOF, MASK_ID
from crossformer.utils.jax_utils import jax2str
from crossformer.utils.mytyping import Data
from crossformer.viz.flow_pca import compute_fk, fit_pca, make_fk_fn, prep_data, render_frames
import wandb


@dataclass
class VizConfig:
    """Config for ``VizCallback``."""

    every: int = 5000
    flow_key: tuple[str, ...] = default(("predict",))
    base_key: tuple[str, ...] = default(("act", "base"))
    robot_xyz_flow_key: tuple[str, ...] = default(("robot_xyz", "predict"))
    robot_xyz_base_key: tuple[str, ...] = default(("robot_xyz", "base"))
    human_xyz_flow_key: tuple[str, ...] = default(("human_xyz", "predict"))
    human_xyz_base_key: tuple[str, ...] = default(("human_xyz", "base"))
    sample_idx: int = 0
    figsize: tuple[float, float] = (12.0, 6.0)
    fps: int = 12
    dpi: int = 120
    joint_dim: int = 7
    fk_link: str = "link_eef"
    max_pts: int | None = None

    def create(self) -> VizCallback:
        return VizCallback(
            flow_key=self.flow_key,
            base_key=self.base_key,
            robot_xyz_flow_key=self.robot_xyz_flow_key,
            robot_xyz_base_key=self.robot_xyz_base_key,
            human_xyz_flow_key=self.human_xyz_flow_key,
            human_xyz_base_key=self.human_xyz_base_key,
            sample_idx=self.sample_idx,
            figsize=self.figsize,
            fps=self.fps,
            dpi=self.dpi,
            joint_dim=self.joint_dim,
            fk_link=self.fk_link,
            max_pts=self.max_pts,
        )


@dataclass
class VizCallback:
    """Render flow trajectories in joint and URDF FK PCA space."""

    flow_key: tuple[str, ...] = ("action",)
    base_key: tuple[str, ...] = ("observation", "proprio", "joints")
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
    _fk_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def every(self, batch: Data, step: int, log_every: int) -> np.ndarray | None:
        if (step + 1) % log_every != 0:
            return None
        return self(batch)

    def __call__(self, batch: dict) -> np.ndarray:
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
            raise ValueError("Expected robot joints or xyz inputs for VizCallback")

        base_xyz = [x for x in (base_fk_xyz, robot_xyz_base_sub, human_xyz_base_sub) if x is not None]
        base_xyz = np.concatenate(base_xyz, axis=0)
        t1 = perf_counter()
        print(f"[VizCallback] Data Prep & FK Time: {t1 - t0:.3f}s")

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
        print(f"[VizCallback] Render Loop Time: {t2 - t1:.3f}s")

        return frames

    def _call_v2(self, batch: dict) -> np.ndarray:
        """Simplified path for the split-by-bodypart batch layout.

        Expects:
            batch["actions"][{joints,position,...}]   (B, W, H, n)
            batch["pred"][{joints,position,...}]      (F, B, W, H, n)
            batch["mask"]["embodiment"][{single,human_single}]  (B, 1) bool
        """
        actions = batch["actions"]
        pred = batch["pred"]
        emb_mask = batch["mask"]["embodiment"]

        def _slice(arr: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
            keep = np.asarray(mask).reshape(-1).astype(bool)
            return np.take(np.asarray(arr), np.flatnonzero(keep), axis=axis)

        t0 = perf_counter()
        base_sub = flow_sub = base_fk_xyz = flow_fk_xyz = None
        robot_xyz_base_sub = robot_xyz_flow_sub = None
        human_xyz_base_sub = human_xyz_flow_sub = None

        if "joints" in actions and "joints" in pred and "single" in emb_mask:
            j_base = _slice(actions["joints"], emb_mask["single"], axis=0)
            j_flow = _slice(pred["joints"], emb_mask["single"], axis=1)
            if j_base.size:
                base_sub, flow_sub = prep_data(j_base, j_flow, max_pts=self.max_pts)
                if self._fk_fn is None:
                    self._fk_fn = make_fk_fn(link=self.fk_link)
                base_fk_xyz, flow_fk_xyz = compute_fk(base_sub, flow_sub, self._fk_fn)

        if "position" in actions and "position" in pred and "single" in emb_mask:
            r_base = _slice(actions["position"], emb_mask["single"], axis=0)
            r_flow = _slice(pred["position"], emb_mask["single"], axis=1)
            if r_base.size:
                robot_xyz_base_sub, robot_xyz_flow_sub = prep_data(r_base, r_flow, max_pts=self.max_pts)

        if "position" in actions and "position" in pred and "human_single" in emb_mask:
            h_base = _slice(actions["position"], emb_mask["human_single"], axis=0)
            h_flow = _slice(pred["position"], emb_mask["human_single"], axis=1)
            if h_base.size:
                human_xyz_base_sub, human_xyz_flow_sub = prep_data(h_base, h_flow, max_pts=self.max_pts)

        if base_fk_xyz is None and robot_xyz_base_sub is None and human_xyz_base_sub is None:
            raise ValueError("Expected joints or position inputs for VizCallback._call_v2")

        base_xyz = np.concatenate(
            [x for x in (base_fk_xyz, robot_xyz_base_sub, human_xyz_base_sub) if x is not None], axis=0
        )
        t1 = perf_counter()
        print(f"[VizCallback v2] Data Prep & FK Time: {t1 - t0:.3f}s")

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
        print(f"[VizCallback v2] Render Loop Time: {t2 - t1:.3f}s")

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

    def _select_xyz(self, arr: Any) -> np.ndarray:
        xyz = np.asarray(arr, dtype=np.float32)
        if xyz.ndim < 2:
            raise ValueError(f"Expected xyz ndim >= 2, got {xyz.shape}")
        if xyz.shape[-1] < 3:
            raise ValueError(f"Expected xyz dim >= 3, got {xyz.shape[-1]}")
        return xyz[..., :3]

    def _get(self, batch: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        cur: Any = batch
        for key in path:
            cur = cur[key]
        return cur

    def _maybe_get(self, batch: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
        cur: Any = batch
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                return None
            cur = cur[key]
        return cur

    def _maybe_select(
        self,
        batch: Mapping[str, Any],
        path: tuple[str, ...],
        fn: Callable[[Any], np.ndarray],
    ) -> np.ndarray | None:
        arr = self._maybe_get(batch, path)
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
class ActionBatchDenormalizer:
    """Denormalize action batches by DOF."""

    stats: Mapping[str, DatasetStatistics | Mapping[str, Any]] | None = None

    def denormalize(
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
                xs = arr[b, :, :, a]
                stat = self._dof_array_stats(stats, dof_name)
                if stat is not None:
                    xs = stat.unnormalize(xs)
                vals.setdefault(dof_name, []).append(xs.reshape(-1))

        return {k: np.concatenate(v).astype(np.float32) for k, v in vals.items()}

    def sample_lines(
        self,
        arr: Any,
        dof_ids: Any,
        ds_names: list[str],
        sample_idx: int = 0,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Denormalize one sample and return ``{dof_name: (H,)}`` arrays."""
        arr = np.asarray(arr, dtype=np.float32)
        dof_ids = np.asarray(dof_ids)
        if arr.ndim == 3 and horizon is None:
            arr = arr[:, None, :, :]
        arr = self._reshape_actions(arr, dof_ids.shape[-1], horizon=horizon)
        dof_ids = self._reshape_dof_ids(dof_ids, arr.shape[0])
        if len(ds_names) == 1 and arr.shape[0] > 1:
            ds_names = ds_names * arr.shape[0]
        if len(ds_names) != arr.shape[0]:
            raise ValueError(f"Expected {arr.shape[0]} dataset names, got {len(ds_names)}")

        b = min(sample_idx, arr.shape[0] - 1)
        stats = self._action_stats(ds_names[b])
        chunk = arr[b].mean(axis=0)

        lines: dict[str, np.ndarray] = {}
        for a, dof_id in enumerate(dof_ids[b]):
            dof_id = int(dof_id)
            if dof_id == MASK_ID:
                continue
            dof_name = self._dof_name(dof_id)
            xs = chunk[:, a].copy()
            stat = self._dof_array_stats(stats, dof_name)
            if stat is not None:
                xs = stat.unnormalize(xs)
            lines[dof_name] = xs
        return lines

    def denormalize_slot(
        self,
        arr: Any,
        dof_ids: Any,
        ds_name: str,
    ) -> np.ndarray:
        """Denormalize one slot-ordered action vector using DOF ids."""
        xs = np.asarray(arr, dtype=np.float32).copy()
        ids = np.asarray(dof_ids).reshape(-1)
        if xs.ndim != 1:
            raise ValueError(f"Expected arr shape (A,), got {xs.shape}")
        if ids.shape[0] != xs.shape[0]:
            raise ValueError(f"Expected dof_ids len {xs.shape[0]}, got {ids.shape}")

        from crossformer.embody import NO_NORM_DOF_IDS

        stats = self._action_stats(ds_name)
        for i, dof_id in enumerate(ids):
            dof_id = int(dof_id)
            if dof_id == MASK_ID or dof_id in NO_NORM_DOF_IDS:
                continue
            stat = self._dof_array_stats(stats, self._dof_name(dof_id))
            if stat is None:
                continue
            xs[i] = np.asarray(stat.unnormalize(np.asarray([xs[i]], dtype=np.float32)))[0]
        return xs

    def decode_dataset_names(self, arr: Any) -> list[str]:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return [jax2str(arr).rstrip("\x00")]
        names = [jax2str(x).rstrip("\x00") for x in arr]
        known = sorted({x for x in names if x})
        if len(known) == 1:
            return [x or known[0] for x in names]
        return names

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

    def _action_stats(self, ds_name: str) -> Mapping[str, Any]:
        if self.stats is None:
            raise ValueError("ActionBatchDenormalizer.stats is required")
        if not ds_name:
            raise KeyError("dataset name is empty")
        stats = self.stats[ds_name]
        if isinstance(stats, DatasetStatistics):
            return stats.action
        return stats["action"]

    def _dof_stat(self, stats: Mapping[str, Any], dof_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        stat = self._dof_array_stats(stats, dof_name)
        if stat is None:
            return None
        return np.asarray(stat.mean, dtype=np.float32), np.asarray(stat.std, dtype=np.float32)

    def _dof_array_stats(self, stats: Mapping[str, Any], dof_name: str) -> ArrayStatistics | None:
        part, idx = self._dof_part_idx(stats, dof_name)
        if part is None:
            return None
        stat = stats[part]
        if not isinstance(stat, ArrayStatistics):
            stat = ArrayStatistics.from_json(stat)
        mean = np.asarray(stat.mean, dtype=np.float32).reshape(-1)
        std = np.asarray(stat.std, dtype=np.float32).reshape(-1)
        mask = None if stat.mask is None else np.asarray(stat.mask).reshape(-1)
        if mask is not None and not bool(mask[idx]):
            return None
        return ArrayStatistics(
            mean=mean[idx : idx + 1],
            std=std[idx : idx + 1],
            minimum=np.asarray(stat.minimum, dtype=np.float32).reshape(-1)[idx : idx + 1],
            maximum=np.asarray(stat.maximum, dtype=np.float32).reshape(-1)[idx : idx + 1],
            mask=None if mask is None else mask[idx : idx + 1],
            p99=None if stat.p99 is None else np.asarray(stat.p99, dtype=np.float32).reshape(-1)[idx : idx + 1],
            p01=None if stat.p01 is None else np.asarray(stat.p01, dtype=np.float32).reshape(-1)[idx : idx + 1],
        )

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


@dataclass
class HistVizCallback(VizCallback):
    """Log action histograms by DOF for data and predictions."""

    stats: Mapping[str, DatasetStatistics | Mapping[str, Any]] | None = None
    data_key: tuple[str, ...] = ("act", "base")
    dof_key: tuple[str, ...] = ("act", "id")
    ds_key: tuple[str, ...] = ("info", "dataset_name")
    pred_key: tuple[str, ...] = ("predict",)
    denorm: ActionBatchDenormalizer = field(init=False)

    def __post_init__(self):
        self.denorm = ActionBatchDenormalizer(self.stats)

    def __call__(
        self, batch: Mapping[str, Any], predict: Mapping[str, Any] | np.ndarray | None = None
    ) -> dict[str, Any]:
        data = self._get(batch, self.data_key)
        dof_ids = self._get(batch, self.dof_key)
        ds_names = self.denorm.decode_dataset_names(self._get(batch, self.ds_key))
        predict = self._get(predict, self.pred_key) if isinstance(predict, Mapping) else predict

        out = {"data": self._hist_tree(data, dof_ids, ds_names)}
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
        vals = self.denorm.denormalize(arr, dof_ids, ds_names, horizon=horizon)
        out = {}
        for dof_name, xs in vals.items():
            if xs.size == 0:
                continue
            out[dof_name] = wandb.Histogram(xs)
        return out


@dataclass
class ChunkVizCallback(HistVizCallback):
    """Plot denormalized action chunks: ground truth (dashed) vs predicted (solid).

    One subplot per DOF so data vs prediction is easy to compare.
    Returns ``dict[str, wandb.Image]`` keyed by DOF name for logging.
    """

    sample_idx: int = 0
    subplot_w: float = 3.5  # width per subplot
    subplot_h: float = 2.5  # height per subplot
    max_cols: int = 4
    dpi: int = 120

    def __call__(
        self, batch: Mapping[str, Any], predict: Mapping[str, Any] | np.ndarray | None = None
    ) -> dict[str, wandb.Image]:
        data = self._get(batch, self.data_key)
        dof_ids = self._get(batch, self.dof_key)
        ds_names = self.denorm.decode_dataset_names(self._get(batch, self.ds_key))
        predict = self._get(predict, self.pred_key) if isinstance(predict, Mapping) else predict

        data_lines = self.denorm.sample_lines(data, dof_ids, ds_names, sample_idx=self.sample_idx)
        pred_lines = (
            self.denorm.sample_lines(
                predict,
                dof_ids,
                ds_names,
                sample_idx=self.sample_idx,
                horizon=np.asarray(data).shape[-2],
            )
            if predict is not None
            else None
        )
        return self._render(data_lines, pred_lines)

    def _render(
        self,
        data_lines: dict[str, np.ndarray],
        pred_lines: dict[str, np.ndarray] | None,
    ) -> dict[str, wandb.Image]:
        names = list(data_lines)
        n = len(names)
        if n == 0:
            return {}
        ncols = min(n, self.max_cols)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.subplot_w * ncols, self.subplot_h * nrows),
            dpi=self.dpi,
            squeeze=False,
        )
        for i, name in enumerate(names):
            ax = axes[i // ncols, i % ncols]
            h = np.arange(len(data_lines[name]))
            ax.plot(h, data_lines[name], "--", color="C0", label="data")
            if pred_lines is not None and name in pred_lines:
                ax.plot(h, pred_lines[name], "-", color="C1", label="pred")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("H", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)
        # hide unused axes
        for i in range(n, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)
        fig.tight_layout()

        # rasterise to numpy
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape((*fig.canvas.get_width_height()[::-1], 4))[..., :3].copy()
        plt.close(fig)

        # one combined image + per-dof crops are overkill; just return the grid
        return {"grid": wandb.Image(frame)}
