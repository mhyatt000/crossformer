"""DOF-aware denormalization of bundled action batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from crossformer.data.grain.metadata import ArrayStatistics, DatasetStatistics
from crossformer.embody import DOF, KP3DC, KP3DC_HAND, MASK_ID
from crossformer.utils.jax_utils import jax2str

# kp DOF name -> (stats key, flat index): kp_{link}_{x|y|z} into the 42-dim
# kp3dc_robot block, kp3dc_hand_{j}_{x|y|z} into the 63-dim kp3dc_hand block.
# Per-view copies share DOF ids and stats; act.view picks the camera frame.
_KP_STATS_IDX: dict[str, tuple[str, int]] = {
    name: (key, i)
    for key, part in (("kp3dc_robot", KP3DC), ("kp3dc_hand", KP3DC_HAND))
    for i, name in enumerate(part.dof_names)
}


def dof_name(dof_id: int) -> str:
    """Map a DOF id back to its name (``dof_<id>`` when unknown)."""
    for name, idx in DOF.items():
        if idx == dof_id:
            return name
    return f"dof_{dof_id}"


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
                name = dof_name(dof_id)
                xs = arr[b, :, :, a]
                stat = self._dof_array_stats(stats, name)
                if stat is not None:
                    xs = stat.unnormalize(xs)
                vals.setdefault(name, []).append(xs.reshape(-1))

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
            name = dof_name(dof_id)
            xs = chunk[:, a].copy()
            stat = self._dof_array_stats(stats, name)
            if stat is not None:
                xs = stat.unnormalize(xs)
            lines[name] = xs
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
            stat = self._dof_array_stats(stats, dof_name(dof_id))
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
        kp = _KP_STATS_IDX.get(dof_name)
        if kp is not None:
            return kp if kp[0] in stats else (None, 0)

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
        return dof_name(dof_id)
