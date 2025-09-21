"""Dataset metadata utilities for the Grain pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from crossformer.data.grain import utils


EPS = 1e-8


@dataclass
class ArrayStatistics:
    mean: np.ndarray
    std: np.ndarray
    maximum: np.ndarray
    minimum: np.ndarray
    p99: np.ndarray
    p01: np.ndarray

    def to_json(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "max": self.maximum.tolist(),
            "min": self.minimum.tolist(),
            "p99": self.p99.tolist(),
            "p01": self.p01.tolist(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Sequence[float]]) -> "ArrayStatistics":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            maximum=np.asarray(data["max"], dtype=np.float32),
            minimum=np.asarray(data["min"], dtype=np.float32),
            p99=np.asarray(data["p99"], dtype=np.float32),
            p01=np.asarray(data["p01"], dtype=np.float32),
        )


@dataclass
class DatasetStatistics:
    action: ArrayStatistics
    proprio: dict[str, ArrayStatistics]
    num_transitions: int
    num_trajectories: int

    def to_json(self) -> dict:
        return {
            "action": self.action.to_json(),
            "proprio": {k: v.to_json() for k, v in self.proprio.items()},
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "DatasetStatistics":
        proprio = {
            key: ArrayStatistics.from_json(value)  # type: ignore[arg-type]
            for key, value in data.get("proprio", {}).items()  # type: ignore[union-attr]
        }
        return cls(
            action=ArrayStatistics.from_json(data["action"]),  # type: ignore[arg-type]
            proprio=proprio,
            num_transitions=int(data.get("num_transitions", 0)),
            num_trajectories=int(data.get("num_trajectories", 0)),
        )


class NormalizationType(str):
    NORMAL = "normal"
    BOUNDS = "bounds"


def _stats_from_array(array: np.ndarray) -> ArrayStatistics:
    return ArrayStatistics(
        mean=array.mean(axis=0),
        std=array.std(axis=0),
        maximum=array.max(axis=0),
        minimum=array.min(axis=0),
        p99=np.quantile(array, 0.99, axis=0),
        p01=np.quantile(array, 0.01, axis=0),
    )


def _cache_path(hash_dependencies: Iterable[str], save_dir: str | Path | None) -> Path:
    key = "".join(hash_dependencies).encode("utf-8")
    unique_hash = hashlib.sha256(key, usedforsecurity=False).hexdigest()
    if save_dir is not None:
        return Path(save_dir) / f"dataset_statistics_{unique_hash}.json"
    cache_dir = Path.home() / ".cache" / "crossformer"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"dataset_statistics_{unique_hash}.json"


def compute_dataset_statistics(
    trajectories: Iterable[dict],
    *,
    proprio_keys: Sequence[str],
    hash_dependencies: Sequence[str],
    save_dir: str | Path | None = None,
    force_recompute: bool = False,
) -> DatasetStatistics:
    """Computes statistics for trajectories or loads cached values."""

    cache_path = _cache_path(hash_dependencies, save_dir)
    if cache_path.exists() and not force_recompute:
        with cache_path.open("r") as f:
            return DatasetStatistics.from_json(json.load(f))

    actions = []
    proprio: dict[str, list[np.ndarray]] = {key: [] for key in proprio_keys}
    num_transitions = 0
    num_trajectories = 0

    for traj in trajectories:
        action = utils.ensure_numpy(traj["action"]).astype(np.float32)
        actions.append(action)
        num_transitions += action.shape[0]
        num_trajectories += 1
        obs = traj.get("observation", {})
        for key in proprio_keys:
            if key in obs:
                proprio[key].append(utils.ensure_numpy(obs[key]).astype(np.float32))

    if not actions:
        raise ValueError("Cannot compute statistics from an empty dataset.")
    action_array = np.concatenate(actions, axis=0)
    proprio_stats = {
        key: _stats_from_array(np.concatenate(values, axis=0))
        for key, values in proprio.items()
        if values
    }
    stats = DatasetStatistics(
        action=_stats_from_array(action_array),
        proprio=proprio_stats,
        num_transitions=num_transitions,
        num_trajectories=num_trajectories,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(stats.to_json(), f)
    return stats


def normalize_action_and_proprio(
    traj: dict,
    *,
    metadata: DatasetStatistics,
    normalization_type: str,
    proprio_keys: Sequence[str],
    action_mask: Sequence[bool] | None = None,
    skip_norm_keys: Sequence[str] = (),
) -> dict:
    """Normalizes actions and proprioceptive observations."""

    traj = utils.clone_structure(traj)
    action = utils.ensure_numpy(traj["action"]).astype(np.float32)
    if normalization_type == NormalizationType.NORMAL:
        mean = metadata.action.mean
        std = np.maximum(metadata.action.std, EPS)
        normalized = (action - mean) / std
    elif normalization_type == NormalizationType.BOUNDS:
        span = np.maximum(metadata.action.maximum - metadata.action.minimum, EPS)
        normalized = 2.0 * (action - metadata.action.minimum) / span - 1.0
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    if action_mask is not None:
        action_mask_array = np.asarray(action_mask, dtype=bool)
        if action_mask_array.shape[-1] != normalized.shape[-1]:
            raise ValueError(
                "Length of action mask does not match action dimension."
            )
        normalized = np.where(action_mask_array, normalized, action)
    traj["action"] = normalized

    obs = utils.as_dict(traj.get("observation"))
    for key in proprio_keys:
        if key not in obs or key in skip_norm_keys:
            continue
        if key not in metadata.proprio:
            continue
        value = utils.ensure_numpy(obs[key]).astype(np.float32)
        stats = metadata.proprio[key]
        if normalization_type == NormalizationType.NORMAL:
            mean = stats.mean
            std = np.maximum(stats.std, EPS)
            obs[key] = (value - mean) / std
        else:
            span = np.maximum(stats.maximum - stats.minimum, EPS)
            obs[key] = 2.0 * (value - stats.minimum) / span - 1.0
    traj["observation"] = obs
    return traj

