"""Dataset metadata utilities for the Grain pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import grain
import jax
import numpy as np
from tqdm import tqdm

from crossformer.utils import databrief
from crossformer.utils.jax_utils import cpu
from crossformer.utils.mytyping import Data

log = logging.getLogger(__name__)

EPS = 1e-8


class OnlineStats:
    def __init__(self, shape, dtype=np.float64):
        self.n = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=dtype)
        self.M2 = np.zeros(shape, dtype=dtype)
        self.min = np.full(shape, np.inf, dtype=dtype)
        self.max = np.full(shape, -np.inf, dtype=dtype)

    def update(self, x: np.ndarray, mask: np.ndarray | None = None):
        x = np.asarray(x, dtype=self.mean.dtype)
        if mask is None:
            mask = np.ones_like(x, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != x.shape:
                mask = np.broadcast_to(mask, x.shape)

        # min / max (elementwise)
        self.min = np.where(mask, np.minimum(self.min, x), self.min)
        self.max = np.where(mask, np.maximum(self.max, x), self.max)

        # Welford (vectorized, per-dim)
        self.n += mask.astype(np.int64)
        delta = x - self.mean
        denom = np.maximum(self.n, 1)
        self.mean = np.where(mask, self.mean + delta / denom, self.mean)
        delta2 = x - self.mean
        self.M2 = np.where(mask, self.M2 + delta * delta2, self.M2)

    def finalize(self, sample_std=False):
        valid = self.n > 0
        if sample_std:
            denom = np.maximum(self.n - 1, 1)
            std = np.where(self.n > 1, np.sqrt(self.M2 / denom), 0)
        else:
            denom = np.maximum(self.n, 1)
            std = np.where(valid, np.sqrt(self.M2 / denom), 0)

        return {
            "mean": self.mean,
            "std": std,
            "minimum": np.where(valid, self.min, 0),
            "maximum": np.where(valid, self.max, 0),
            "mask": valid,
        }


@databrief
class ArrayStatistics:
    mean: jax.Array
    std: jax.Array
    maximum: jax.Array
    minimum: jax.Array
    mask: jax.Array | None = None
    p99: jax.Array | None = None
    p01: jax.Array | None = None

    def to_json(self) -> dict:
        return jax.tree.map(lambda x: x.tolist(), asdict(self))

    @classmethod
    def from_json(cls, data: Mapping[str, Sequence[float]]) -> ArrayStatistics:
        toarr = np.array  # partial(jnp.array, device=cpu)
        s = jax.tree.map(toarr, data, is_leaf=lambda x: not isinstance(x, Mapping))
        # s = jax.tree.map(jax.device_get, s)
        return cls(**s)

    @classmethod
    def compute(cls, arr: np.ndarray) -> ArrayStatistics:
        """Compute stats from first timestep only; result shape is (A,)."""
        if arr.ndim >= 3:
            arr = arr[:, 0]
        return cls(
            mean=arr.mean(axis=0),
            std=arr.std(axis=0),
            maximum=arr.max(axis=0),
            minimum=arr.min(axis=0),
            mask=np.ones(arr.shape[1:], dtype=bool),
            p99=np.quantile(arr, 0.99, axis=0),
            p01=np.quantile(arr, 0.01, axis=0),
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        y = (x - self.mean) / np.maximum(self.std, EPS)
        return y if self.mask is None else np.where(self.mask, y, x)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        y = x * np.maximum(self.std, EPS) + self.mean
        return y if self.mask is None else np.where(self.mask, y, x)


@dataclass
class DatasetStatistics:
    action: dict[str, ArrayStatistics]
    proprio: dict[str, ArrayStatistics]
    num_transitions: int
    num_trajectories: int

    def to_json(self) -> dict:
        return {
            "action": jax.tree.map(lambda x: x.to_json(), self.action),
            "proprio": jax.tree.map(lambda x: x.to_json(), self.proprio),
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> DatasetStatistics:
        action = {
            key: ArrayStatistics.from_json(value)  # type: ignore[arg-type]
            for key, value in data["action"].items()  # type: ignore[union-attr]
        }
        proprio = {
            key: ArrayStatistics.from_json(value)  # type: ignore[arg-type]
            for key, value in data.get("proprio", {}).items()  # type: ignore[union-attr]
        }
        return cls(
            action=action,
            proprio=proprio,
            num_transitions=int(data.get("num_transitions", 0)),
            num_trajectories=int(data.get("num_trajectories", 0)),
        )


class NormalizationType(str):
    NORMAL = "normal"
    BOUNDS = "bounds"


_STATS_VERSION = "v3_first_timestep_action_mask"


def _cache_path(hash_dependencies: Iterable[str], save_dir: str | Path | None) -> Path:
    key = "".join([_STATS_VERSION, *hash_dependencies]).encode("utf-8")
    unique_hash = hashlib.sha256(key, usedforsecurity=False).hexdigest()
    if save_dir is not None:
        return Path(save_dir) / f"dataset_statistics_{unique_hash}.json"
    cache_dir = Path.home() / ".cache" / "arrayrecords"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"dataset_statistics_{unique_hash}.json"


def compute_dataset_statistics(
    ds: Data,
    *,
    proprio_keys: Sequence[str],
    hash_dependencies: Sequence[str],
    save_dir: str | Path | None = None,
    force_recompute: bool = False,
) -> DatasetStatistics:
    """Computes statistics for steps or loads cached values."""

    log.info(f"checking cache for stats {save_dir}")
    cache_path = _cache_path(hash_dependencies, save_dir)
    if cache_path.exists() and not force_recompute:
        log.info("they were in the cache")
        with cache_path.open("r") as f:
            return DatasetStatistics.from_json(json.load(f))
    log.info("no stats found in cache")

    N = int(ds[-1]["info"]["id"]["episode"][0] + 1)
    assert N, "No steps found in dataset."
    t = len(ds)

    _bs = 1  # 1024
    # Build OnlineStats only for action/proprio subtrees, using first-timestep shape (A,)
    sample = ds[0]

    def _make_stream(x):
        a = np.array(x)
        return OnlineStats(a[0].shape, dtype=a.dtype)

    streams = {
        "action": jax.tree.map(_make_stream, sample["action"]),
        "proprio": jax.tree.map(_make_stream, sample["observation"]["proprio"]),
    }

    def _update(x):
        for key, value in x["action"].items():
            action_mask = x.get("action_norm_mask", {}).get(key)
            streams["action"][key].update(value[0], mask=action_mask)
        for key, value in x["observation"]["proprio"].items():
            streams["proprio"][key].update(value[0])
        return x

    # ds = ds.slice(slice(5))
    mpds = (
        ds.to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=128))
        # .batch(_bs)
        # .mp_prefetch(grain.MultiprocessingOptions(num_workers=0))
        .map(_update)
    )

    def take_keys(x):
        return {k: v for k, v in x.items() if k == "action" or k == "observation"}

    mpds = mpds.map(take_keys)

    mpit = iter(mpds)
    trees = list(tqdm(mpit, desc="Loading ds for stats computation...", total=t // _bs))
    stats = jax.tree.map(lambda s: ArrayStatistics(**s.finalize()), streams)

    print(stats)

    # trees = [next(mpit) for _ in tqdm(range(2), desc="Computing ds stats...", total=t)]

    # for ep-wise
    # trees = jax.tree.map(lambda *a: np.concatenate([*a], axis=0), *trees)
    # for step-wise
    # trees = jax.tree.map(lambda *a: np.stack([*a], axis=0), *trees)

    # actions = trees["action"]
    # assert t, "No transitions found in dataset."
    # proprio = trees["observation"]["proprio"]

    print("computing stats for dataset with", N, "transitions and", t, "trajectories")
    stats = DatasetStatistics(
        action=stats["action"],
        proprio=stats["proprio"],
        num_transitions=N,
        num_trajectories=t,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(stats.to_json(), f)
        print(f"Saved dataset statistics to {cache_path}")
    return stats


def _resolve_action_mask(
    key: str,
    stats: ArrayStatistics,
    action_value: np.ndarray,
    action_mask: Sequence[bool] | Mapping[str, Sequence[bool]] | None,
) -> np.ndarray | None:
    mask = stats.mask
    override = action_mask.get(key) if isinstance(action_mask, Mapping) else action_mask
    if override is not None:
        override = np.asarray(override, dtype=bool)
        if override.shape[-1] != action_value.shape[-1]:
            raise ValueError(
                f"Length of action mask {override.shape} does not match action dimension {action_value.shape}."
            )
        mask = override if mask is None else np.logical_and(mask, override)
    return None if mask is None else np.asarray(mask, dtype=bool)


def _apply_action_mask(action: np.ndarray, normalized: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return normalized
    if mask.shape[-1] != normalized.shape[-1]:
        raise ValueError(f"Length of action mask {mask.shape} does not match action dimension {normalized.shape}.")
    return np.where(mask, normalized, action)


def normalize_action_and_proprio(
    step: dict,
    *,
    metadata: DatasetStatistics,
    normalization_type: str,
    proprio_keys: Sequence[str],
    action_mask: Sequence[bool] | Mapping[str, Sequence[bool]] | None = None,
    skip_norm_keys: Sequence[str] = (),
    device: jax.Device | None = cpu,
) -> dict:
    """Normalizes actions and proprioceptive observations."""

    action = step["action"]
    if normalization_type == NormalizationType.NORMAL:
        normalized = jax.tree.map(lambda a, ma: ma.normalize(a), action, metadata.action)
    elif normalization_type == NormalizationType.BOUNDS:
        normalized = jax.tree.map(
            lambda a, ma: 2.0 * (a - ma.minimum) / np.maximum(ma.maximum - ma.minimum, EPS) - 1.0,
            action,
            metadata.action,
        )
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")
    step["action"] = {
        k: _apply_action_mask(
            action[k],
            normalized[k],
            _resolve_action_mask(k, metadata.action[k], action[k], action_mask),
        )
        for k in action
    }

    obs = step.get("observation")
    for key in proprio_keys:
        if key not in obs or key in skip_norm_keys:
            continue
        if key not in metadata.proprio:
            continue
        if normalization_type == NormalizationType.NORMAL:
            obs[key] = metadata.proprio[key].normalize(obs[key])
        elif normalization_type == NormalizationType.BOUNDS:
            ma = metadata.proprio[key]
            obs[key] = 2.0 * (obs[key] - ma.minimum) / np.maximum(ma.maximum - ma.minimum, EPS) - 1.0
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
    step["observation"] = obs
    return step
