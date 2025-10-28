"""Dataset metadata utilities for the Grain pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import grain
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from crossformer.utils.jax_utils import cpu, with_device_context
from crossformer.utils.typing import Data

log = logging.getLogger(__name__)

EPS = 1e-8


@dataclass
class ArrayStatistics:
    mean: jax.Array
    std: jax.Array
    maximum: jax.Array
    minimum: jax.Array
    p99: jax.Array
    p01: jax.Array

    def to_json(self) -> dict:
        return jax.tree.map(lambda x: x.tolist(), asdict(self))

    @classmethod
    def from_json(cls, data: Mapping[str, Sequence[float]]) -> ArrayStatistics:
        arr = partial(jnp.array, device=cpu)
        s = jax.tree.map(arr, data, is_leaf=lambda x: not isinstance(x, Mapping))
        s = jax.tree.map(jax.device_get, s)
        return cls(**s)

    @classmethod
    def compute(cls, arr: np.ndarray) -> ArrayStatistics:
        return cls(
            mean=arr.mean(axis=0),
            std=arr.std(axis=0),
            maximum=arr.max(axis=0),
            minimum=arr.min(axis=0),
            p99=jnp.quantile(arr, 0.99, axis=0),
            p01=jnp.quantile(arr, 0.01, axis=0),
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
            "proprio": jax.tree.map(lambda x: x.to_json(), self.proprio),
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> DatasetStatistics:
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


def _cache_path(hash_dependencies: Iterable[str], save_dir: str | Path | None) -> Path:
    key = "".join(hash_dependencies).encode("utf-8")
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

    actions = []
    proprio: dict[str, list[np.ndarray]] = {key: [] for key in proprio_keys}
    num_transitions = 0

    # assumes you can have it all in RAM
    # but we can
    # items: list[Data]= list(ds)
    N = int(ds[-1]["info"]["id"]["episode_id"] + 1)
    assert N, "No steps found in dataset."
    t = len(ds)

    _bs = 1  # 1024
    mpds = (
        ds.to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=512))
        # .batch(_bs)
        .mp_prefetch(grain.MultiprocessingOptions(num_workers=32))
    )

    def take_keys(x):
        return {k: v for k, v in x.items() if k == "action" or k == "observation"}

    mpds = mpds.map(take_keys)

    mpit = iter(mpds)
    trees = list(tqdm(mpit, desc="Computing ds stats...", total=t // _bs))
    # trees = [next(mpit) for _ in tqdm(range(2), desc="Computing ds stats...", total=t)]

    trees = jax.tree.map(lambda *a: jnp.concatenate([*a], axis=0), *trees)
    # trees = jax.tree.map(lambda *a: jnp.stack([*a], axis=0), *trees)

    actions = trees["action"]
    assert t, "No transitions found in dataset."

    proprio = trees["observation"]["proprio"]

    stats = DatasetStatistics(
        action=ArrayStatistics.compute(actions),
        proprio=jax.tree.map(ArrayStatistics.compute, proprio),
        num_transitions=N,
        num_trajectories=t,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(stats.to_json(), f)
    return stats


def normalize_action_and_proprio(
    step: dict,
    *,
    metadata: DatasetStatistics,
    normalization_type: str,
    proprio_keys: Sequence[str],
    action_mask: Sequence[bool] | None = None,
    skip_norm_keys: Sequence[str] = (),
    device: jax.Device | None = cpu,
) -> dict:
    """Normalizes actions and proprioceptive observations."""

    @with_device_context(device)
    def maximum(x, y):
        return jnp.maximum(x, y)

    @with_device_context(device)
    def where(cond, x, y):
        return jnp.where(cond, x, y)

    action = step["action"]
    if normalization_type == NormalizationType.NORMAL:
        mean = metadata.action.mean
        std = maximum(metadata.action.std, EPS)
        normalized = (action - mean) / std
    elif normalization_type == NormalizationType.BOUNDS:
        span = maximum(metadata.action.maximum - metadata.action.minimum, EPS)
        normalized = 2.0 * (action - metadata.action.minimum) / span - 1.0
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    if action_mask is not None:
        action_mask_array = jnp.array(action_mask, dtype=bool, device=device)
        if action_mask_array.shape[-1] != normalized.shape[-1]:
            raise ValueError("Length of action mask does not match action dimension.")
        normalized = jax.device_put(jnp.where(action_mask_array, normalized, action), device)
    step["action"] = normalized

    obs = step.get("observation")
    for key in proprio_keys:
        if key not in obs or key in skip_norm_keys:
            continue
        if key not in metadata.proprio:
            continue
        value = obs[key]
        stats = metadata.proprio[key]
        if normalization_type == NormalizationType.NORMAL:
            mean = stats.mean
            std = maximum(stats.std, EPS)
            obs[key] = (value - mean) / std
        else:
            span = maximum(stats.maximum - stats.minimum, EPS)
            obs[key] = 2.0 * (value - stats.minimum) / span - 1.0
    step["observation"] = obs
    return step
