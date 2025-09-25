"""Builders converting raw data sources into Grain datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import fnmatch
from functools import partial
import json
import logging
from typing import Any

import grain.python as gp
import numpy as np
from tqdm import tqdm

from crossformer.data.grain import metadata, utils
from crossformer.utils.spec import ModuleSpec

logger = logging.getLogger(__name__)


def _resolve_callable(spec_or_fn: ModuleSpec | Callable | None) -> Callable | None:
    if spec_or_fn is None:
        return None
    if isinstance(spec_or_fn, Mapping) and set(spec_or_fn.keys()) == {
        "module",
        "name",
        "args",
        "kwargs",
    }:
        return ModuleSpec.instantiate(spec_or_fn)  # type: ignore[arg-type]
    if not callable(spec_or_fn):
        raise TypeError(f"Expected callable or ModuleSpec, got {type(spec_or_fn)!r}")
    return spec_or_fn  # type: ignore[return-value]


def _resolve_source(
    source: Sequence[dict] | gp.RandomAccessDataSource | Callable[[], Any],
) -> Sequence[dict] | gp.RandomAccessDataSource:
    if callable(source):
        return _resolve_source(source())
    if isinstance(source, gp.RandomAccessDataSource):
        return source
    if isinstance(source, Sequence):
        return source
    raise TypeError(
        "Data source must be a Sequence, RandomAccessDataSource, or callable returning one."
    )


def _iter_source(source: Sequence[dict] | gp.RandomAccessDataSource) -> Iterable[dict]:
    for index in range(len(source)):  # type: ignore[arg-type]
        element = source[index]
        if hasattr(element, "data"):
            element = element.data
        yield utils.clone_structure(element)


def _sample_match_key(traj: Mapping[str, Any], template: str) -> Any:
    matches = [key for key in traj if fnmatch.fnmatch(key, template)]
    if not matches:
        raise ValueError(
            f"No keys match template {template!r}; available keys: {traj.keys()}"
        )
    return traj[matches[0]]


@dataclass
class GrainDatasetConfig:
    name: str
    source: Sequence[dict] | gp.RandomAccessDataSource | Callable[[], Any]
    standardize_fn: ModuleSpec | Callable | None = None
    image_obs_keys: Mapping[str, str | None] = field(default_factory=dict)
    depth_obs_keys: Mapping[str, str | None] = field(default_factory=dict)
    proprio_obs_keys: Mapping[str, str | None] | None = None
    proprio_obs_dims: Mapping[str, int] | None = None
    language_key: str | None = None
    action_proprio_normalization_type: str = metadata.NormalizationType.NORMAL
    dataset_statistics: metadata.DatasetStatistics | Mapping[str, Any] | str | None = (
        None
    )
    statistics_save_dir: str | None = None
    force_recompute_dataset_statistics: bool = False
    action_normalization_mask: Sequence[bool] | None = None
    filter_fns: Sequence[ModuleSpec | Callable[[dict], bool]] = ()
    skip_norm: bool = False
    skip_norm_keys: Sequence[str] = ()
    seed: int = 0

    debug: bool = False


def _restructure_trajectory(
    traj: dict,
    *,
    name: str,
    standardize: Callable | None,
    config: GrainDatasetConfig,
) -> dict:
    traj = utils.clone_structure(traj)
    if standardize is not None:
        traj = standardize(traj)
    if "observation" not in traj or "action" not in traj:
        raise ValueError("Trajectory must contain 'observation' and 'action' keys.")

    # from rich.pretty import pprint
    # from crossformer.utils.spec import spec
    # pprint(spec(traj))

    # action = np.asarray(traj["action"], dtype=np.float32)

    # @codex add some action processing fn here.
    # in place of hard coded action definition

    # concat traj proprio joints and proprio gripper
    proprio = traj["observation"]["proprio"]
    action = np.concatenate([proprio["joints"], proprio["gripper"]])
    traj_len = action.shape[0]
    if action.ndim == 1 or traj_len == 0:
        return {}

    old_obs = traj["observation"]
    new_obs: dict[str, Any] = {}
    for new, old in config.image_obs_keys.items():
        key = f"image_{new}"
        if old is None:
            new_obs[key] = np.full((traj_len,), "", dtype=object)
        else:
            new_obs[key] = np.asarray(old_obs[old])
    for new, old in config.depth_obs_keys.items():
        key = f"depth_{new}"
        if old is None:
            new_obs[key] = np.full((traj_len,), "", dtype=object)
        else:
            new_obs[key] = np.asarray(old_obs[old])
    if config.proprio_obs_keys is not None:
        if config.proprio_obs_dims is None:
            raise ValueError(
                "proprio_obs_dims must be provided when proprio_obs_keys is set."
            )
        for new, old in config.proprio_obs_keys.items():
            key = f"proprio_{new}"
            if old is None:
                new_obs[key] = np.zeros(
                    (traj_len, config.proprio_obs_dims[new]), dtype=np.float32
                )
            else:
                new_obs[key] = np.asarray(old_obs[old], dtype=np.float32)

    new_obs["timestep"] = np.arange(traj_len, dtype=np.int32)

    task = {}
    if config.language_key is not None:
        language = _sample_match_key(traj, config.language_key)
        language = np.asarray(language)
        if language.shape == ():
            language = np.repeat(language, traj_len)
        if language.shape[0] != traj_len:
            language = np.broadcast_to(language, (traj_len,))
        task["language_instruction"] = language.astype(object)

    return {
        "observation": new_obs,
        "task": task,
        "action": action,
        "dataset_name": np.repeat(name, traj_len),
    }


def _load_dataset_statistics(
    config: GrainDatasetConfig,
    proprio_keys: Sequence[str],
    trajectories: Sequence[dict],
) -> metadata.DatasetStatistics:
    if isinstance(config.dataset_statistics, metadata.DatasetStatistics):
        return config.dataset_statistics
    if isinstance(config.dataset_statistics, Mapping):
        return metadata.DatasetStatistics.from_json(config.dataset_statistics)
    if isinstance(config.dataset_statistics, str):
        with open(config.dataset_statistics, "r") as f:
            return metadata.DatasetStatistics.from_json(json.load(f))

    hash_dependencies = [
        config.name,
        str(sorted(config.image_obs_keys.items())),
        str(sorted(config.depth_obs_keys.items())),
        str(sorted(proprio_keys)),
    ]
    return metadata.compute_dataset_statistics(
        trajectories,
        proprio_keys=proprio_keys,
        hash_dependencies=hash_dependencies,
        save_dir=config.statistics_save_dir,
        force_recompute=config.force_recompute_dataset_statistics,
    )


def build_trajectory_dataset(
    config: GrainDatasetConfig,
) -> tuple[gp.MapDataset, metadata.DatasetStatistics]:
    """Builds a :class:`grain.MapDataset` emitting standardized trajectories."""

    source = _resolve_source(config.source)
    standardize = _resolve_callable(config.standardize_fn)
    filter_functions = [
        fn for fn in (_resolve_callable(fn) for fn in config.filter_fns) if fn
    ]

    processed: list[dict] = []
    bar = partial(tqdm, desc="Processing trajectories", total=len(source), unit="ep")
    for raw_traj in bar(_iter_source(source)):
        if filter_functions and any(not fn(raw_traj) for fn in filter_functions):
            continue
        traj = _restructure_trajectory(
            raw_traj,
            name=config.name,
            standardize=standardize,
            config=config,
        )
        if not traj:
            continue
        processed.append(traj)

        if config.debug and len(processed) % 1000 == 0:
            break

    proprio_keys = [
        f"proprio_{key}"
        for key, value in (config.proprio_obs_keys or {}).items()
        if value is not None
    ]
    stats = _load_dataset_statistics(config, proprio_keys, processed)

    if not config.skip_norm:
        mask = config.action_normalization_mask
        normalized = [
            metadata.normalize_action_and_proprio(
                traj,
                metadata=stats,
                normalization_type=config.action_proprio_normalization_type,
                proprio_keys=proprio_keys,
                action_mask=mask,
                skip_norm_keys=config.skip_norm_keys,
            )
            for traj in processed
        ]
    else:
        normalized = processed

    dataset = gp.MapDataset.range(len(normalized)).map(
        lambda idx: utils.clone_structure(normalized[idx])
    )
    dataset.dataset_statistics = stats  # type: ignore[attr-defined]
    return dataset, stats
