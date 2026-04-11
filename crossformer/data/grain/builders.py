"""Builders converting raw data sources into Grain datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import fnmatch
from functools import partial
import hashlib
import json
import logging
from typing import Any

import grain.python as gp

from crossformer.data.grain import metadata
from crossformer.data.grain.embody import build_action_norm_mask, note_bodypart
from crossformer.data.grain.restructure import _restructure_step_mano, _restructure_trajectory
from crossformer.embody import Dataset
from crossformer.utils.spec import ModuleSpec

log = logging.getLogger(__name__)


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
    raise TypeError("Data source must be a Sequence, RandomAccessDataSource, or callable returning one.")


def _sample_match_key(traj: Mapping[str, Any], template: str) -> Any:
    """Returns the first key in traj that matches the given template using fnmatch."""
    matches = [key for key in traj if fnmatch.fnmatch(key, template)]
    if not matches:
        raise ValueError(f"No keys match template {template!r}; available keys: {traj.keys()}")
    return traj[matches[0]]


@dataclass
class Keys:
    image: Mapping[str | Any] | Sequence[str] = field(default_factory=list)
    depth: Mapping[str | Any] | Sequence[str] = field(default_factory=list)
    proprio: Mapping[str | Any] = field(default_factory=dict)
    lang: str | None = None
    # lang: Mapping[str| Any] = field(default_factory=dict)


@dataclass
class GrainDatasetConfig:
    name: str
    source: Sequence[dict] | gp.RandomAccessDataSource | Callable[[], Any]
    standardize_fn: ModuleSpec | Callable | None = None

    keys: Keys = field(default_factory=Keys)
    action_proprio_normalization_type: str = metadata.NormalizationType.NORMAL
    dataset_statistics: metadata.DatasetStatistics | Mapping[str, Any] | str | None = None
    statistics_save_dir: str | None = None
    force_recompute_dataset_statistics: bool = False
    action_normalization_mask: Sequence[bool] | None = None
    filter_fns: Sequence[ModuleSpec | Callable[[dict], bool]] = ()
    skip_norm: bool = False
    skip_norm_keys: Sequence[str] = ()
    seed: int = 0

    batch_size: int = 256
    debug: bool = False


def stable_hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) / 2**256


def _load_dataset_statistics(
    config: GrainDatasetConfig,
    ds: Sequence[dict],
) -> metadata.DatasetStatistics:
    if isinstance(config.dataset_statistics, metadata.DatasetStatistics):
        return config.dataset_statistics
    if isinstance(config.dataset_statistics, Mapping):
        return metadata.DatasetStatistics.from_json(config.dataset_statistics)
    if isinstance(config.dataset_statistics, str):
        with open(config.dataset_statistics, "r") as f:
            return metadata.DatasetStatistics.from_json(json.load(f))

    log.info("Computing dataset statistics from scratch...")
    hash_dependencies = [
        config.name,  # str(asdict(config))
        "bodypart_norm_mask_v1",
        str(len(ds)),
        str(sorted(config.keys.image)),
        str(sorted(config.keys.depth)),
        str(sorted(config.keys.proprio.keys())),
    ]
    return metadata.compute_dataset_statistics(
        ds,
        proprio_keys=list(config.keys.proprio.keys()),
        hash_dependencies=hash_dependencies,
        save_dir=config.statistics_save_dir,
        force_recompute=config.force_recompute_dataset_statistics,
    )


def build_trajectory_dataset(
    config: GrainDatasetConfig,
) -> tuple[gp.MapDataset, metadata.DatasetStatistics]:
    """Builds a :class:`grain.MapDataset` emitting standardized trajectories."""

    ds = source = _resolve_source(config.source)
    embodiment = Dataset.REGISTRY[config.name].embodiment

    filters = [_resolve_callable(fn) for fn in config.filter_fns if fn]
    for f in filters:
        ds = ds.filter(f)

    standardize = _resolve_callable(config.standardize_fn)

    r = _restructure_step_mano if "k3ds" in ds[0]["action"] else _restructure_trajectory
    restructure = ModuleSpec.create(r, name=config.name, lang_key=config.keys.lang)
    ds = ds.map(ModuleSpec.instantiate(restructure))
    ds = ds.map(partial(note_bodypart, embodiment=embodiment))

    stats = _load_dataset_statistics(config, ds)
    norm_mask: dict = build_action_norm_mask(ds[0]["action"], embodiment)

    def norm_all(x: dict, stats: metadata.DatasetStatistics):
        _norm = partial(metadata.normalize_tree, mask=norm_mask) if norm_mask else metadata.normalize_tree
        x["action"] = _norm(x["action"], stats.action)
        x["observation"]["proprio"] = _norm(x["observation"]["proprio"], stats.proprio)
        return x

    ds = ds.map(partial(norm_all, stats=stats)) if not config.skip_norm else ds

    ds.dataset_statistics = stats  # type: ignore[attr-defined]
    return ds, stats
