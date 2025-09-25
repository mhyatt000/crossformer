"""Dataset builder how-to paired with :mod:`tests.grain.test_builders_module`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from crossformer.data.grain import builders
from crossformer.utils.spec import ModuleSpec


def make_synthetic_trajectory(length: int, *, language: str, seed: int = 0) -> dict:
    """Match the synthetic trajectories used inside the unit tests."""
    rng = np.random.default_rng(seed)
    observation = {
        "img1": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "img2": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "img_wrist": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "state": {
            "cartesian": rng.normal(size=(length, 6)).astype(np.float32),
            "joints": rng.normal(size=(length, 7)).astype(np.float32),
            "gripper": rng.normal(size=(length, 1)).astype(np.float32),
        },
        "language_embedding": rng.normal(size=(length, 8)).astype(np.float32),
    }
    return {
        "observation": observation,
        "action": rng.normal(size=(length, 4)).astype(np.float32),
        "language_instruction": np.array([language] * length, dtype=object),
    }


def standardize_synthetic(traj: dict) -> dict:
    """Flatten nested state dicts for downstream tokenizers."""
    obs = dict(traj["observation"])
    state = obs.pop("state")
    for key, value in state.items():
        obs[f"state_{key}"] = value
    traj = dict(traj)
    traj["observation"] = obs
    return traj


def dataset_config(tmp_dir: Path) -> builders.GrainDatasetConfig:
    """Create a config identical to the one exercised by the tests."""
    trajectories = [
        make_synthetic_trajectory(5, language="pick", seed=0),
        make_synthetic_trajectory(4, language="place", seed=1),
        make_synthetic_trajectory(3, language="", seed=2),
    ]
    return builders.GrainDatasetConfig(
        name="spec_dataset",
        source=lambda: trajectories,
        standardize_fn=ModuleSpec.create(standardize_synthetic),
        image_obs_keys={"main": "img1", "aux": "img2", "wrist": "img_wrist"},
        depth_obs_keys={},
        proprio_obs_keys={
            "cartesian": "state_cartesian",
            "joints": "state_joints",
            "gripper": "state_gripper",
            "embedding": "language_embedding",
        },
        proprio_obs_dims={"cartesian": 6, "joints": 7, "gripper": 1, "embedding": 8},
        language_key="language_instruction",
        statistics_save_dir=str(tmp_dir / "stats"),
        action_normalization_mask=[True, False, True, True],
        skip_norm_keys=["proprio_embedding"],
        filter_fns=[lambda traj: traj["language_instruction"][0] != ""],
    )


def build_dataset_demo(tmp_dir: Path) -> tuple[list, builders.DatasetStatistics]:
    """Build the dataset and return both data and statistics."""
    config = dataset_config(tmp_dir)
    dataset, stats = builders.build_trajectory_dataset(config)
    return dataset, stats


def reuse_statistics_demo(config: builders.GrainDatasetConfig, stats: builders.DatasetStatistics, tmp_dir: Path) -> builders.DatasetStatistics:
    """Persist statistics and reuse them without recomputation."""
    stats_path = tmp_dir / "precomputed.json"
    stats_path.write_text(json.dumps(stats.to_json()))
    config.dataset_statistics = str(stats_path)
    config.force_recompute_dataset_statistics = False
    _, stats2 = builders.build_trajectory_dataset(config)
    return stats2


def iter_source_demo(source: Iterable) -> list:
    """Resolve static sequences or callables into iterable datasets."""
    resolved = builders._resolve_source(source)
    return list(builders._iter_source(resolved))


def sample_match_demo(traj: dict) -> str:
    """Locate a key in a trajectory using wildcard patterns."""
    return builders._sample_match_key(traj, "task*")


def load_statistics_from_mapping_demo(config: builders.GrainDatasetConfig, stats: builders.DatasetStatistics) -> builders.DatasetStatistics:
    """Load dataset statistics from an in-memory mapping."""
    mapping = stats.to_json()
    config.dataset_statistics = mapping
    config.force_recompute_dataset_statistics = False
    _, stats2 = builders.build_trajectory_dataset(config)
    return stats2


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    tmp = Path("/tmp/crossformer_builders_example")
    tmp.mkdir(parents=True, exist_ok=True)
    dataset, stats = build_dataset_demo(tmp)
    print("filtered trajectories", len(dataset))
    stats_reused = reuse_statistics_demo(dataset_config(tmp), stats, tmp)
    print("stats reused", np.allclose(stats_reused.action.mean, stats.action.mean))
    print("iter source", iter_source_demo([0, 1, 2]))
    print("sample match", sample_match_demo({"task_language": "hello", "other": 1}))
    mapped = load_statistics_from_mapping_demo(dataset_config(tmp), stats)
    print("mapped stats", mapped.action.mean.shape)
