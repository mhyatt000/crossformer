from __future__ import annotations

import json
from pathlib import Path

from conftest import make_synthetic_trajectory, standardize_synthetic
import numpy as np
import pytest

from crossformer.data.grain import builders
from crossformer.utils.spec import ModuleSpec


@pytest.fixture
def dataset_config(tmp_path: Path) -> builders.GrainDatasetConfig:
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
        statistics_save_dir=str(tmp_path / "stats"),
        action_normalization_mask=[True, False, True, True],
        skip_norm_keys=["proprio_embedding"],
        filter_fns=[lambda traj: traj["language_instruction"][0] != ""],
    )


def test_build_trajectory_dataset_applies_standardization(
    dataset_config: builders.GrainDatasetConfig,
):
    dataset, stats = builders.build_trajectory_dataset(dataset_config)
    assert len(dataset) == 2
    first = dataset[0]
    assert set(first["observation"].keys()) >= {
        "image_main",
        "image_aux",
        "image_wrist",
        "proprio_cartesian",
        "proprio_joints",
        "proprio_gripper",
        "proprio_embedding",
        "timestep",
    }
    assert first["task"]["language_instruction"].shape[0] == first["action"].shape[0]
    assert np.isclose(first["action"].mean(axis=0)[1], 0.0, atol=1.0)
    assert first["observation"]["proprio_embedding"].dtype == np.float32
    assert stats.num_trajectories == 2
    assert dataset.dataset_statistics.action.mean.shape == (4,)


def test_build_trajectory_dataset_with_existing_statistics(dataset_config: builders.GrainDatasetConfig, tmp_path: Path):
    dataset, stats = builders.build_trajectory_dataset(dataset_config)
    stats_path = tmp_path / "precomputed.json"
    stats_path.write_text(json.dumps(stats.to_json()))

    dataset_config.dataset_statistics = str(stats_path)
    dataset_config.force_recompute_dataset_statistics = False
    dataset2, stats2 = builders.build_trajectory_dataset(dataset_config)
    assert len(dataset2) == len(dataset)
    assert np.allclose(stats2.action.mean, stats.action.mean)


@pytest.mark.parametrize(
    "input_value, expected",
    [
        ([], []),
        ([0, 1, 2], [0, 1, 2]),
    ],
)
def test_iter_source_handles_sequences(input_value, expected):
    config = builders.GrainDatasetConfig(name="iter", source=input_value)
    source = builders._resolve_source(config.source)
    items = list(builders._iter_source(source))
    assert len(items) == len(expected)


def test_sample_match_key_finds_patterns():
    traj = {"task_language": "hello", "other": 1}
    assert builders._sample_match_key(traj, "task*") == "hello"
    with pytest.raises(ValueError):
        builders._sample_match_key(traj, "missing*")


def test_load_dataset_statistics_from_mapping(
    dataset_config: builders.GrainDatasetConfig,
):
    _, stats = builders.build_trajectory_dataset(dataset_config)
    mapping = stats.to_json()
    dataset_config.dataset_statistics = mapping
    dataset_config.force_recompute_dataset_statistics = False
    _, stats2 = builders.build_trajectory_dataset(dataset_config)
    assert np.allclose(stats2.action.mean, stats.action.mean)
