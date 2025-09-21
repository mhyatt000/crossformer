from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crossformer.data.grain import metadata


@pytest.fixture
def simple_stats(tmp_path: Path) -> metadata.DatasetStatistics:
    action = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    proprio = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]], dtype=np.float32)
    trajectories = [
        {"action": action, "observation": {"proprio_arm": proprio}},
        {"action": action + 1.0, "observation": {"proprio_arm": proprio + 1.0}},
    ]
    stats = metadata.compute_dataset_statistics(
        trajectories,
        proprio_keys=["proprio_arm"],
        hash_dependencies=["unit-test"],
        save_dir=tmp_path,
        force_recompute=True,
    )
    return stats


def test_array_statistics_roundtrip(simple_stats: metadata.DatasetStatistics, tmp_path: Path):
    json_path = tmp_path / "stats.json"
    json_path.write_text(json.dumps(simple_stats.to_json()))
    loaded = metadata.DatasetStatistics.from_json(json.loads(json_path.read_text()))
    assert np.allclose(loaded.action.mean, simple_stats.action.mean)
    assert loaded.num_transitions == simple_stats.num_transitions
    assert "proprio_arm" in loaded.proprio


def test_compute_dataset_statistics_caches_results(simple_stats: metadata.DatasetStatistics, tmp_path: Path):
    cache_files = list(tmp_path.glob("dataset_statistics_*.json"))
    assert len(cache_files) == 1
    cached = metadata.compute_dataset_statistics(
        [],
        proprio_keys=["proprio_arm"],
        hash_dependencies=["unit-test"],
        save_dir=tmp_path,
        force_recompute=False,
    )
    assert np.allclose(cached.action.mean, simple_stats.action.mean)
    assert cached.num_trajectories == simple_stats.num_trajectories


def test_normalize_action_and_proprio_with_mask(simple_stats: metadata.DatasetStatistics):
    trajectory = {
        "action": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "observation": {"proprio_arm": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
    }
    normalized = metadata.normalize_action_and_proprio(
        trajectory,
        metadata=simple_stats,
        normalization_type=metadata.NormalizationType.NORMAL,
        proprio_keys=["proprio_arm"],
        action_mask=[True, False],
    )
    assert normalized["action"].shape == (2, 2)
    # second dimension should remain equal to original because mask disables normalization
    assert np.allclose(normalized["action"][:, 1], trajectory["action"][:, 1])
    assert not np.allclose(normalized["action"][:, 0], trajectory["action"][:, 0])
    assert np.allclose(normalized["observation"]["proprio_arm"].mean(), 0.0, atol=1e-5)


def test_normalize_action_and_proprio_bounds_mode(simple_stats: metadata.DatasetStatistics):
    trajectory = {
        "action": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "observation": {"proprio_arm": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
    }
    normalized = metadata.normalize_action_and_proprio(
        trajectory,
        metadata=simple_stats,
        normalization_type=metadata.NormalizationType.BOUNDS,
        proprio_keys=["proprio_arm"],
        skip_norm_keys=["proprio_arm"],
    )
    assert normalized["observation"]["proprio_arm"].shape == (2, 2)
    assert np.all((-1.001 <= normalized["action"]) & (normalized["action"] <= 1.001))


def test_normalize_action_and_proprio_invalid_type(simple_stats: metadata.DatasetStatistics):
    with pytest.raises(ValueError):
        metadata.normalize_action_and_proprio(
            {"action": np.zeros((2, 2), dtype=np.float32)},
            metadata=simple_stats,
            normalization_type="unknown",
            proprio_keys=[],
        )


def test_normalize_action_and_proprio_mask_length_mismatch(simple_stats: metadata.DatasetStatistics):
    with pytest.raises(ValueError):
        metadata.normalize_action_and_proprio(
            {"action": np.zeros((2, 3), dtype=np.float32)},
            metadata=simple_stats,
            normalization_type=metadata.NormalizationType.NORMAL,
            proprio_keys=[],
            action_mask=[True, False],
        )
