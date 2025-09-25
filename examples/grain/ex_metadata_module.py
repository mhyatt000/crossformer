"""Dataset statistics tour matching :mod:`tests.grain.test_metadata_module`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossformer.data.grain import metadata


def synthetic_trajectories() -> list[dict[str, object]]:
    """Generate small trajectories with action and proprio streams."""
    action = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    proprio = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]], dtype=np.float32)
    return [
        {"action": action, "observation": {"proprio_arm": proprio}},
        {"action": action + 1.0, "observation": {"proprio_arm": proprio + 1.0}},
    ]


def compute_stats_demo(tmp_dir: Path) -> metadata.DatasetStatistics:
    """Compute and cache dataset statistics in a temp directory."""
    stats = metadata.compute_dataset_statistics(
        synthetic_trajectories(),
        proprio_keys=["proprio_arm"],
        hash_dependencies=["example"],
        save_dir=tmp_dir,
        force_recompute=True,
    )
    return stats


def json_roundtrip_demo(stats: metadata.DatasetStatistics, tmp_dir: Path) -> metadata.DatasetStatistics:
    """Write statistics to JSON and read them back into a dataclass."""
    path = tmp_dir / "stats.json"
    path.write_text(json.dumps(stats.to_json()))
    return metadata.DatasetStatistics.from_json(json.loads(path.read_text()))


def normalize_demo(stats: metadata.DatasetStatistics) -> dict[str, np.ndarray]:
    """Normalize actions and proprio signals with masks and bounds."""
    trajectory = {
        "action": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "observation": {"proprio_arm": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
    }
    normalized = metadata.normalize_action_and_proprio(
        trajectory,
        metadata=stats,
        normalization_type=metadata.NormalizationType.NORMAL,
        proprio_keys=["proprio_arm"],
        action_mask=[True, False],
    )
    bounds = metadata.normalize_action_and_proprio(
        trajectory,
        metadata=stats,
        normalization_type=metadata.NormalizationType.BOUNDS,
        proprio_keys=["proprio_arm"],
        skip_norm_keys=["proprio_arm"],
    )
    return {"normal": normalized["action"], "bounds": bounds["action"]}


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    tmp = Path("/tmp/crossformer_metadata_example")
    tmp.mkdir(parents=True, exist_ok=True)
    stats = compute_stats_demo(tmp)
    roundtrip = json_roundtrip_demo(stats, tmp)
    print("num transitions", stats.num_transitions, roundtrip.num_transitions)
    normalized = normalize_demo(stats)
    print("normalized action", normalized["normal"].shape)
