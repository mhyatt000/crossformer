"""Dataset pipeline primer following :mod:`tests.grain.test_pipelines`."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from crossformer.data.grain import builders
from crossformer.data.grain import pipelines
from crossformer.data.grain import transforms
from crossformer.utils.spec import ModuleSpec


def make_synthetic_trajectory(length: int, *, language: str, seed: int = 0) -> dict:
    """Create synthetic trajectories with multiple observation modalities."""
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
    """Flatten nested state dicts to match builder expectations."""
    obs = dict(traj["observation"])
    state = obs.pop("state")
    for key, value in state.items():
        obs[f"state_{key}"] = value
    traj = dict(traj)
    traj["observation"] = obs
    return traj


def pipeline_config(tmp_dir: Path) -> builders.GrainDatasetConfig:
    """Assemble a :class:`GrainDatasetConfig` mirroring the test fixture."""
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
        },
        proprio_obs_dims={"cartesian": 6, "joints": 7, "gripper": 1},
        language_key="language_instruction",
        statistics_save_dir=str(tmp_dir / "stats"),
    )


def single_dataset_demo(tmp_dir: Path) -> pipelines.GrainDataset:
    """Build a single dataset pipeline with post-chunk transforms."""
    config = pipeline_config(tmp_dir)
    return pipelines.make_single_dataset(
        config,
        train=False,
        traj_transform_kwargs={
            "window_size": 3,
            "action_horizon": 2,
            "skip_unlabeled": True,
            "goal_relabeling_strategy": "uniform",
            "max_action": 10.0,
            "max_proprio": 10.0,
            "post_chunk_transforms": [ModuleSpec.create(transforms.zero_out_future_proprio)],
            "head_to_dataset": {"arm": [config.name]},
        },
        shuffle_buffer_size=3,
        seed=7,
    )


def frame_transform_demo(dataset: Iterable[dict]) -> list[dict]:
    """Annotate frames using :func:`pipelines.apply_frame_transforms`."""
    def annotate(frame: dict) -> dict:
        frame = dict(frame)
        frame["annotated"] = True
        return frame

    transformed = pipelines.apply_frame_transforms(
        dataset,
        [ModuleSpec.create(annotate)],
    )
    return list(transformed)


def interleaved_dataset_demo(tmp_dir: Path) -> pipelines.InterleavedDataset:
    """Mix two dataset configs with sampling weights."""
    cfg_a = builders.GrainDatasetConfig(
        name="dataset_a",
        source=[make_synthetic_trajectory(4, language="alpha", seed=3)],
        standardize_fn=ModuleSpec.create(standardize_synthetic),
        image_obs_keys={"main": "img1"},
        depth_obs_keys={},
        proprio_obs_keys={"cartesian": "state_cartesian"},
        proprio_obs_dims={"cartesian": 6},
        language_key="language_instruction",
        statistics_save_dir=str(tmp_dir / "stats_a"),
    )
    cfg_b = builders.GrainDatasetConfig(
        name="dataset_b",
        source=[make_synthetic_trajectory(3, language="beta", seed=4)],
        standardize_fn=ModuleSpec.create(standardize_synthetic),
        image_obs_keys={"main": "img1"},
        depth_obs_keys={},
        proprio_obs_keys={"cartesian": "state_cartesian"},
        proprio_obs_dims={"cartesian": 6},
        language_key="language_instruction",
        statistics_save_dir=str(tmp_dir / "stats_b"),
    )
    return pipelines.make_interleaved_dataset(
        [cfg_a, cfg_b],
        train=False,
        sample_weights=[0.75, 0.25],
        shuffle_buffer_size=4,
        seed=9,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    tmp = Path("/tmp/crossformer_pipelines_example")
    tmp.mkdir(parents=True, exist_ok=True)
    single = single_dataset_demo(tmp)
    frames = list(single.dataset)
    print("single dataset frames", len(frames))
    annotated = frame_transform_demo(frames)
    print("annotated flag", annotated[0]["annotated"])
    interleaved = interleaved_dataset_demo(tmp)
    print("interleaved keys", list(interleaved.statistics.keys()))
