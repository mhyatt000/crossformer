from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from crossformer.data.grain import builders, pipelines, transforms
from crossformer.data.grain.tests.conftest import make_synthetic_trajectory, standardize_synthetic
from crossformer.utils.spec import ModuleSpec


def _annotate_frame(frame: dict) -> dict:
    frame = dict(frame)
    frame["annotated"] = True
    return frame


@pytest.fixture
def pipeline_config(tmp_path: Path) -> builders.GrainDatasetConfig:
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
        statistics_save_dir=str(tmp_path / "stats"),
    )


def test_make_single_dataset_pipeline(pipeline_config: builders.GrainDatasetConfig):
    result = pipelines.make_single_dataset(
        pipeline_config,
        train=False,
        traj_transform_kwargs={
            "window_size": 3,
            "action_horizon": 2,
            "skip_unlabeled": True,
            "goal_relabeling_strategy": "uniform",
            "max_action": 10.0,
            "max_proprio": 10.0,
            "post_chunk_transforms": [ModuleSpec.create(transforms.zero_out_future_proprio)],
            "head_to_dataset": {"arm": [pipeline_config.name]},
        },
        shuffle_buffer_size=3,
        seed=7,
    )

    frames = list(result.dataset)
    assert all(frame["dataset_name"] == pipeline_config.name for frame in frames)
    assert all(frame["task"]["language_instruction"] != "" for frame in frames)
    assert frames[0]["action"].shape[-2:] == (2, 4)
    assert bool(frames[0]["action_head_masks"]["arm"])
    # statistics still reflect all processed trajectories
    assert result.statistics.num_trajectories == 3
    assert result.statistics.action.mean.shape == (4,)


def test_apply_frame_transforms_module_spec():
    import grain.python as gp

    base = gp.MapDataset.range(3).map(lambda idx: {"value": idx})
    iter_dataset = base.to_iter_dataset()
    transformed = pipelines.apply_frame_transforms(
        iter_dataset, [ModuleSpec.create(_annotate_frame)]
    )
    frames = list(transformed)
    assert all(frame["annotated"] for frame in frames)


def test_make_interleaved_dataset_mixtures(tmp_path: Path):
    cfg_a = builders.GrainDatasetConfig(
        name="dataset_a",
        source=[make_synthetic_trajectory(4, language="alpha", seed=3)],
        standardize_fn=ModuleSpec.create(standardize_synthetic),
        image_obs_keys={"main": "img1"},
        depth_obs_keys={},
        proprio_obs_keys={"cartesian": "state_cartesian"},
        proprio_obs_dims={"cartesian": 6},
        language_key="language_instruction",
        statistics_save_dir=str(tmp_path / "stats_a"),
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
        statistics_save_dir=str(tmp_path / "stats_b"),
    )

    result = pipelines.make_interleaved_dataset(
        [cfg_a, cfg_b],
        train=False,
        sample_weights=[0.75, 0.25],
        shuffle_buffer_size=4,
        seed=9,
    )

    frames = list(itertools.islice(result.dataset, 6))
    dataset_names = {frame["dataset_name"] for frame in frames}
    assert dataset_names <= {"dataset_a", "dataset_b"}
    assert set(result.statistics.keys()) == {"dataset_a", "dataset_b"}


def test_make_interleaved_dataset_invalid_weights(tmp_path: Path):
    cfg = builders.GrainDatasetConfig(
        name="dataset_a",
        source=[make_synthetic_trajectory(2, language="alpha", seed=0)],
        standardize_fn=ModuleSpec.create(standardize_synthetic),
        image_obs_keys={"main": "img1"},
        depth_obs_keys={},
        proprio_obs_keys={"cartesian": "state_cartesian"},
        proprio_obs_dims={"cartesian": 6},
        language_key="language_instruction",
        statistics_save_dir=str(tmp_path / "stats_a"),
    )

    with pytest.raises(ValueError):
        pipelines.make_interleaved_dataset([cfg], train=False, sample_weights=[0.5, 0.5])
