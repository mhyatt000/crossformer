from __future__ import annotations

import numpy as np
import pytest

from crossformer.data.grain import transforms


@pytest.fixture
def base_trajectory() -> dict:
    length = 5
    window = 3
    rng = np.random.default_rng(0)
    observation = {
        "rgb": rng.integers(0, 255, size=(length, 4, 4, 3), dtype=np.uint8),
        "proprio": rng.normal(size=(length, 3)).astype(np.float32),
    }
    trajectory = {
        "observation": observation,
        "task": {"language_instruction": np.array(["pick"] * length, dtype=object)},
        "action": rng.normal(size=(length, 2)).astype(np.float32),
        "dataset_name": np.array(["demo"] * length, dtype=object),
    }
    trajectory = transforms.add_pad_mask_dict(trajectory)
    trajectory = transforms.pad_actions_and_proprio(
        trajectory, max_action_dim=3, max_proprio_dim=4
    )
    trajectory = transforms.chunk_action_and_observation(
        trajectory, window_size=window, action_horizon=2
    )
    return trajectory


def test_add_pad_mask_dict_creates_boolean_masks():
    traj = {"observation": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}}
    result = transforms.add_pad_mask_dict(traj)
    mask = result["observation"]["pad_mask_dict"]["image"]
    assert mask.dtype == bool
    assert np.all(mask)


def test_add_head_action_mask_with_mapping(base_trajectory):
    mapping = {"arm": ["demo"], "unused": ["other"]}
    result = transforms.add_head_action_mask(base_trajectory, mapping)
    assert set(result["action_head_masks"].keys()) == {"arm", "unused"}
    assert np.all(result["action_head_masks"]["arm"])
    assert not np.any(result["action_head_masks"]["unused"])


def test_pad_actions_and_proprio_padding_behavior():
    traj = {
        "action": np.ones((2, 2), dtype=np.float32),
        "observation": {"proprio": np.ones((2, 2), dtype=np.float32)},
    }
    padded = transforms.pad_actions_and_proprio(
        traj, max_action_dim=4, max_proprio_dim=3
    )
    assert padded["action"].shape == (2, 4)
    assert padded["observation"]["proprio"].shape == (2, 3)
    assert np.all(padded["action_pad_mask"][:, :2])
    assert not np.any(padded["action_pad_mask"][:, 2:])


def test_chunk_action_and_observation_shapes(base_trajectory):
    obs = base_trajectory["observation"]
    assert obs["rgb"].shape == (5, 3, 4, 4, 3)
    assert obs["timestep_pad_mask"].shape == (5, 3)
    assert base_trajectory["action"].shape == (5, 3, 2, 3)
    assert base_trajectory["action_pad_mask"].shape == (5, 3, 2, 3)
    assert base_trajectory["observation"]["task_completed"].shape == (5, 3, 2)


def test_chunk_action_and_observation_override_window():
    traj = {
        "observation": {"value": np.arange(4, dtype=np.float32)},
        "action": np.ones((4, 1), dtype=np.float32),
    }
    chunked = transforms.chunk_action_and_observation(
        transforms.add_pad_mask_dict(traj),
        window_size=3,
        action_horizon=1,
        override_window_size=2,
    )
    mask = chunked["observation"]["timestep_pad_mask"]
    assert np.all(~mask[:, 0])
    assert np.all(mask[:, -1])


def test_subsample_random_selection():
    traj = {
        "action": np.arange(10)[:, None],
        "observation": {"value": np.arange(10)[:, None]},
    }
    rng = np.random.default_rng(0)
    subsampled = transforms.subsample(traj, length=5, rng=rng)
    assert subsampled["action"].shape[0] == 5
    assert np.unique(subsampled["action"]).shape[0] == 5


def test_zero_out_future_proprio_raises_on_invalid_shape():
    traj = {"observation": {"proprio": np.ones((3,), dtype=np.float32)}}
    with pytest.raises(ValueError):
        transforms.zero_out_future_proprio(traj)


def test_zero_out_future_proprio_on_chunked(base_trajectory):
    updated = transforms.zero_out_future_proprio(base_trajectory)
    proprio = updated["observation"].get("proprio")
    if proprio is not None:
        assert np.all(proprio[:, 1:] == 0)


def test_flatten_trajectory_emits_frames(base_trajectory):
    frames = list(transforms.flatten_trajectory(base_trajectory))
    assert len(frames) == 5
    first = frames[0]
    assert first["action"].shape == (3, 2, 3)
    assert first["observation"]["rgb"].shape == (3, 4, 4, 3)
    assert first["dataset_name"] == "demo"


def test_drop_empty_language_raises():
    traj = {
        "action": np.zeros((2, 1)),
        "task": {"language_instruction": np.array(["", ""], dtype=object)},
    }
    with pytest.raises(ValueError):
        transforms.drop_empty_language(traj)


def test_uniform_goal_relabel_adds_goal():
    traj = {
        "action": np.zeros((3, 1), dtype=np.float32),
        "observation": {"image": np.arange(9, dtype=np.float32).reshape(3, 3)},
    }
    rng = np.random.default_rng(0)
    relabeled = transforms.uniform_goal_relabel(traj, rng=rng)
    assert "task" in relabeled
    assert "image" in relabeled["task"]


def test_maybe_cast_dtype_changes_type():
    value = np.array([1, 2, 3], dtype=np.int32)
    cast = transforms.maybe_cast_dtype(value, np.float32)
    assert cast.dtype == np.float32
    # no-op when dtype already matches
    same = transforms.maybe_cast_dtype(cast, np.float32)
    assert same.dtype == np.float32
