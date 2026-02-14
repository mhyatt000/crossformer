from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest
import tensorflow as tf

from crossformer.data import traj_transforms


@settings(max_examples=50, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    window_size=st.integers(min_value=1, max_value=4),
    action_horizon=st.integers(min_value=1, max_value=4),
    obs_dim=st.integers(min_value=1, max_value=4),
    action_dim=st.integers(min_value=1, max_value=4),
    use_goal=st.booleans(),
    data=st.data(),
)
def test_chunk_act_obs_history_alignment(traj_len, window_size, action_horizon, obs_dim, action_dim, use_goal, data):
    """Chunking should assemble histories and masks consistently for raw actions."""
    observation = {"state": tf.random.uniform((traj_len, obs_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)}
    action = tf.random.uniform((traj_len, action_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    traj = {
        "observation": observation,
        "action": action,
        "task": {},
        "action_pad_mask": tf.ones((traj_len, action_dim), dtype=tf.bool),
    }
    if use_goal:
        goal_value = data.draw(st.integers(min_value=0, max_value=traj_len - 1))
        traj["task"]["timestep"] = tf.constant([goal_value] * traj_len, dtype=tf.int32)

    result = traj_transforms.chunk_act_obs(
        traj,
        window_size=window_size,
        action_horizon=action_horizon,
    )

    state_np = observation["state"].numpy()
    action_np = action.numpy()
    history_indices = np.arange(traj_len)[:, None] + np.arange(-window_size + 1, 1)
    expected_timestep_mask = history_indices >= 0
    history_indices_clipped = np.maximum(history_indices, 0)

    expected_state = np.take(state_np, history_indices_clipped, axis=0)
    np.testing.assert_allclose(result["observation"]["state"].numpy(), expected_state)
    np.testing.assert_array_equal(result["observation"]["timestep_pad_mask"].numpy(), expected_timestep_mask)

    chunk_indices = np.arange(traj_len)[:, None] + np.arange(action_horizon)
    chunk_indices = np.minimum(chunk_indices, traj_len - 1)
    action_chunk = np.take(action_np, chunk_indices, axis=0)
    expected_action = np.take(action_chunk, history_indices_clipped, axis=0)
    np.testing.assert_allclose(result["action"].numpy(), expected_action)

    if use_goal:
        goal_vec = np.full(traj_len, goal_value, dtype=np.int32)
    else:
        goal_vec = np.full(traj_len, traj_len - 1, dtype=np.int32)
    t = np.arange(traj_len)[:, None, None]
    w = np.arange(window_size)[None, :, None]
    h = np.arange(action_horizon)[None, None, :]
    relative_goal = goal_vec[:, None, None] - (t - (window_size + 1) + w + h)
    expected_task_completed = relative_goal <= 0
    np.testing.assert_array_equal(result["observation"]["task_completed"].numpy(), expected_task_completed)

    expected_action_mask = np.broadcast_to(
        np.logical_not(expected_task_completed)[..., None],
        (*expected_task_completed.shape, action_dim),
    )
    np.testing.assert_array_equal(result["action_pad_mask"].numpy(), expected_action_mask)


@settings(max_examples=25, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    window_size=st.integers(min_value=1, max_value=4),
    action_horizon=st.integers(min_value=1, max_value=4),
    obs_dim=st.integers(min_value=1, max_value=4),
    action_dim=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_chunk_act_obs_prechunked_actions(traj_len, window_size, action_horizon, obs_dim, action_dim, data):
    """When actions are pre-chunked the history axis should still be correct."""
    observation = {"state": tf.random.uniform((traj_len, obs_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)}
    action = tf.random.uniform(
        (traj_len, action_horizon, action_dim),
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    traj = {
        "observation": observation,
        "action": action,
        "task": {},
        "action_pad_mask": tf.ones((traj_len, action_horizon, action_dim), dtype=tf.bool),
    }
    include_goal = data.draw(st.booleans())
    if include_goal:
        goal_value = data.draw(st.integers(min_value=0, max_value=traj_len - 1))
        traj["task"]["timestep"] = tf.constant([goal_value] * traj_len, dtype=tf.int32)

    result = traj_transforms.chunk_act_obs(
        traj,
        window_size=window_size,
        action_horizon=action_horizon,
    )

    history_indices = np.arange(traj_len)[:, None] + np.arange(-window_size + 1, 1)
    expected_timestep_mask = history_indices >= 0
    history_indices_clipped = np.maximum(history_indices, 0)

    expected_state = np.take(observation["state"].numpy(), history_indices_clipped, axis=0)
    np.testing.assert_allclose(result["observation"]["state"].numpy(), expected_state)
    np.testing.assert_array_equal(result["observation"]["timestep_pad_mask"].numpy(), expected_timestep_mask)

    expected_action = np.take(action.numpy(), history_indices_clipped, axis=0)
    np.testing.assert_allclose(result["action"].numpy(), expected_action)

    if include_goal:
        goal_vec = np.full(traj_len, goal_value, dtype=np.int32)
    else:
        goal_vec = np.full(traj_len, traj_len - 1, dtype=np.int32)
    t = np.arange(traj_len)[:, None, None]
    w = np.arange(window_size)[None, :, None]
    h = np.arange(action_horizon)[None, None, :]
    relative_goal = goal_vec[:, None, None] - (t - (window_size + 1) + w + h)
    expected_task_completed = relative_goal <= 0
    np.testing.assert_array_equal(result["observation"]["task_completed"].numpy(), expected_task_completed)

    expected_action_mask = np.broadcast_to(
        np.logical_not(expected_task_completed)[..., None],
        (*expected_task_completed.shape, action_dim),
    )
    np.testing.assert_array_equal(result["action_pad_mask"].numpy(), expected_action_mask)


@settings(max_examples=30, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    window_size=st.integers(min_value=1, max_value=5),
    action_horizon=st.integers(min_value=1, max_value=3),
    obs_dim=st.integers(min_value=1, max_value=4),
    action_dim=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_chunk_act_obs_override_window_size(traj_len, window_size, action_horizon, obs_dim, action_dim, data):
    """Override window sizes should limit how many history steps remain valid."""
    observation = {"state": tf.random.uniform((traj_len, obs_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)}
    action = tf.random.uniform((traj_len, action_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    traj = {
        "observation": observation,
        "action": action,
        "task": {},
        "action_pad_mask": tf.ones((traj_len, action_dim), dtype=tf.bool),
    }

    override = data.draw(st.integers(min_value=1, max_value=window_size - 1)) if window_size > 1 else 1

    result = traj_transforms.chunk_act_obs(
        traj,
        window_size=window_size,
        action_horizon=action_horizon,
        override_window_size=override,
    )

    history_indices = np.arange(traj_len)[:, None] + np.arange(-window_size + 1, 1)
    base_mask = history_indices >= 0
    history_indices_clipped = np.maximum(history_indices, 0)

    expected_state = np.take(observation["state"].numpy(), history_indices_clipped, axis=0)
    np.testing.assert_allclose(result["observation"]["state"].numpy(), expected_state)

    keep = min(window_size, override)
    expected_mask = np.zeros_like(base_mask)
    if keep:
        expected_mask[:, -keep:] = base_mask[:, -keep:]
    np.testing.assert_array_equal(result["observation"]["timestep_pad_mask"].numpy(), expected_mask)


@settings(max_examples=25, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    prechunk=st.integers(min_value=1, max_value=4),
    action_dim=st.integers(min_value=1, max_value=4),
    deficit=st.integers(min_value=1, max_value=3),
)
def test_chunk_act_obs_raises_when_prechunk_too_small(traj_len, prechunk, action_dim, deficit):
    """Requests for longer horizons than available pre-chunks should fail."""
    action = tf.random.uniform((traj_len, prechunk, action_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    traj = {
        "observation": {"state": tf.zeros((traj_len, 1), dtype=tf.float32)},
        "action": action,
        "task": {},
        "action_pad_mask": tf.ones((traj_len, prechunk, action_dim), dtype=tf.bool),
    }
    with pytest.raises(ValueError):
        traj_transforms.chunk_act_obs(
            traj,
            window_size=2,
            action_horizon=prechunk + deficit,
        )


@settings(max_examples=50, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    action_dim=st.integers(min_value=1, max_value=4),
    action_padding=st.integers(min_value=0, max_value=3),
    include_proprio=st.booleans(),
    proprio_dim=st.integers(min_value=1, max_value=4),
    proprio_padding=st.integers(min_value=0, max_value=3),
)
def test_pad_actions_and_proprio_padding(
    traj_len, action_dim, action_padding, include_proprio, proprio_dim, proprio_padding
):
    """Padding should preserve existing values and mark new dims as masked."""
    action = tf.random.uniform((traj_len, action_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    observation = {}
    if include_proprio:
        observation["proprio"] = tf.random.uniform((traj_len, proprio_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    traj = {
        "observation": observation,
        "action": action,
    }

    max_action_dim = action_dim + action_padding
    max_proprio_dim = proprio_dim + proprio_padding if include_proprio else None

    action_before = action.numpy()
    proprio_before = observation["proprio"].numpy() if include_proprio else None

    result = traj_transforms.pad_actions_and_proprio(
        traj, max_action_dim=max_action_dim, max_proprio_dim=max_proprio_dim
    )

    padded_action = result["action"].numpy()
    assert padded_action.shape == (traj_len, max_action_dim)
    np.testing.assert_allclose(padded_action[:, :action_dim], action_before)
    if action_padding:
        np.testing.assert_array_equal(padded_action[:, action_dim:], np.zeros((traj_len, action_padding)))

    action_mask = result["action_pad_mask"].numpy()
    assert action_mask.shape == (traj_len, max_action_dim)
    np.testing.assert_array_equal(action_mask[:, :action_dim], np.ones((traj_len, action_dim), dtype=bool))
    if action_padding:
        np.testing.assert_array_equal(action_mask[:, action_dim:], np.zeros((traj_len, action_padding), dtype=bool))

    if include_proprio:
        proprio_after = result["observation"]["proprio"].numpy()
        assert proprio_after.shape == (traj_len, max_proprio_dim)
        np.testing.assert_allclose(proprio_after[:, :proprio_dim], proprio_before)
        if proprio_padding:
            np.testing.assert_array_equal(
                proprio_after[:, proprio_dim:],
                np.zeros((traj_len, proprio_padding)),
            )
    else:
        assert "proprio" not in result["observation"]


@settings(max_examples=25, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    action_dim=st.integers(min_value=2, max_value=5),
    shrink=st.integers(min_value=1, max_value=3),
)
def test_pad_actions_and_proprio_action_dim_validation(traj_len, action_dim, shrink):
    """Requesting padding smaller than the action dimension should error."""
    max_action_dim = action_dim - shrink
    if max_action_dim <= 0:
        pytest.skip("Shrink removed all action dimensions")
    action = tf.zeros((traj_len, action_dim), dtype=tf.float32)
    traj = {
        "observation": {},
        "action": action,
    }
    with pytest.raises(ValueError):
        traj_transforms.pad_actions_and_proprio(traj, max_action_dim=max_action_dim, max_proprio_dim=None)


@settings(max_examples=25, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    proprio_dim=st.integers(min_value=2, max_value=5),
    shrink=st.integers(min_value=1, max_value=3),
)
def test_pad_actions_and_proprio_proprio_dim_validation(traj_len, proprio_dim, shrink):
    """Proprio dimensions larger than the allowed maximum should error."""
    max_proprio_dim = proprio_dim - shrink
    if max_proprio_dim <= 0:
        pytest.skip("Shrink removed all proprio dimensions")
    traj = {
        "observation": {"proprio": tf.zeros((traj_len, proprio_dim), dtype=tf.float32)},
        "action": tf.zeros((traj_len, 1), dtype=tf.float32),
    }
    with pytest.raises(ValueError):
        traj_transforms.pad_actions_and_proprio(traj, max_action_dim=None, max_proprio_dim=max_proprio_dim)


def test_pad_actions_and_proprio_action_dim_value_error():
    """A wider action tensor than allowed should trigger validation."""
    traj = {
        "observation": {},
        "action": tf.zeros((2, 3), dtype=tf.float32),
    }
    with pytest.raises(ValueError):
        traj_transforms.pad_actions_and_proprio(traj, max_action_dim=2, max_proprio_dim=None)


def test_pad_actions_and_proprio_proprio_dim_value_error():
    """A wider proprio tensor than allowed should trigger validation."""
    traj = {
        "observation": {"proprio": tf.zeros((2, 4), dtype=tf.float32)},
        "action": tf.zeros((2, 1), dtype=tf.float32),
    }
    with pytest.raises(ValueError):
        traj_transforms.pad_actions_and_proprio(traj, max_action_dim=None, max_proprio_dim=3)


def test_add_head_action_mask_defaults_to_all_true():
    """When no mapping is provided every dataset should be included."""
    traj = {"dataset_name": tf.constant([b"set1", b"set2"], dtype=tf.string)}
    result = traj_transforms.add_head_action_mask(traj, head_to_dataset=None)
    mask = result["action_head_masks"]["action"].numpy()
    np.testing.assert_array_equal(mask, np.ones(2, dtype=bool))


def test_add_head_action_mask_respects_mapping():
    """Head-specific datasets should build matching boolean selectors."""
    dataset_names = tf.constant([b"set1", b"set2", b"set3"], dtype=tf.string)
    traj = {"dataset_name": dataset_names}
    head_to_dataset = {
        "policy": tf.constant([b"set1", b"set3"], dtype=tf.string),
        "value": tf.constant([b"set2"], dtype=tf.string),
    }

    result = traj_transforms.add_head_action_mask(traj, head_to_dataset)

    policy_mask = result["action_head_masks"]["policy"].numpy()
    np.testing.assert_array_equal(policy_mask, np.array([True, False, True]))

    value_mask = result["action_head_masks"]["value"].numpy()
    np.testing.assert_array_equal(value_mask, np.array([False, True, False]))


@settings(max_examples=50, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    subsample_length=st.integers(min_value=1, max_value=6),
    feature_dim=st.integers(min_value=1, max_value=3),
)
def test_subsample_preserves_structure(traj_len, subsample_length, feature_dim):
    """Subsampling keeps consistent lengths across the trajectory structure."""
    action = tf.range(traj_len, dtype=tf.int32)
    observation_state = tf.cast(
        tf.reshape(tf.range(traj_len * feature_dim), (traj_len, feature_dim)),
        dtype=tf.float32,
    )
    traj = {
        "action": action,
        "observation": {"state": observation_state},
        "task": {"label": tf.range(traj_len, dtype=tf.int32)},
    }

    result = traj_transforms.subsample(dict(traj), subsample_length=subsample_length)

    expected_len = min(traj_len, subsample_length)
    action_result = result["action"].numpy()
    assert action_result.shape[0] == expected_len

    if traj_len <= subsample_length:
        np.testing.assert_array_equal(action_result, action.numpy())
    else:
        assert action_result.shape[0] == subsample_length
        assert len(np.unique(action_result)) == subsample_length
        assert set(action_result.tolist()).issubset(set(range(traj_len)))

    obs_result = result["observation"]["state"].numpy()
    assert obs_result.shape == (expected_len, feature_dim)
    original_rows = {tuple(row.tolist()) for row in observation_state.numpy()}
    result_rows = {tuple(row.tolist()) for row in obs_result}
    assert result_rows.issubset(original_rows)

    task_result = result["task"]["label"].numpy()
    assert task_result.shape[0] == expected_len
    assert set(task_result.tolist()).issubset(set(range(traj_len)))


@settings(max_examples=40, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    history=st.integers(min_value=1, max_value=4),
    feature_dim=st.integers(min_value=1, max_value=3),
)
def test_zero_out_future_proprio(traj_len, history, feature_dim):
    """Future proprio steps should be zeroed while the first step stays intact."""
    proprio = tf.random.uniform((traj_len, history, feature_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    before = proprio.numpy()
    traj = {"observation": {"proprio": proprio}}

    result = traj_transforms.zero_out_future_proprio(traj)
    after = result["observation"]["proprio"].numpy()

    np.testing.assert_allclose(after[:, 0], before[:, 0])
    if history > 1:
        np.testing.assert_array_equal(
            after[:, 1:],
            np.zeros((traj_len, history - 1, feature_dim), dtype=after.dtype),
        )


@settings(max_examples=50, deadline=None)
@given(
    traj_len=st.integers(min_value=1, max_value=6),
    data=st.data(),
)
def test_add_pad_mask_dict_string_detection(traj_len, data):
    """String tensors are marked as padding based on emptiness while numerics stay true."""
    obs_string_flags = np.array(data.draw(st.lists(st.booleans(), min_size=traj_len, max_size=traj_len)))
    task_string_flags = np.array(data.draw(st.lists(st.booleans(), min_size=traj_len, max_size=traj_len)))

    observation_strings = tf.constant([b"text" if flag else b"" for flag in obs_string_flags], dtype=tf.string)
    task_strings = tf.constant([b"goal" if flag else b"" for flag in task_string_flags], dtype=tf.string)

    traj = {
        "observation": {
            "image_rgb": observation_strings,
            "state": tf.ones((traj_len, 2), dtype=tf.float32),
        },
        "task": {
            "language_instruction": task_strings,
            "goal": tf.ones((traj_len,), dtype=tf.float32),
        },
        "action": tf.zeros((traj_len, 1), dtype=tf.float32),
    }

    result = traj_transforms.add_pad_mask_dict(traj)

    obs_mask = result["observation"]["pad_mask_dict"]["image_rgb"].numpy()
    np.testing.assert_array_equal(obs_mask, obs_string_flags)
    state_mask = result["observation"]["pad_mask_dict"]["state"].numpy()
    np.testing.assert_array_equal(state_mask, np.ones(traj_len, dtype=bool))

    task_mask = result["task"]["pad_mask_dict"]["language_instruction"].numpy()
    np.testing.assert_array_equal(task_mask, task_string_flags)
    goal_mask = result["task"]["pad_mask_dict"]["goal"].numpy()
    np.testing.assert_array_equal(goal_mask, np.ones(traj_len, dtype=bool))
