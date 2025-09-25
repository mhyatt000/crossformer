"""Trajectory transformation tour based on :mod:`tests.test_traj_transforms`."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from crossformer.data import traj_transforms


def make_synthetic_trajectory(
    length: int = 4,
    obs_dim: int = 3,
    action_dim: int = 2,
    seed: int = 0,
) -> dict[str, dict[str, tf.Tensor] | tf.Tensor]:
    """Create TensorFlow-backed tensors that mirror the test fixtures."""
    rng = np.random.default_rng(seed)
    observation = {
        "state": tf.constant(
            rng.normal(size=(length, obs_dim)).astype(np.float32)
        )
    }
    action = tf.constant(rng.normal(size=(length, action_dim)).astype(np.float32))
    traj = {
        "observation": observation,
        "action": action,
        "task": {},
        "action_pad_mask": tf.ones((length, action_dim), dtype=tf.bool),
    }
    return traj


def chunk_history_example() -> dict[str, tf.Tensor]:
    """Chunk raw actions and observe the resulting masks."""
    traj = make_synthetic_trajectory(length=5, obs_dim=2, action_dim=2)
    chunked = traj_transforms.chunk_act_obs(
        traj, window_size=3, action_horizon=2
    )
    state = chunked["observation"]["state"].numpy()
    timestep_mask = chunked["observation"]["timestep_pad_mask"].numpy()
    action = chunked["action"].numpy()
    action_mask = chunked["action_pad_mask"].numpy()
    return {
        "state": state,
        "timestep_mask": timestep_mask,
        "action": action,
        "action_mask": action_mask,
    }


def padding_and_goal_example() -> dict[str, tf.Tensor]:
    """Pad trajectories, tag goals, and build per-head action masks."""
    traj = make_synthetic_trajectory(length=4)
    traj["observation"]["pad_mask_dict"] = {
        "state": tf.constant([[True], [True], [False], [True]])
    }
    traj["task"]["timestep"] = tf.constant([3, 3, 3, 3], dtype=tf.int32)

    padded = traj_transforms.pad_actions_and_proprio(
        traj, max_action_dim=4, max_proprio_dim=None
    )
    chunked = traj_transforms.chunk_act_obs(
        padded, window_size=3, action_horizon=2
    )
    with_masks = traj_transforms.add_head_action_mask(
        chunked, head_to_dataset={"arm": tf.constant([b"demo"] * chunked["action"].shape[0])}
    )
    return {
        "padded_action": padded["action"].numpy(),
        "action_mask": padded["action_pad_mask"].numpy(),
        "task_completed": chunked["observation"]["task_completed"].numpy(),
        "head_mask": with_masks["action_head_masks"]["arm"].numpy(),
    }


def subsample_and_zero_proprio_example() -> dict[str, tf.Tensor]:
    """Subsample frames then clear future proprio steps to zero."""
    traj = make_synthetic_trajectory(length=6, obs_dim=3)
    traj["observation"]["proprio"] = tf.constant(
        np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    )
    padded = traj_transforms.pad_actions_and_proprio(
        traj, max_action_dim=3, max_proprio_dim=4
    )
    chunked = traj_transforms.chunk_act_obs(
        padded, window_size=2, action_horizon=2
    )
    subsampled = traj_transforms.subsample(chunked, subsample_length=3)
    proprio = traj_transforms.zero_out_future_proprio(subsampled)
    pad_dict = traj_transforms.add_pad_mask_dict(proprio)
    return {
        "subsampled_len": subsampled["action"].shape[0],
        "proprio": proprio["observation"]["proprio"].numpy(),
        "pad_mask_dict": {
            key: value.numpy()
            for key, value in pad_dict["observation"]["pad_mask_dict"].items()
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("chunk_history", chunk_history_example()["state"].shape)
    print("padding_and_goal", padding_and_goal_example()["padded_action"].shape)
    print("subsample", subsample_and_zero_proprio_example()["subsampled_len"])
