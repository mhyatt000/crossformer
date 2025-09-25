"""Pipeline transform examples mirroring :mod:`tests.grain.test_transforms_module`."""

from __future__ import annotations

import numpy as np

from crossformer.data.grain import transforms


def make_base_trajectory(length: int = 5, seed: int = 0) -> dict:
    """Create a trajectory with rgb, proprio, and language fields."""
    rng = np.random.default_rng(seed)
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
        trajectory, window_size=3, action_horizon=2
    )
    return trajectory


def pad_mask_demo() -> dict[str, np.ndarray]:
    """Ensure pad mask dictionaries store boolean arrays per modality."""
    traj = {"observation": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}}
    result = transforms.add_pad_mask_dict(traj)
    return {"image": result["observation"]["pad_mask_dict"]["image"]}


def head_mask_demo(traj: dict) -> np.ndarray:
    """Apply head-specific dataset filtering via :func:`add_head_action_mask`."""
    mapping = {"arm": ["demo"], "unused": ["other"]}
    result = transforms.add_head_action_mask(traj, mapping)
    return result["action_head_masks"]["arm"]


def chunk_shape_demo(traj: dict) -> dict[str, tuple[int, ...]]:
    """Inspect the shapes produced by chunking actions and observations."""
    return {
        "rgb": traj["observation"]["rgb"].shape,
        "timestep_mask": traj["observation"]["timestep_pad_mask"].shape,
        "action": traj["action"].shape,
        "action_mask": traj["action_pad_mask"].shape,
        "task_completed": traj["observation"]["task_completed"].shape,
    }


def subsample_demo(traj: dict) -> int:
    """Randomly subsample frames using :func:`subsample`."""
    rng = np.random.default_rng(0)
    subsampled = transforms.subsample(traj, length=3, rng=rng)
    return subsampled["action"].shape[0]


def flatten_demo(traj: dict) -> list[dict]:
    """Flatten the chunked trajectory into per-timestep frames."""
    return list(transforms.flatten_trajectory(traj))


def goal_relabel_demo() -> dict:
    """Relabel goals uniformly to augment data."""
    traj = {
        "action": np.zeros((3, 1), dtype=np.float32),
        "observation": {"image": np.arange(9, dtype=np.float32).reshape(3, 3)},
    }
    rng = np.random.default_rng(0)
    return transforms.uniform_goal_relabel(traj, rng=rng)


def dtype_cast_demo() -> tuple[np.ndarray, np.ndarray]:
    """Demonstrate dtype casting helper used before chunking."""
    values = np.array([1, 2, 3], dtype=np.int32)
    cast = transforms.maybe_cast_dtype(values, np.float32)
    same = transforms.maybe_cast_dtype(cast, np.float32)
    return cast, same


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    base = make_base_trajectory()
    print("pad mask", pad_mask_demo()["image"].dtype)
    print("head mask", head_mask_demo(base))
    print("shapes", chunk_shape_demo(base))
    print("subsample length", subsample_demo(base))
    print("flatten frames", len(flatten_demo(base)))
    print("goal relabel keys", goal_relabel_demo().keys())
    print("dtype cast", [arr.dtype for arr in dtype_cast_demo()])
