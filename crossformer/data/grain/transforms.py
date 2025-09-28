"""Trajectory and frame level transforms implemented for Google Grain.

The original TensorFlow pipeline exposes a fairly rich set of transforms that
operate on either complete trajectories or on frame level chunks.  Rewriting
every single transform for NumPy/JAX would be unnecessary for the initial
Grain migration, however the core operations – chunking, padding, head masks,
and simple subsampling – are required for parity.  This module provides these
rewritten utilities in a framework agnostic way.

All functions operate purely on nested dictionaries containing NumPy arrays or
values that are trivially convertible to NumPy arrays.  They are therefore
compatible with both ``grain.MapDataset`` and ``grain.IterDataset`` pipelines
without introducing TensorFlow as a dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, Optional

import numpy as np
from scipy import ndimage

from crossformer.data.grain import utils


ArrayDict = dict[str, Any]
Trajectory = dict[str, Any]


_INTERPOLATION_TO_ORDER = {"nearest": 0, "bilinear": 1}


def _ensure_array(value: Any) -> np.ndarray:
    return utils.ensure_numpy(value)


def _copy_traj(traj: Trajectory) -> Trajectory:
    return utils.clone_structure(traj)


def add_pad_mask_dict(traj: Trajectory) -> Trajectory:
    """Annotates ``traj`` with padding masks for observation/task dictionaries."""

    traj = _copy_traj(traj)
    pad_masks = utils.tree_map(utils.is_padding, traj)
    observation_masks = pad_masks.get("observation", {})
    task_masks = pad_masks.get("task", {})
    traj.setdefault("observation", {})
    traj.setdefault("task", {})
    traj["observation"]["pad_mask_dict"] = observation_masks
    traj["task"]["pad_mask_dict"] = task_masks
    return traj


def add_head_action_mask(
    traj: Trajectory, head_to_dataset: Mapping[str, Sequence[str]] | None = None
) -> Trajectory:
    """Adds per-head action masks mirroring TensorFlow implementation."""

    traj = _copy_traj(traj)
    dataset_names = traj.get("dataset_name")
    if dataset_names is None:
        raise KeyError("Expected 'dataset_name' field when adding head masks.")
    dataset_names = np.asarray(dataset_names)
    if head_to_dataset is None:
        traj["action_head_masks"] = {
            "action": np.ones_like(dataset_names, dtype=bool)
        }
        return traj

    action_masks = {}
    for head, dataset_list in head_to_dataset.items():
        dataset_array = np.asarray(dataset_list, dtype=dataset_names.dtype)
        mask = np.isin(dataset_names, dataset_array)
        action_masks[head] = mask
    traj["action_head_masks"] = action_masks
    return traj


def pad_actions_and_proprio(
    traj: Trajectory,
    *,
    max_action_dim: int | None,
    max_proprio_dim: int | None,
) -> Trajectory:
    """Pads actions/proprio streams and records action padding mask."""

    traj = _copy_traj(traj)
    actions = _ensure_array(traj["action"])
    traj["action_pad_mask"] = np.ones_like(actions, dtype=bool)

    if max_action_dim is not None:
        action_dim = actions.shape[-1]
        if action_dim > max_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) is greater than max_action_dim ({max_action_dim})"
            )
        pad_width = [(0, 0)] * (actions.ndim - 1) + [(0, max_action_dim - action_dim)]
        traj["action"] = np.pad(actions, pad_width, mode="constant")
        traj["action_pad_mask"] = np.pad(
            traj["action_pad_mask"], pad_width, mode="constant", constant_values=False
        )

    if max_proprio_dim is not None and "proprio" in traj.get("observation", {}):
        proprio = _ensure_array(traj["observation"]["proprio"])
        proprio_dim = proprio.shape[-1]
        if proprio_dim > max_proprio_dim:
            raise ValueError(
                f"proprio_dim ({proprio_dim}) is greater than max_proprio_dim ({max_proprio_dim})"
            )
        pad_width = [(0, 0), (0, max_proprio_dim - proprio_dim)]
        traj["observation"]["proprio"] = np.pad(proprio, pad_width, mode="constant")

    return traj


def chunk_action_and_observation(
    traj: Trajectory,
    *,
    window_size: int,
    action_horizon: int,
    override_window_size: Optional[int] = None,
) -> Trajectory:
    """Chunks observations into histories and actions into windows."""

    traj = _copy_traj(traj)
    actions = _ensure_array(traj["action"])
    traj_len = actions.shape[0]

    history_indices = (
        np.arange(traj_len)[:, None] + np.arange(-window_size + 1, 1)[None, :]
    )
    timestep_pad_mask = history_indices >= 0
    if override_window_size is not None:
        valid_history = np.arange(window_size) >= window_size - override_window_size
        timestep_pad_mask = np.logical_and(timestep_pad_mask, valid_history)
    history_indices = np.maximum(history_indices, 0)

    chunked_obs = {}
    for key, value in traj["observation"].items():
        if key == "pad_mask_dict":
            chunked_obs[key] = value
            continue
        value = _ensure_array(value)
        chunked_obs[key] = value[history_indices]
    chunked_obs["timestep_pad_mask"] = timestep_pad_mask
    traj["observation"] = chunked_obs

    if actions.ndim == 2:
        action_indices = (
            np.arange(traj_len)[:, None] + np.arange(action_horizon)[None, :]
        )
        action_indices = np.minimum(action_indices, traj_len - 1)
        actions = actions[action_indices]
    else:
        if actions.shape[1] < action_horizon:
            raise ValueError(
                "Pre-chunked action does not have enough horizon to satisfy"
                f" requested action_horizon={action_horizon}."
            )
        actions = actions[:, :action_horizon]

    actions = actions[history_indices]
    traj["action"] = actions

    if "task" not in traj:
        traj["task"] = {}
    task = traj["task"]
    goal_timestep = task.get("timestep")
    if goal_timestep is None:
        goal_timestep = np.full((traj_len,), traj_len - 1, dtype=np.int32)
    else:
        goal_timestep = _ensure_array(goal_timestep)

    t, w, h = np.meshgrid(
        np.arange(traj_len), np.arange(window_size), np.arange(action_horizon), indexing="ij"
    )
    relative_goal = goal_timestep[:, None, None] - (t - (window_size + 1) + w + h)
    traj["observation"]["task_completed"] = relative_goal <= 0

    action_pad_mask = traj.get("action_pad_mask")
    if action_pad_mask is None:
        action_pad_mask = np.ones((*actions.shape[:-1], actions.shape[-1]), dtype=bool)
    else:
        action_pad_mask = _ensure_array(action_pad_mask)
        if action_pad_mask.ndim == 2:
            action_pad_mask = action_pad_mask[:, None, None, :]
        elif action_pad_mask.ndim == 3:
            action_pad_mask = action_pad_mask[:, None, :, :]
    traj["action_pad_mask"] = np.logical_and(
        action_pad_mask,
        ~traj["observation"]["task_completed"][:, :, :, None],
    )

    return traj


def subsample(traj: Trajectory, *, length: int, rng: np.random.Generator) -> Trajectory:
    traj = _copy_traj(traj)
    traj_len = traj["action"].shape[0]
    if traj_len <= length:
        return traj
    indices = rng.choice(traj_len, size=length, replace=False)
    indices.sort()

    def gather(value: Any):
        if isinstance(value, dict):
            return {key: gather(sub_value) for key, sub_value in value.items()}
        value = _ensure_array(value)
        return value[indices]

    for key in ("action", "observation", "task"):
        if key in traj:
            traj[key] = gather(traj[key])
    return traj


def zero_out_future_proprio(traj: Trajectory) -> Trajectory:
    traj = _copy_traj(traj)
    proprio = traj["observation"].get("proprio")
    if proprio is None:
        return traj
    proprio = _ensure_array(proprio)
    if proprio.ndim < 2:
        raise ValueError("Expected proprio to be at least 2D after chunking.")
    proprio[:, 1:] = 0
    traj["observation"]["proprio"] = proprio
    return traj


def flatten_trajectory(traj: Trajectory) -> Iterable[dict[str, Any]]:
    """Converts a chunked trajectory into a sequence of frame dictionaries."""

    traj_len = traj["action"].shape[0]
    for idx in range(traj_len):
        frame = {
            "observation": utils.tree_map(lambda x: _ensure_array(x)[idx], traj["observation"]),
            "task": utils.tree_map(lambda x: _ensure_array(x)[idx], traj.get("task", {})),
            "action": _ensure_array(traj["action"])[idx],
            "action_pad_mask": _ensure_array(traj["action_pad_mask"])[idx],
            "dataset_name": traj["dataset_name"][idx]
            if isinstance(traj.get("dataset_name"), (np.ndarray, list))
            else traj.get("dataset_name"),
        }
        if "action_head_masks" in traj:
            frame["action_head_masks"] = {
                head: np.asarray(mask)[idx]
                for head, mask in traj["action_head_masks"].items()
            }
        yield frame


def _normalize_resize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("resize dimension must be positive")
        return size, size
    if len(size) != 2:
        raise ValueError("size must be an int or a length-2 tuple")
    height, width = (int(dim) for dim in size)
    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    return height, width


def _is_image_like(value: Any) -> bool:
    array = _ensure_array(value)
    return array.ndim >= 3 and array.shape[-1] in {1, 3, 4}


def _resize_image_array(
    value: Any,
    *,
    size: tuple[int, int],
    order: int,
) -> np.ndarray:
    array = _ensure_array(value)
    if array.ndim < 3:
        raise ValueError("Expected image-like array with channel dimension")
    height_axis = array.ndim - 3
    width_axis = array.ndim - 2
    zoom_factors = [1.0] * array.ndim
    zoom_factors[height_axis] = size[0] / array.shape[height_axis]
    zoom_factors[width_axis] = size[1] / array.shape[width_axis]
    working_array = array
    orig_dtype = working_array.dtype
    if np.issubdtype(orig_dtype, np.integer):
        working_array = working_array.astype(np.float32)
    resized = ndimage.zoom(working_array, zoom_factors, order=order)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        resized = np.clip(np.rint(resized), info.min, info.max).astype(orig_dtype)
    else:
        resized = resized.astype(orig_dtype)
    return resized


def resize_frame_images(
    frame: dict[str, Any],
    *,
    size: int | tuple[int, int],
    keys: Sequence[str] | None = None,
    interpolation: str = "bilinear",
) -> dict[str, Any]:
    """Resizes image-like observations inside ``frame`` to ``size``.

    Parameters
    ----------
    frame:
        Frame dictionary emitted by :func:`flatten_trajectory`.
    size:
        Target spatial resolution.  Accepts an integer for square resizing or a
        ``(height, width)`` tuple for non-square outputs.
    keys:
        Optional observation keys to resize.  When ``None`` the transform
        automatically selects keys whose arrays end with a channel dimension of
        size ``1``, ``3`` or ``4`` (``rgb``/``rgba``/single channel images).
    interpolation:
        Either ``"nearest"`` or ``"bilinear"``.
    """

    order = _INTERPOLATION_TO_ORDER.get(interpolation)
    if order is None:
        raise ValueError(
            f"Unsupported interpolation '{interpolation}'. "
            f"Expected one of {sorted(_INTERPOLATION_TO_ORDER)}"
        )
    target_size = _normalize_resize_size(size)
    frame = utils.clone_structure(frame)
    observations = frame.get("observation")
    if not isinstance(observations, dict):
        return frame
    if keys is None:
        keys = [
            key
            for key, value in observations.items()
            if key != "pad_mask_dict" and _is_image_like(value)
        ]
    for key in keys:
        if key not in observations:
            continue
        observations[key] = _resize_image_array(
            observations[key], size=target_size, order=order
        )
    frame["observation"] = observations
    return frame


def drop_empty_language(traj: Trajectory) -> Trajectory:
    traj = _copy_traj(traj)
    language = traj.get("task", {}).get("language_instruction")
    if language is None:
        return traj
    language = np.asarray(language)
    keep = language != ""
    if np.all(~keep):
        raise ValueError("Trajectory does not contain any language annotation.")
    return traj


def uniform_goal_relabel(traj: Trajectory, *, rng: np.random.Generator) -> Trajectory:
    traj = _copy_traj(traj)
    obs = traj["observation"]
    traj_len = _ensure_array(traj["action"]).shape[0]
    rand = rng.uniform(size=traj_len)
    low = np.arange(traj_len)
    high = np.full(traj_len, traj_len)
    goal_indices = (rand * (high - low) + low).astype(int)
    goal_indices = np.clip(goal_indices, 0, traj_len - 1)
    goal = utils.tree_map(lambda x: _ensure_array(x)[goal_indices], obs)
    traj.setdefault("task", {})
    traj["task"] = utils.tree_merge(traj["task"], goal)
    return traj


def maybe_cast_dtype(value: Any, dtype: np.dtype | type[np.number]) -> np.ndarray:
    array = _ensure_array(value)
    if array.dtype != dtype:
        array = array.astype(dtype)
    return array

