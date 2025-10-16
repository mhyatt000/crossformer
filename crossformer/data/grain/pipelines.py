"""High level utilities for constructing Grain based data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
import logging
from typing import Any

import grain.python as gp
import numpy as np
from rich.pretty import pprint

from crossformer.data.grain import builders, transforms
from crossformer.data.oxe import HEAD_TO_DATASET
from crossformer.utils.spec import ModuleSpec, spec

log = logging.getLogger(__name__)


def _resolve_callable(spec_or_fn: ModuleSpec | Callable | None) -> Callable | None:
    if spec_or_fn is None:
        return None
    if isinstance(spec_or_fn, Mapping) and set(spec_or_fn.keys()) == {
        "module",
        "name",
        "args",
        "kwargs",
    }:
        return ModuleSpec.instantiate(spec_or_fn)  # type: ignore[arg-type]
    if not callable(spec_or_fn):
        raise TypeError(f"Expected callable or ModuleSpec, got {type(spec_or_fn)!r}")
    return spec_or_fn  # type: ignore[return-value]


def _filter_language_present(traj: dict) -> bool:
    language = traj.get("task", {}).get("language_instruction")
    if language is None:
        return False
    language = np.asarray(language)
    return np.any(language != "")


def _proprio_within_bounds(traj: dict, max_proprio: float) -> bool:
    for key, value in traj.get("observation", {}).items():
        if key.startswith("proprio") and not np.all(np.abs(np.asarray(value)) <= max_proprio):
            return False
    return True


def apply_trajectory_transforms(
    dataset: gp.MapDataset,
    *,
    train: bool,
    window_size: int = 1,
    action_horizon: int = 1,
    override_window_size: int | None = None,
    goal_relabeling_strategy: str | None = None,
    goal_relabeling_kwargs: Mapping[str, Any] | None = None,
    subsample_length: int | None = None,
    skip_unlabeled: bool = False,
    max_action: float | None = None,
    max_proprio: float | None = None,
    max_action_dim: int | None = None,
    max_proprio_dim: int | None = None,
    post_chunk_transforms: Sequence[ModuleSpec | Callable] = (),
    seed: int = 0,
) -> gp.MapDataset:
    """Applies trajectory level transforms mirroring the TensorFlow pipeline."""

    if skip_unlabeled:
        dataset = dataset.filter(_filter_language_present)

    if max_action is not None:
        dataset = dataset.filter(lambda traj: np.all(np.abs(np.asarray(traj["action"])) <= max_action))

    if max_proprio is not None:
        dataset = dataset.filter(_proprio_within_bounds)

    dataset = dataset.map(transforms.add_pad_mask_dict)
    dataset = dataset.map(
        lambda traj: transforms.pad_actions_and_proprio(
            traj,
            max_action_dim=max_action_dim,
            max_proprio_dim=max_proprio_dim,
        )
    )

    dataset = dataset.seed(seed)

    if goal_relabeling_strategy is not None:
        if goal_relabeling_strategy != "uniform":
            raise ValueError(f"Unsupported goal relabeling strategy: {goal_relabeling_strategy}")
        kwargs = goal_relabeling_kwargs or {}
        dataset = dataset.random_map(lambda traj, rng: transforms.uniform_goal_relabel(traj, rng=rng, **kwargs))

    pprint(spec(next(iter(dataset))))
    dataset = dataset.map(
        lambda traj: transforms.chunk_action_and_observation(
            traj,
            window_size=window_size,
            action_horizon=action_horizon,
            override_window_size=override_window_size,
        )
    )

    pprint(spec(next(iter(dataset))))
    print("c")

    dataset = dataset.map(lambda traj: transforms.add_head_action_mask(traj, head_to_dataset=HEAD_TO_DATASET))
    pprint(spec(next(iter(dataset))))
    print("d")

    for transform in post_chunk_transforms:
        callable_transform = _resolve_callable(transform)
        if callable_transform is None:
            continue
        dataset = dataset.map(callable_transform)

    return dataset


class _FlattenIterDataset(gp.IterDataset):
    """
    Flattens a trajectory dataset into a frame dataset by yielding all frames
    from each trajectory in sequence.
    """

    def __init__(self, parent: gp.MapDataset):
        super().__init__(parent)

    def __iter__(self):
        parent_iter = self._parent.__iter__()
        for trajectory in parent_iter:
            yield from transforms.flatten_trajectory(trajectory)


class _BufferShuffleIterDataset(gp.IterDataset):
    def __init__(self, parent: gp.IterDataset, buffer_size: int, seed: int):
        super().__init__(parent)
        self._buffer_size = buffer_size
        self._seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        buffer = []
        for element in self._parent:
            buffer.append(element)
            if len(buffer) >= self._buffer_size:
                rng.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        if buffer:
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()


def apply_frame_transforms(
    dataset: gp.IterDataset,
    frame_transforms: Sequence[ModuleSpec | Callable] = (),
) -> gp.IterDataset:
    """Applies frame level transforms as simple map operations."""

    for transform in frame_transforms:
        callable_transform = _resolve_callable(transform)
        if callable_transform is None:
            continue
        dataset = dataset.map(callable_transform)
    return dataset


@dataclass
class GrainDataset:
    dataset: gp.IterDataset
    statistics: Any


@dataclass
class TransformConfig:
    traj_transform_kwargs: dict[str, Any] | None = None
    frame_transforms: Sequence[ModuleSpec | Callable] = ()
    resize_frames_to: int | tuple[int, int] | None = None
    resize_frame_keys: Sequence[str] | None = None
    resize_interpolation: str = "bilinear"


def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
    tfconfig: TransformConfig | None = TransformConfig(),
    shuffle_buffer_size: int | None = None,
    batch_size: int | None = None,
    drop_remainder: bool = False,
    seed: int = 0,
) -> GrainDataset:
    """Builds a dataset of frames for a single dataset configuration.

    When ``resize_frames_to`` is provided the image observations within each
    frame are resized via :func:`transforms.resize_frame_images` before applying
    any additional ``frame_transforms``.
    """

    # 1. Build the trajectory dataset
    # 1.1. restructure keys
    # 1.2. compute / load statistics
    # 1.3. normalize with statistics and norm mask
    traj_dataset, stats = builders.build_trajectory_dataset(config)

    # 2. Apply trajectory transforms
    # 2.1. filter no lang
    # 2.2. maybe filter max action
    # 2.3. add pad mask
    # 2.4. seed
    # 2.5. goal relabel
    # 2.6. chunking and windowing
    # 2.7. maybe other transforms
    traj_dataset = apply_trajectory_transforms(traj_dataset, train=train, seed=seed)  # , **asdict(tfconfig))
    from rich.pretty import pprint

    from crossformer.utils.spec import spec

    pprint(spec(next(iter(traj_dataset))))

    re = tfconfig.resize_frames_to
    re_wh = re[0] if isinstance(re, tuple) else re

    # chain = augmax.Chain( augmax.Resize(re_wh),)
    log.warn("TODO use augmax for frame transform")
    # Augmenting a single image on the GPU
    # transformed_image = jax.jit(transform)(rng, image)
    # Augmenting an entire batch of images on the GPU
    # sub_rngs = jax.random.split(rng, images.shape[0])
    # transformed_images = jax.jit(jax.vmap(transform))(sub_rngs, images)

    # 3. do frame level transforms
    ds: gp.IterDataset = _FlattenIterDataset(traj_dataset)
    combined_transforms: list[ModuleSpec | Callable] = []
    if re is not None:
        re_keys = tfconfig.resize_frame_keys if tfconfig.resize_frame_keys is not None else config.keys.image
        combined_transforms.append(
            partial(
                transforms.resize_frame_images,
                size=re,
                keys=re_keys,
                interpolation=tfconfig.resize_interpolation,
            )
        )

    combined_transforms.extend(tfconfig.frame_transforms)
    ds = apply_frame_transforms(ds, combined_transforms)

    if shuffle_buffer_size and shuffle_buffer_size > 1:
        ds = _BufferShuffleIterDataset(ds, buffer_size=shuffle_buffer_size, seed=seed)

    ds = ds.repeat() if train else ds
    ds = ds.batch(batch_size, drop_remainder=drop_remainder) if batch_size is not None else ds
    ds.dataset_statistics = stats  # type: ignore[attr-defined]
    return GrainDataset(dataset=ds, statistics=stats)


def make_interleaved_dataset(
    configs: Sequence[builders.GrainDatasetConfig],
    *,
    train: bool,
    sample_weights: Sequence[float] | None = None,
    shuffle_buffer_size: int = 1,
    traj_transform_kwargs: dict[str, Any] | None = None,
    frame_transforms: Sequence[ModuleSpec | Callable] = (),
    resize_frames_to: int | tuple[int, int] | None = None,
    resize_frame_keys: Sequence[str] | None = None,
    resize_interpolation: str = "bilinear",
    batch_size: int | None = None,
    drop_remainder: bool = False,
    seed: int = 0,
) -> GrainDataset:
    """Creates a weighted mixture of datasets similar to the TensorFlow pipeline.

    Resizing is performed independently for every dataset when
    ``resize_frames_to`` is specified.
    """

    if not configs:
        raise ValueError("At least one dataset configuration must be provided.")

    datasets = []
    statistics = {}
    for cfg in configs:
        ds = make_single_dataset(
            cfg,
            train=train,
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transforms=frame_transforms,
            resize_frames_to=resize_frames_to,
            resize_frame_keys=resize_frame_keys,
            resize_interpolation=resize_interpolation,
            shuffle_buffer_size=None,
            batch_size=None,
            seed=seed,
        )
        datasets.append(ds.dataset)
        statistics[cfg.name] = ds.statistics

    if sample_weights is None:
        weights = np.ones(len(datasets), dtype=np.float32)
    else:
        weights = np.asarray(sample_weights, dtype=np.float32)
        if weights.shape[0] != len(datasets):
            raise ValueError("Number of sample weights must match number of datasets.")
    weights = weights / weights.sum()

    class _MixtureIterDataset(gp.IterDataset):
        def __init__(self, children: Sequence[gp.IterDataset], probs: np.ndarray):
            super().__init__(children)
            self._children = children
            self._probs = probs

        def __iter__(self):
            rng = np.random.default_rng(seed)
            child_iters = [iter(child) for child in self._children]
            while True:
                choice = rng.choice(len(child_iters), p=self._probs)
                try:
                    yield next(child_iters[choice])
                except StopIteration:
                    return

    mixture_dataset: gp.IterDataset = _MixtureIterDataset(datasets, weights)

    if shuffle_buffer_size and shuffle_buffer_size > 1:
        mixture_dataset = _BufferShuffleIterDataset(mixture_dataset, buffer_size=shuffle_buffer_size, seed=seed)

    if train:
        mixture_dataset = mixture_dataset.repeat()

    if batch_size is not None:
        mixture_dataset = mixture_dataset.batch(batch_size, drop_remainder=drop_remainder)

    mixture_dataset.dataset_statistics = statistics  # type: ignore[attr-defined]
    return GrainDataset(dataset=mixture_dataset, statistics=statistics)
