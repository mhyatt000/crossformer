"""High level utilities for constructing Grain based data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
import logging
from typing import Any

import grain
import grain.python as gp
import numpy as np

from crossformer.data.grain import builders, transforms
from crossformer.data.grain.datasets import _DecodedArrayRecord, _EpisodeDataset, CacheIter, drop
from crossformer.data.grain.map.window import (  # noqa
    FlatMapDataset,
    FlattenTreeDataset,
    MyFlatMap,
    WindowedFlatMap,
    WindowFlatDataset,
)
from crossformer.data.grain.util.remap import _remap_lang, rekey
from crossformer.data.oxe import HEAD_TO_DATASET
from crossformer.utils.spec import ModuleSpec

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
    ds: gp.MapDataset,
    *,
    window_size: int = 1,
    action_horizon: int = 50,
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

    # filters
    ds = ds.filter(_filter_language_present) if skip_unlabeled else ds
    _max_act_filter = lambda traj: np.all(np.abs(np.asarray(traj["action"])) <= max_action)
    ds = ds.filter(_max_act_filter) if max_action else ds
    ds = ds.filter(_proprio_within_bounds) if max_proprio else ds

    ds = ds.map(transforms.add_pad_mask_dict)
    ds = ds.map(
        lambda traj: transforms.pad_actions_and_proprio(
            traj,
            max_action_dim=max_action_dim,
            max_proprio_dim=max_proprio_dim,
        )
    )

    ds = ds.seed(seed)

    if goal_relabeling_strategy is not None:
        if goal_relabeling_strategy != "uniform":
            raise ValueError(f"Unsupported goal relabeling strategy: {goal_relabeling_strategy}")
        kwargs = goal_relabeling_kwargs or {}
        ds = ds.random_map(lambda traj, rng: transforms.uniform_goal_relabel(traj, rng=rng, **kwargs))

    log.debug("TODO chunk actions by jax.tree.map")
    chunk = partial(
        transforms.chunk_action_and_observation,
        window_size=window_size,
        action_horizon=action_horizon,
        override_window_size=override_window_size,
    )
    ds = ds.map(chunk)

    ds = ds.map(lambda traj: transforms.add_head_action_mask(traj, head_to_dataset=HEAD_TO_DATASET))

    for transform in post_chunk_transforms:
        fn = _resolve_callable(transform)
        ds = ds.map(fn) if fn else ds

    return ds


class _FlattenIterDataset(gp.IterDataset):
    """
    Flattens a trajectory dataset into a frame dataset by yielding all frames
    from each trajectory in sequence.
    """

    def __init__(self, parent: gp.MapDataset):
        super().__init__(parent)

    def __iter__(self):
        parent_iter = self._parent
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
    ds: gp.IterDataset,
    frame_transforms: Sequence[ModuleSpec | Callable] = (),
) -> gp.IterDataset:
    """Applies frame level transforms as simple map operations."""

    def onimg(tree, fn):
        """dont apply fn on trajectory, apply on image dict"""
        im = tree["observation"]["image"]
        im = fn(im)
        tree["observation"]["image"] = im
        return tree

    for transform in frame_transforms:
        fn = _resolve_callable(transform)
        # ds = ds.map(partial(onimg, fn=fn)) if fn else ds
        log.warn("TODO use augmax for frame transform")
        log.warn("TODO apply frame transforms only on image dict")
        ds = ds.map(fn) if fn else ds
    return ds


@dataclass
class GrainDataset:
    dataset: gp.IterDataset
    statistics: Any
    config: builders.GrainDatasetConfig


@dataclass
class TransformConfig:
    traj_transform_kwargs: dict[str, Any] | None = None
    frame_transforms: Sequence[ModuleSpec | Callable] = ()
    resize_frames_to: int | tuple[int, int] | None = None
    resize_frame_keys: Sequence[str] | None = None
    resize_interpolation: str = "bilinear"


def flat(tree):
    return {".".join(k): v for k, v in ftu.flatten_dict(tree).items()}


def unflat(tree):
    return ftu.unflatten_dict({tuple(k.split(".")): v for k, v in tree.items()})


def compatibility(tree: dict):
    """Ensures compatibility with older dataset formats by renaming keys."""

    # compatibility with current dataloader
    tree = flat(tree)
    tree = rekey(
        tree,
        inp=[
            "observation.image.overhead",
            "observation.image.side",
            "observation.image.worm",
            "observation.image.wrist",
            "observation.pad_mask_dict.overhead",
            "observation.pad_mask_dict.side",
            "observation.pad_mask_dict.worm",
            "observation.pad_mask_dict.wrist",
        ],
        out=[
            "observation.image_overhead",
            "observation.image_side",
            "observation.image_worm",
            "observation.image_wrist",
            "observation.pad_mask_dict.image_overhead",
            "observation.pad_mask_dict.image_side",
            "observation.pad_mask_dict.image_worm",
            "observation.pad_mask_dict.image_wrist",
        ],
    )
    tree["observation.proprio.single_arm"] = np.concatenate(
        [
            tree["observation.proprio.gripper"],
            tree["observation.proprio.joints"],
            tree["observation.proprio.position"],
        ],
        axis=-1,
    )

    tree = unflat(tree)


def make_data_source(cfg: Config) -> grain.MapDataset:
    dataset_name = cfg.dataset_name or cfg.arec_path.name

    shards = sorted(cfg.arec_path.glob("*.arrayrecord"))
    if not shards:
        raise FileNotFoundError(f"No ArrayRecord shards found in {cfg.arec_path}")

    ds = _DecodedArrayRecord(shards[:5])
    ds = _EpisodeDataset(ds)

    # ep_lengths = ds.lengths()
    ds = CacheIter(ds)
    ds = grain.MapDataset.source(ds)

    ds = ds.map(partial(drop, keys=["discount", "is_first", "is_terminal", "reward"]))
    ds = ds.map(
        partial(
            rekey,
            inp=["language_instruction", "language_embedding"],
            out=["language.instruction", "language.embedding"],
        )
    )

    example = ds[0]
    pprint(spec(example))

    mappings = _infer_observation_mappings(example)
    pprint(mappings)
    assert mappings, "Trajectory missing observation key"

    language = example.get("language.instruction")
    standardize_fn = _remap_lang if language is not None else None
    keys = builders.Keys(
        *mappings,
        "language.instruction" if language is not None else None,
    )

    dataset_config = builders.GrainDatasetConfig(
        name=dataset_name,
        source=ds,
        keys=keys,
        standardize_fn=standardize_fn,
        skip_norm_keys=cfg.data.transform.skip_norm_keys,
        force_recompute_dataset_statistics=cfg.recompute,
    )

    traj_kwargs = cfg.data.traj.create(with_head_to_dataset=False)
    traj_kwargs["window_size"] = cfg.window_size or traj_kwargs.get("window_size", 1)
    traj_kwargs.pop("task_augment_strategy")
    traj_kwargs.pop("task_augment_kwargs")
    tfconfig = pipelines.TransformConfig(
        traj_transform_kwargs=traj_kwargs,
        frame_transforms={},
        resize_frames_to=64,
        resize_frame_keys=None,
    )
    return ds, dataset_config, tfconfig


def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
    tfconfig: TransformConfig | None = TransformConfig(),
    shuffle_buffer_size: int | None = None,
    drop_remainder: bool = True,
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
    ds, stats = builders.build_trajectory_dataset(config)
    ds = ds.shuffle(seed=10)

    # 2. Apply trajectory transforms
    # 2.1. filter no lang
    # 2.2. maybe filter max action
    # 2.3. add pad mask and head masks
    # 2.4. seed
    # 2.5. goal relabel
    # 2.6. chunking and windowing
    # 2.7. maybe other transforms
    ds = traj_ds = apply_trajectory_transforms(ds, seed=seed)  # , **asdict(tfconfig))

    # chain = augmax.Chain( augmax.Resize(re_wh),)
    log.warn("TODO use augmax for frame transform")
    # Augmenting a single image on the GPU
    # transformed_image = jax.jit(transform)(rng, image)
    # Augmenting an entire batch of images on the GPU
    # sub_rngs = jax.random.split(rng, images.shape[0])
    # transformed_images = jax.jit(jax.vmap(transform))(sub_rngs, images)

    log.warning("we need to prefetch episodes to maintain good speed")
    log.warning("TODO find a best place to prefetch")
    # ds = list(traj_ds.to_iter_dataset())

    ds = ds.repeat() if train else ds
    read_options = grain.sources.ReadOptions(num_threads=1, prefetch_buffer_size=5_000)
    ds = ds.to_iter_dataset(read_options).flat_map(MyFlatMap())
    # ds = list(ds)

    # shuffle along episode + shuffle episodes
    ds = grain.experimental.WindowShuffleIterDataset(ds, window_size=1000, seed=42)
    # ds = ds.map(lambda tree: jax.tree.map(lambda x: jnp.asarray(x), tree))
    # ds = ds.map(jax.device_put)

    # 3. do frame level transforms
    # 3.1. x decoding is already done
    # 3.2. resize frames if needed
    # 3.3. augmentations and dropout
    log.warning("TODO 3.3. aug&dropout")

    re = tfconfig.resize_frames_to
    re_wh = re[0] if isinstance(re, tuple) else re

    combined_transforms: list[ModuleSpec | Callable] = []
    if re is not None:
        # if tfconfig.resize_frame_keys not set we default to all of them
        re_keys = tfconfig.resize_frame_keys or config.keys.image
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

    def add_mask(tree: dict):
        tree["mask"] = {}
        return tree

    ds = ds.map(add_mask)

    ds.map(compatibility)

    bs = config.batch_size
    ds = ds.batch(bs, drop_remainder=drop_remainder) if bs is not None else ds
    # ds = ds.mp_prefetch( gmp.MultiprocessingOptions(num_workers=2, per_worker_buffer_size=1, enable_profiling=True))

    log.warning("TODO batch earlier in pipe to speed up processing")
    ds.dataset_statistics = stats  # type: ignore[attr-defined]
    log.debug("returning final dataset")
    return GrainDataset(dataset=ds, statistics=stats, config=config)


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
