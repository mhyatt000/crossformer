"""High level utilities for constructing Grain based data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import fnmatch
from functools import partial
import logging
from typing import Any, Sequence, TypedDict

import augmax
import grain.python as gp
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import jaxtyping as jt
import numpy as np

from crossformer.data.grain import builders, metadata, transforms
from crossformer.data.grain.datasets import (
    drop,
)
from crossformer.data.grain.util.remap import rekey
from crossformer.utils.deco import deprecate
from crossformer.utils.mytyping import DeprecatedError
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.tree import flat, unflat
from crossformer.utils.tree.core import drop_fn

log = logging.getLogger(__name__)
_cpu_device = None


def _grain_cpu_device():
    global _cpu_device
    if _cpu_device is None:
        _cpu_device = jax.devices("cpu")[0]
    return _cpu_device


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
    return language is not None
    language = np.asarray(language)
    return np.any(language != "")


def _proprio_within_bounds(traj: dict, max_proprio: float) -> bool:
    for key, value in traj.get("observation", {}).items():
        if key.startswith("proprio") and not jnp.all(jnp.abs(value) <= max_proprio):
            return False
    return True


def to_jax_key(r):
    if isinstance(r, np.random.Generator):
        return jax.random.key(r.integers(2**32 - 1, dtype=np.uint32))
    if isinstance(r, int | np.integer):
        return jax.random.key(np.uint32(r))
    return r  # already a JAX key (uint32[2])


def apply_trajectory_transforms(
    ds: gp.MapDataset,
    *,
    window_size: int = 1,
    action_horizon: int = 20,
    override_window_size: int | None = None,
    goal_relabeling_strategy: str | None = "uniform",
    goal_relabeling_kwargs: Mapping[str, Any] | None = None,
    subsample_length: int | None = None,
    skip_unlabeled: bool = False,
    max_action: float | None = None,
    max_proprio: float | None = None,
    max_action_dim: int | None = None,
    max_proprio_dim: int | None = None,
    post_chunk_transforms: Sequence[ModuleSpec | Callable] = (),
    seed: int = 0,
    config: builders.GrainDatasetConfig | None = None,
) -> gp.MapDataset:
    """Applies trajectory level transforms mirroring the TensorFlow pipeline."""

    # filters
    ds = ds.filter(_filter_language_present) if skip_unlabeled else ds
    _max_act_filter = lambda traj: jnp.all(jnp.abs(traj["action"]) <= max_action)
    ds = ds.filter(_max_act_filter) if max_action else ds
    ds = ds.filter(_proprio_within_bounds) if max_proprio else ds

    ds = ds.map(transforms.add_pad_mask_dict)
    # ds = ds.map(
    # lambda traj: transforms.pad_actions_and_proprio(
    # traj,
    # max_action_dim=max_action_dim,
    # max_proprio_dim=max_proprio_dim,
    # )
    # )

    """
    if goal_relabeling_strategy is not None:
        if goal_relabeling_strategy != "uniform":
            raise ValueError(f"Unsupported goal relabeling strategy: {goal_relabeling_strategy}")
        kwargs = goal_relabeling_kwargs or {}
        ds = ds.random_map(lambda traj, rng: transforms.uniform_goal_relabel(traj, rng=to_jax_key(rng), **kwargs))
    log.warning("TODO goal relabel kwargs not implemented")

    log.info("TODO chunk actions by jax.tree.map")
    chunk = partial(
        transforms.chunk_action_and_observation,
        window_size=window_size,
        action_horizon=action_horizon,
        override_window_size=override_window_size,
    )
    ds = ds.map(chunk)  # override grain.experimental.ConcatThenSplit
    """

    ds = ds.map(partial(transforms.add_head_action_mask, name=config.name))

    for transform in post_chunk_transforms:
        fn = _resolve_callable(transform)
        ds = ds.map(fn) if fn else ds

    return ds


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


def do_frame_transforms(config, tfconfig, ds, *, imaug: bool = True, rotate: bool = True):
    # 3. do frame level transforms
    # 3.1. x decoding is already done
    # 3.2. resize frames if needed
    # 3.3. augmentations and dropout
    jd = partial(jax.jit, donate_argnums=0)
    frame_transform_aug = jax.jit(get_frame_transform(config, tfconfig, imaug=imaug, rotate=rotate))

    def squeeze(x, dim):
        return jax.tree.map(lambda y: jnp.squeeze(y, axis=dim), x)

    def unsqueeze(x, dim):
        return jax.tree.map(lambda y: jnp.expand_dims(y, axis=dim), x)

    @partial(jax.jit, donate_argnums=(0, 1))
    def frame_aug_with_reshape(rng, batch):
        # unsqueeze task images on extra dim=1
        # batch["task"]["image"] = unsqueeze(batch["task"]["image"], dim=1)
        batch = frame_transform_aug(rng, batch=batch)
        # batch["task"]["image"] = squeeze(batch["task"]["image"], dim=1)
        return batch

    def rng_for_batch(rng, batch):
        key = to_jax_key(rng)
        leaf = jax.tree.leaves(batch)[0]
        sharding = getattr(leaf, "sharding", None)
        mesh = getattr(sharding, "mesh", None)
        if mesh is None:
            return key
        return jax.device_put(key, NamedSharding(mesh, PartitionSpec()))

    ds = (
        # @todo is it better to use mp or to jit with constant size?
        ds.random_map(lambda x, rng: frame_aug_with_reshape(rng=rng_for_batch(rng, x), batch=x))
    )
    return ds


@dataclass
class GrainDataLoader:
    dataset: gp.IterDataset
    statistics: metadata.DatasetStatistics | dict[str, metadata.DatasetStatistics]
    config: builders.GrainDatasetConfig

    @property
    def ds(self):
        return self.dataset

    @property
    @deprecate("for compatibility with tfds", strict=False)
    def dataset_statistics(self) -> Any:
        """return serialize stats for 1+ datasets"""
        if isinstance(self.statistics, metadata.DatasetStatistics):
            return jax.tree.map(jnp.array, self.statistics.to_json())
        else:
            return {k: jax.tree.map(jnp.array, v.to_json()) for k, v in self.statistics.items()}


@dataclass
class TransformConfig:
    traj_transform_kwargs: dict[str, Any] | None = None
    frame_transforms: Sequence[ModuleSpec | Callable] = ()
    resize_frames_to: int | tuple[int, int] | None = None
    resize_frame_keys: Sequence[str] | None = None
    resize_interpolation: str = "bilinear"


class Batch(TypedDict, total=False):  # total=False makes extra keys allowed
    observation: jt.Float[jt.Array, "N 3"]
    action: jt.Array  # or Int[Array, "N"]
    meta: NotRequired[dict]


@deprecate("for compatibility with older dataset (TFDS)", strict=False)
def compatibility(tree: dict):
    """Ensures compatibility with older dataset formats by renaming keys."""

    # compatibility with current dataloader
    tree = flat(tree)

    # IMAGE
    # fix image keys
    side = fnmatch.filter(tree.keys(), "*image.side*")
    tree = rekey(tree, inp=side, out=[k.replace("image.side", "image_side") for k in side])
    worm = fnmatch.filter(tree.keys(), "*image.worm*")
    tree = rekey(tree, inp=worm, out=[k.replace("image.worm", "image_primary") for k in worm])
    low = fnmatch.filter(tree.keys(), "*image.low*")
    tree = rekey(tree, inp=low, out=[k.replace("image.low", "image_primary") for k in low])
    over = fnmatch.filter(tree.keys(), "*image.over*")
    tree = rekey(tree, inp=over, out=[k.replace("image.over", "image_over") for k in over])
    wrist = fnmatch.filter(tree.keys(), "*image.wrist*")
    tree = rekey(tree, inp=wrist, out=[k.replace("image.wrist", "image_left_wrist") for k in wrist])

    # pad_mask_dict = fnmatch.filter(tree.keys(), "*pad_mask_dict.image*")
    # tree = rekey(tree, inp=pad_mask_dict, out=[k.replace("pad_mask_dict.image", "pad_mask_dict") for k in pad_mask_dict])

    overhead = list(fnmatch.filter(tree.keys(), "*overhead*"))
    tree = drop(tree, overhead)
    worm = list(fnmatch.filter(tree.keys(), "*image_worm*"))
    tree = rekey(tree, inp=worm, out=[k.replace("image_worm", "image_primary") for k in worm])
    wrist = list(fnmatch.filter(tree.keys(), "*image_wrist*"))
    tree = rekey(tree, inp=wrist, out=[k.replace("image_wrist", "image_left_wrist") for k in wrist])

    final = list(fnmatch.filter(tree.keys(), "*image.*"))  # image_a image_b image_c
    tree = rekey(tree, inp=final, out=[k.replace("image.", "image_") for k in final])

    # LANG
    language = fnmatch.filter(tree.keys(), "*language*")
    tree = rekey(tree, inp=language, out=[k.replace("language.embedding", "language_instruction") for k in language])

    # PROPRIO
    proprio = fnmatch.filter(tree.keys(), "*proprio.*")
    tree = rekey(tree, proprio, out=[k.replace("proprio.", "proprio_") for k in proprio])
    # drop all proprio_[gripper|joints|position]
    # noprop = fnmatch.filter(tree.keys(), "*proprio_*")
    # noprop = [k for k in noprop if "single" not in k]
    # tree = drop(tree, noprop)

    tree = unflat(tree)
    return tree


def _infer_observation_mappings(tree: dict) -> tuple[dict, dict, dict, dict] | None:
    raise DeprecatedError("jul 1 2026")
    obs = tree.get("observation", {})
    image_keys = set(obs.get("image", {}))
    depth_keys = set(obs.get("depth", {}))

    proprio = {k: v.shape[-1] for k, v in spec(obs.get("proprio", {}), simple=False).items()}
    # proprio_keys, proprio_dims = zip(*proprio.items())
    return image_keys, depth_keys, proprio


def get_task_ids(ds, lengths):
    global_length = sum(l for l in lengths.values())

    offsets = {}  # quadratic but who cares
    for eid in lengths:
        eids = [x for x in lengths if x < eid]
        offsets[str(eid)] = sum(lengths[str(i)] for i in eids)

    # offsets = { int(eid): sum(lengths[str(i)] for i in range(int(eid))) for eid in lengths.keys() }
    def step2global_step(eid, id):
        return offsets[str(int(eid))] + id

    def sample_goal(x, rng):
        id = x["info"]["id"]
        # pprint((len(lengths),id))
        # pprint(id['episode_id']>=len(lengths))
        l = lengths[str(int(id["episode_id"]))]
        x["info"]["length"] = l
        x["info"]["global_length"] = global_length
        goal_id = jax.device_get(jax.random.randint(rng, (), 0, l - id["step_id"]))
        id["goal_rel_id"] = goal_id
        id["goal_abs_id"] = id["step_id"] + id["goal_rel_id"]
        id["goal_global_id"] = id["global_id"] + goal_id
        x["info"]["id"] = id
        return x

    def add_global_id(x):
        # handle if info.id is missing
        id = x.get("info", {}).get("id", {})
        id = id if id else {"episode_id": x.get("episode_id"), "step_id": x.get("step_id")}
        id["global_id"] = step2global_step(id["episode_id"], id["step_id"])
        x["info"] = x.get("info", {}) | {"id": id}
        return x

    ds = ds.map(add_global_id).random_map(lambda x, r: sample_goal(x, rng=to_jax_key(r)))
    return ds


def get_task_goals(ds):
    _ds = ds  # pin _ds so we can index into it without closure issues

    # search the ds for the global goal id and get the observation
    def make_goal(x):
        gid = int(x["info"]["id"]["goal_global_id"])
        task = {"task": _ds[gid]["observation"]}
        return x | task

    ds = ds.map(make_goal)
    return ds


def hwc2chw(img):
    return jnp.transpose(img, (2, 0, 1))  # HWC -> CHW


class RandomAspect(augmax.GeometricTransformation):
    """Apply independent x/y aspect scaling without changing output size."""

    def __init__(
        self,
        x_range: tuple[float, float] = (1.0, 1.0),
        y_range: tuple[float, float] | None = None,
        p: float = 0.5,
    ):
        super().__init__()
        self.x_range = x_range
        self.y_range = x_range if y_range is None else y_range
        self.probability = p

        for lo, hi in (self.x_range, self.y_range):
            if lo <= 0 or hi <= 0:
                raise ValueError("Aspect ranges must be positive.")
            if lo > hi:
                raise ValueError("Aspect range lower bound must be <= upper bound.")

    def _sample_aspect(self, rng, bounds: tuple[float, float]):
        lo, hi = bounds
        lo = jnp.log(jnp.asarray(lo, dtype=jnp.float32))
        hi = jnp.log(jnp.asarray(hi, dtype=jnp.float32))
        return jnp.exp(jax.random.uniform(rng, (), minval=lo, maxval=hi))

    def transform_coordinates(self, rng: jnp.ndarray, coordinates, invert=False):
        k_apply, kx, ky = jax.random.split(rng, 3)
        do = jax.random.bernoulli(k_apply, self.probability)

        sx = jnp.where(do, self._sample_aspect(kx, self.x_range), 1.0)
        sy = jnp.where(do, self._sample_aspect(ky, self.y_range), 1.0)

        if not invert:
            sy, sx = 1.0 / sy, 1.0 / sx

        transform = jnp.array(
            [
                [sy, 0, 0],
                [0, sx, 0],
                [0, 0, 1],
            ]
        )
        coordinates.push_transform(transform)


class FloatToByte(augmax.ByteToFloat):
    """Inverse of ByteToFloat: float [0, 1] -> uint8 [0, 255]."""

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        return super().pixelwise(rng, pixel, invert=not invert)


def get_frame_transform(
    config: builders.GrainDatasetConfig,
    tfconfig: TransformConfig,
    *,
    imaug: bool = True,
    rotate: bool = True,
) -> Callable:
    re = tfconfig.resize_frames_to

    chain_ops: list = []
    if re is not None:
        if isinstance(re, tuple):
            h, w = re
            chain_ops.append(augmax.Resize(width=w, height=h))
        else:
            chain_ops.append(augmax.Resize(re))
    if imaug:
        chain_ops += [
            augmax.ChannelShuffle(p=0.5),
            # color ops require float32 [0, 1]; model normalization expects 0-255,
            # so FloatToByte closes the float region
            augmax.ByteToFloat(),
            # augmax halves brightness/contrast ranges internally: effective
            # brightness ±0.4, contrast slant 0.6x-1.6x. sized to cover the
            # sim/real gap (sim ~50/255 brighter, ~2x contrast on exo views)
            augmax.RandomBrightness((-0.8, 0.8), p=0.5),
            augmax.RandomContrast((-0.6, 0.6), p=0.5),
            augmax.RandomGamma((0.7, 1.5), p=0.5),
            # per-channel gamma: mild WB/color-cast variation; p<0.5 to limit aug stacking
            augmax.RandomChannelGamma((0.8, 1.25), p=0.3),
            # augmax 0.4.1: zero strengths misalign ColorJitter's rng keys and its
            # saturation branch is a no-op; keep all > 0, hue is the useful part
            augmax.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            augmax.Blur(size=3, p=0.3),  # runs at post-Resize resolution
            FloatToByte(),
        ]
        # RandomAspect(x_range=(0.9, 1.1), y_range=(0.9, 1.1), p=0.5),
        # augmax.RandomGrayscale(p= 0.5),
        # augmax.ChannelDrop(),
        # augmax.Warp(strength= 5, coarseness= 32),
        # augmax.Normalize(),
        # augmax.Solarization(),
    if rotate:
        chain_ops.append(augmax.Rotate((-15, 15), p=0.3))
    chain = augmax.Chain(*chain_ops)

    def augment(rng, im, f):
        # im: (*X, H, W, C) -- flatten every leading dim (batch/window/view/...),
        # augment each frame, then restore. -3: guards H/W changing (e.g. Resize).
        *lead, h, w, c = im.shape
        n = int(np.prod(lead)) if lead else 1
        flat = im.reshape(n, h, w, c)
        out = jax.vmap(f)(jax.random.split(rng, n), flat)  # (n, H', W', C)
        return out.reshape(*lead, *out.shape[-3:])

    def parallelize_all_keys(rng, batch, f):
        img = batch["observation"]["image"]
        if isinstance(img, dict):  # legacy named-camera dict: augment each view
            for k in img:
                rng, sub = jax.random.split(rng)
                img[k] = augment(sub, img[k], f)
        else:  # stacked (..., V, H, W, C): one tensor; augment() flattens the V axis
            batch["observation"]["image"] = augment(rng, img, f)
        return batch

    frame_transform_aug = partial(parallelize_all_keys, f=chain)
    return frame_transform_aug


def drop_str(x: dict):
    return drop_fn(x, lambda k, v: isinstance(v, str))


def add_horizon_mask(x: dict) -> dict:
    """Add mask.horizon with shape (W, H)."""
    sid = np.asarray(x["info"]["id"]["step"])
    len = np.asarray(x["info"]["len"])
    action = np.asarray(x["act"]["base"])

    # Expect (*shape)
    H, A = action.shape
    h = np.arange(H, dtype=sid.dtype)
    x.setdefault("mask", {})["horizon"] = (sid[..., None] + h < len[..., None]).astype(np.bool_)
    return x


def add_mask(x: dict, train=True):
    x = add_horizon_mask(x) if train else x

    # flag = x["info"]["id"]["episode"] % 2  # 95% of data
    # merge — embody_transform may have already written mask.act
    x.setdefault("mask", {}).update(
        {
            "action_head_masks": x["action_head_masks"],
            # "action_pad_mask": x["action_pad_mask"],
            "timestep_pad_mask": np.ones_like(x["observation"]["timestep"]).astype(np.bool_),
            # "only_adjustment": ~flag.astype(jnp.bool_),
        }
    )

    # bwd compatibility
    x["observation"]["timestep_pad_mask"] = x["mask"]["timestep_pad_mask"]
    return x
