"""Standalone test harness for dropout primitives against xgym_sweep.

Defaults: mix=xgym_sweep, encoder=small-stem-8-film (film OFF), steps=10000.
Each dropout method is toggled by its probability flag; 0.0 disables it.

Usage (one method at a time, or stack any combination):
    uv run scripts/train/dropout.py --patch-prob 0.5
    uv run scripts/train/dropout.py --image-key-shuffle-prob 0.3
    uv run scripts/train/dropout.py --proprio-sample-drop-prob 0.2
    uv run scripts/train/dropout.py --proprio-token-drop-prob 0.3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import DataSource
from crossformer.cn.model_factory import Vision
from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.utils.callbacks.rast import RastConfig
from scripts.train import xflow
from scripts.train.xflow import _resolve_version
from scripts.train.xflow import Config as XFlowConfig

_EXTR = Path("~/data/extr/cam").expanduser()
_CAMS = (_EXTR / "over/HT.npz", _EXTR / "side/HT.npz", _EXTR / "low/HT.npz")


def _image_keys(obs: dict) -> list[str]:
    return [k for k in obs if k.startswith("image_")]


def _proprio_keys(obs: dict) -> list[str]:
    return [k for k in obs if k.startswith("proprio_")]


def _pin(name: str, version: str) -> None:
    ds = DataSource.REGISTRY[name]
    ds.version = version
    ds.builder = ArrayRecordBuilder(name=ds.name, version=version, branch=ds.branch)


def image_view_drop(obs, rng, prob):
    """
    Randomly drop entire image views for some batch sample.

    For each sample, independently decide whether to drop one
    randomly chosen camera. ALL timesteps of that camera view are zeroed
    for that sample.
    """
    keys = _image_keys(obs)
    b = obs[keys[0]].shape[0]
    hit = rng.random(b) < prob
    chosen = rng.integers(0, len(keys), size=b)
    drop = {k: hit & (chosen == i) for i, k in enumerate(keys)}
    pmd = obs.get("pad_mask_dict", {})
    imgs = {k: obs[k] for k in keys}
    masks = {k: np.asarray(pmd.get(k, np.ones(obs[k].shape[:2], bool)), bool) for k in keys}
    new_imgs = jax.tree.map(lambda a, d: np.where(d.reshape((b,) + (1,) * (a.ndim - 1)), 0, a), imgs, drop)
    new_masks = jax.tree.map(lambda m, d: m & ~d[:, None], masks, drop)
    return {**obs, **new_imgs, "pad_mask_dict": {**pmd, **new_masks}}


def patch_occlude(obs, rng, prob, min_frac=0.05, max_frac=0.5):
    """
    Randomly occlude rectangular patches within images.

    For each batch sample and timestep with probability 'prob',
    masks out a random rectangular region of the image. The patch
    size is sampled as a fraction of the total image area.
    """
    keys = _image_keys(obs)

    def patch_one(a):
        b, t, h, w = a.shape[:4]  # batch, time, height, width
        img_area = h * w
        hit = rng.random((b, t)) < prob  # which (sample, timestep) gets occluded
        # sample occlusion area as fraction of image then derive rectangle dimensions
        target_frac = rng.uniform(min_frac, max_frac, size=(b, t))  # fraction of image to be occluded
        target_area = target_frac * img_area
        aspect = rng.uniform(0.5, 2.0, size=(b, t))  # height / width ratio of rectanglev
        ph = np.clip(np.sqrt(target_area * aspect), 1, h).astype(int)  # patch height
        pw = np.clip(np.sqrt(target_area / aspect), 1, w).astype(int)  # patch width
        y0 = rng.integers(0, np.maximum(h - ph, 1))  # top edge of rectangle
        x0 = rng.integers(0, np.maximum(w - pw, 1))  # left edge of rectangle
        ar_h, ar_w = (
            np.arange(
                h,
            ),
            np.arange(w),
        )  # pixel coordinate grids
        y_in = (ar_h >= y0[..., None]) & (ar_h < (y0 + ph)[..., None])  # rows inside
        x_in = (ar_w >= x0[..., None]) & (ar_w < (x0 + pw)[..., None])  # cols inside
        rect = hit[..., None, None] & y_in[..., None] & x_in[..., None, :]  # final mask (b, t, h, w)
        return np.where(rect[..., None], 0, a)

    return {**obs, **jax.tree.map(patch_one, {k: obs[k] for k in keys})}


def proprio_sample_drop(obs: dict, rng: np.random.Generator, prob: float) -> dict:
    """
    Drop ALL proprio sensors for randoly selected batch samples.

    For each sample, flips a coin with probability 'prob'. if heads,
    zeros all out all proprioceptive keys for that entire sample.
    """
    proprio_keys = _proprio_keys(obs)
    batch_size = obs[proprio_keys[0]].shape[0]
    dropped = rng.random(batch_size) < prob
    out = dict(obs)
    pad_masks = dict(obs.get("pad_mask_dict", {}))

    for key in proprio_keys:
        values = np.asarray(out[key]).copy()
        values[dropped] = 0
        out[key] = values

        if key in pad_masks:
            mask = np.asarray(pad_masks[key], dtype=bool).copy()
            mask[dropped] = False
            pad_masks[key] = mask
        else:
            # Create mask so downstream knows it was dropped
            timesteps = values.shape[1] if values.ndim >= 2 else 1
            mask = np.ones((batch_size, timesteps), dtype=bool)
            mask[dropped] = False
            pad_masks[key] = mask
    out["pad_mask_dict"] = pad_masks
    return out


def proprio_token_drop(_obs: dict, rng: np.random.Generator, prob: float) -> dict:
    """
    Independently drop individual proprio tokens (timesteps) per sensor type.

    For each proprio sensor and each timestep within each sample,
    independently decides whether to drop it with probability 'prob'.
    """
    obs = jax.tree.map(lambda x: x.copy(), _obs)  # assignment dest is read only
    proprio_keys = _proprio_keys(obs)
    batch_size, window = obs[proprio_keys[0]].shape[:2]
    for key in proprio_keys:
        dropped = rng.random((batch_size, window)) < prob
        obs[key][dropped] = 0  # BWA
        obs["pad_mask_dict"][key][dropped] = False  # what isshape?

    return obs


def image_key_shuffle(_obs: dict, rng: np.random.Generator, prob: float) -> dict:
    """
    Randomly shuffle camera views across camera slots for some samples.

    For each sample with probability 'prob', permutes which camera appears in
    each image_* slot. pad_mask_dict entries travel with their view.
    """
    if prob <= 0.0:
        return _obs
    keys = _image_keys(_obs)
    b, K = _obs[keys[0]].shape[0], len(keys)

    shuffled = rng.random((b,)) < prob  # (B,)
    identity = np.broadcast_to(np.arange(K), (b, K))  # (B, K)
    random_perm = np.argsort(rng.random((b, K)), axis=1)  # (B, K)
    perm = np.where(shuffled[:, None], random_perm, identity)  # (B, K)

    bi = np.arange(b)[:, None]
    imgs = np.moveaxis(np.stack([_obs[k] for k in keys]), 0, 1)  # (B, K, T, H, W, C)
    pmd = _obs["pad_mask_dict"]
    masks = np.moveaxis(np.stack([pmd[k] for k in keys]), 0, 1)  # (B, K, T)
    gathered_imgs = np.moveaxis(imgs[bi, perm], 1, 0)  # (K, B, ...)
    gathered_masks = np.moveaxis(masks[bi, perm], 1, 0)  # (K, B, T)

    out, new_pmd = dict(_obs), dict(pmd)
    for i, k in enumerate(keys):
        out[k] = gathered_imgs[i]
        new_pmd[k] = gathered_masks[i]
    out["pad_mask_dict"] = new_pmd
    return out


def image_key_shuffle_vmap(_obs: dict, key: jax.Array, prob: float) -> dict:
    out = jax.tree.map(lambda x: x.copy(), _obs)  # assignment dest is read only
    keys = _image_keys(out)
    length = len(keys)
    pmd = out["pad_mask_dict"]

    # Stack all keys to vectorize
    imgs = jnp.stack([out[k] for k in keys], axis=1)  # (B, K, T, H, W, C)
    masks = jnp.stack([pmd[k] for k in keys], axis=1)  # (B, K, T)

    def shuffle_one(img_key, mask_key, subkey):  # subkey ->from jax.random.split
        perm = jax.random.permutation(subkey, length)
        idx = jnp.where(jax.random.uniform(subkey) < prob, perm, jnp.arange(keys))
        return img_key[idx], mask_key[idx]

    new_imgs, new_masks = jax.vmap(shuffle_one)(imgs, masks, jax.random.split(key, imgs.shape[0]))

    for i, k in enumerate(keys):
        out[k] = new_imgs[:, i]
        pmd[k] = new_masks[:, i]
    return out


# -- config + main ------------------------------------------------------------


@dataclass
class Config(XFlowConfig):
    mix: str = "xgym_sweep"
    xgym_sweep_single_version: str = "0.5.5"
    sweep_mano_version: str | None = None  # None = latest on disk
    steps: int = 10_000

    model: cn.ModelFactory = default(
        cn.ModelFactory(
            size=cn.Size.DUMMY,
            window=20,
            image_keys=(),
            proprio_keys=(),
            vision=Vision(use_film=False, encoder="small-stem-8-film"),
        )
    )
    rast: RastConfig = default(RastConfig(cams=_CAMS))

    # dropout probs (0.0 = off)
    patch_prob: float = 0.0
    patch_min_frac: float = 0.05
    patch_max_frac: float = 0.5

    view_drop_prob: float = 0.0
    image_key_shuffle_prob: float = 0.0

    proprio_sample_drop_prob: float = 0.0
    proprio_body_part_drop_prob: float = 0.0
    proprio_token_drop_prob: float = 0.0

    dropout_seed: int = 7


def _auto_name(cfg: Config) -> str:
    active = []
    if cfg.patch_prob > 0.0:
        active.append(f"patch{cfg.patch_prob:g}")
    if cfg.view_drop_prob > 0.0:
        active.append(f"view{cfg.view_drop_prob:g}")
    if cfg.image_key_shuffle_prob > 0.0:
        active.append(f"shuf{cfg.image_key_shuffle_prob:g}")
    if cfg.proprio_sample_drop_prob > 0.0:
        active.append(f"psamp{cfg.proprio_sample_drop_prob:g}")
    if cfg.proprio_body_part_drop_prob > 0.0:
        active.append(f"pbody{cfg.proprio_body_part_drop_prob:g}")
    if cfg.proprio_token_drop_prob > 0.0:
        active.append(f"ptok{cfg.proprio_token_drop_prob:g}")
    return "_".join(active) if active else "none"


def main(cfg: Config):
    if not cfg.name:
        cfg.name = _auto_name(cfg)
    cfg.wandb.group = "dropout_10k"

    _pin("xgym_sweep_single", cfg.xgym_sweep_single_version)
    mano_version = cfg.sweep_mano_version or _resolve_version(None, dataset_name="sweep_mano")
    _pin("sweep_mano", mano_version)
    print(f"  run name: {cfg.name}  group: {cfg.wandb.group}")
    print(f"  xgym_sweep_single: {cfg.xgym_sweep_single_version}")
    print(f"  sweep_mano:        {mano_version}")
    print(
        f"  dropout: patch={cfg.patch_prob} view={cfg.view_drop_prob} "
        f"key_shuf={cfg.image_key_shuffle_prob} prop_sample={cfg.proprio_sample_drop_prob} "
        f"prop_body_part={cfg.proprio_body_part_drop_prob} prop_token={cfg.proprio_token_drop_prob}"
    )

    rng = np.random.default_rng(cfg.dropout_seed)
    orig = xflow.flatten_obs

    def flatten_with_dropout(obs, obs_keys):
        if cfg.patch_prob > 0.0:
            obs = patch_occlude(obs, rng, cfg.patch_prob, cfg.patch_min_frac, cfg.patch_max_frac)
        if cfg.view_drop_prob > 0.0:
            obs = image_view_drop(obs, rng, cfg.view_drop_prob)
        if cfg.image_key_shuffle_prob > 0.0:
            obs = image_key_shuffle(obs, rng, cfg.image_key_shuffle_prob)
        if cfg.proprio_sample_drop_prob > 0.0:
            obs = proprio_sample_drop(obs, rng, cfg.proprio_sample_drop_prob)
        if cfg.proprio_body_part_drop_prob > 0.0:
            obs = proprio_body_part_drop(obs, rng, cfg.proprio_body_part_drop_prob)
        if cfg.proprio_token_drop_prob > 0.0:
            obs = proprio_token_drop(obs, rng, cfg.proprio_token_drop_prob)
        return orig(obs, obs_keys)

    xflow.flatten_obs = flatten_with_dropout
    xflow.main(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
