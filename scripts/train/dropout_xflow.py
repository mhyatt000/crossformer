"""XFlow training with composable input-dropout on xgym_sweep.

Four dropout methods, each independently toggleable via its probability flag
(set to 0.0 to disable). A single run can use any combination, including none.

Methods
-------
1. Patch occlusion         (--patch-prob)
     Black out up to --patch-count random rectangles per (sample, time, view).
2. Per-sample image-view drop (--view-drop-prob)
     Independently Bernoulli-zero whole image views per (sample, time, key).
     Guarantees at least one surviving view.
3. Per-sample proprio drop (--proprio-sample-drop-prob)
     Per sample, zero out ALL proprio_* tensors together.
4. Per-sample proprio-token drop (--proprio-token-drop-prob)
     Per sample, drop one randomly chosen proprio_* key (token).

Use scripts/train/sweep_dropout.py to iterate all 16 on/off combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from crossformer.cn.base import default
from crossformer.cn.dataset.mix import DataSource
from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.utils.callbacks.rast import RastConfig
from scripts.train import xflow
from scripts.train.xflow import _resolve_version
from scripts.train.xflow import Config as XFlowConfig

_EXTR = Path("~/data/extr/cam").expanduser()
_CAMS = (_EXTR / "over/HT.npz", _EXTR / "side/HT.npz", _EXTR / "low/HT.npz")

# -- dropout primitives -------------------------------------------------------


def _image_keys(obs: dict) -> list[str]:
    return [k for k in obs if k.startswith("image_")]


def _proprio_keys(obs: dict) -> list[str]:
    return [k for k in obs if k.startswith("proprio_")]


def patch_occlude(obs: dict, rng: np.random.Generator, prob: float, count: int, max_frac: float) -> dict:
    """Black out up to `count` random rectangles per (sample, time, view).

    Each rectangle is drawn independently with probability `prob`. Rectangle
    side lengths are uniform in [1, max_frac*H] x [1, max_frac*W].
    """
    if prob <= 0.0 or count <= 0:
        return obs
    out = dict(obs)
    for k in _image_keys(obs):
        img = np.asarray(out[k]).copy()
        if img.ndim < 4:
            continue
        b, t, h, w = img.shape[:4]
        for _ in range(count):
            draw = rng.random((b, t)) < prob
            if not draw.any():
                continue
            for bi, ti in np.argwhere(draw):
                ph = int(rng.integers(1, max(2, int(max_frac * h))))
                pw = int(rng.integers(1, max(2, int(max_frac * w))))
                y0 = int(rng.integers(0, h - ph + 1))
                x0 = int(rng.integers(0, w - pw + 1))
                img[bi, ti, y0 : y0 + ph, x0 : x0 + pw] = 0
        out[k] = img
    return out


def image_view_drop(
    obs: dict,
    rng: np.random.Generator,
    prob: float,
    always_keep_key: str | None = None,
) -> dict:
    """Per-sample, per-time, per-key Bernoulli drop of whole image views."""
    image_keys = _image_keys(obs)
    if prob <= 0.0 or not image_keys:
        return obs
    pmd = obs.get("pad_mask_dict", {})
    b, t = obs[image_keys[0]].shape[:2]
    k_n = len(image_keys)
    existing = np.stack(
        [np.asarray(pmd[k], dtype=bool) if k in pmd else np.ones((b, t), bool) for k in image_keys],
        axis=-1,
    )
    drop = rng.random((b, t, k_n)) < prob
    if always_keep_key is not None:
        if always_keep_key not in image_keys:
            raise ValueError(f"always_keep_key={always_keep_key!r} not in {image_keys}")
        drop[..., image_keys.index(always_keep_key)] = False
    else:
        kept = existing & ~drop
        none_kept = ~kept.any(axis=-1) & existing.any(axis=-1)
        if none_kept.any():
            for bi, ti in np.argwhere(none_kept):
                valid = np.flatnonzero(existing[bi, ti])
                drop[bi, ti, int(rng.choice(valid))] = False
    out = dict(obs)
    new_pmd = dict(pmd)
    for i, k in enumerate(image_keys):
        d = drop[..., i]
        if not d.any():
            continue
        arr = np.asarray(out[k]).copy()
        arr[d] = 0
        out[k] = arr
        if k in new_pmd:
            new_pmd[k] = np.asarray(new_pmd[k], dtype=bool) & ~d
        else:
            m = np.ones((b, t), dtype=bool)
            m[d] = False
            new_pmd[k] = m
    out["pad_mask_dict"] = new_pmd
    return out


def image_key_shuffle(obs: dict, rng: np.random.Generator, prob: float) -> dict:
    """Per sample, permute which view occupies each image_* key slot.

    Image contents are untouched; only the mapping from view -> key is shuffled.
    pad_mask_dict entries are permuted to track their associated view.
    """
    ikeys = _image_keys(obs)
    if prob <= 0.0 or len(ikeys) < 2:
        return obs
    b = obs[ikeys[0]].shape[0]
    apply = rng.random((b,)) < prob
    if not apply.any():
        return obs
    out = dict(obs)
    pmd = dict(obs.get("pad_mask_dict", {}))
    srcs = {k: np.asarray(out[k]).copy() for k in ikeys}
    msrcs = {k: (np.asarray(pmd[k], dtype=bool).copy() if k in pmd else None) for k in ikeys}
    for bi in np.flatnonzero(apply):
        perm = rng.permutation(len(ikeys))
        for i, k in enumerate(ikeys):
            src_k = ikeys[int(perm[i])]
            out[k][bi] = srcs[src_k][bi]
            if msrcs[src_k] is not None or msrcs[k] is not None:
                if k not in pmd:
                    pmd[k] = np.ones(out[k].shape[:2], bool)
                pmd[k][bi] = msrcs[src_k][bi] if msrcs[src_k] is not None else np.ones(out[k].shape[1], bool)
    out["pad_mask_dict"] = pmd
    return out


def proprio_sample_drop(obs: dict, rng: np.random.Generator, prob: float) -> dict:
    """Per sample, Bernoulli zero out ALL proprio_* tensors together."""
    pkeys = _proprio_keys(obs)
    if prob <= 0.0 or not pkeys:
        return obs
    b = obs[pkeys[0]].shape[0]
    drop = rng.random((b,)) < prob
    if not drop.any():
        return obs
    out = dict(obs)
    pmd = dict(obs.get("pad_mask_dict", {}))
    for k in pkeys:
        arr = np.asarray(out[k]).copy()
        arr[drop] = 0
        out[k] = arr
        if k in pmd:
            m = np.asarray(pmd[k], dtype=bool).copy()
            m[drop] = False
            pmd[k] = m
        else:
            t = arr.shape[1] if arr.ndim >= 2 else 1
            m = np.ones((b, t), dtype=bool)
            m[drop] = False
            pmd[k] = m
    out["pad_mask_dict"] = pmd
    return out


def proprio_token_drop(obs: dict, rng: np.random.Generator, prob: float) -> dict:
    """Per sample, Bernoulli drop exactly one randomly chosen proprio_* key."""
    pkeys = _proprio_keys(obs)
    if prob <= 0.0 or len(pkeys) == 0:
        return obs
    b = obs[pkeys[0]].shape[0]
    drop = rng.random((b,)) < prob
    if not drop.any():
        return obs
    choice = rng.integers(0, len(pkeys), size=b)
    out = dict(obs)
    pmd = dict(obs.get("pad_mask_dict", {}))
    # materialize arrays we may modify
    mats = {k: np.asarray(out[k]).copy() for k in pkeys}
    masks = {
        k: (np.asarray(pmd[k], dtype=bool).copy() if k in pmd else np.ones(mats[k].shape[:2], bool)) for k in pkeys
    }
    for bi in np.flatnonzero(drop):
        k = pkeys[int(choice[bi])]
        mats[k][bi] = 0
        masks[k][bi] = False
    for k in pkeys:
        out[k] = mats[k]
        pmd[k] = masks[k]
    out["pad_mask_dict"] = pmd
    return out


# -- config -------------------------------------------------------------------


@dataclass
class Config(XFlowConfig):
    mix: str = "xgym_sweep"
    xgym_sweep_single_version: str = "0.5.5"
    sweep_mano_version: str | None = "0.0.2"

    # (1) patch occlusion
    patch_prob: float = 0.0
    patch_count: int = 3
    patch_max_frac: float = 0.25

    # (2) per-sample per-view image drop
    view_drop_prob: float = 0.0
    always_keep_key: str | None = None

    # (2b) per-sample image-key shuffle (permute view -> key mapping)
    image_key_shuffle_prob: float = 0.0

    # (3) per-sample full-proprio drop
    proprio_sample_drop_prob: float = 0.0

    # (4) per-sample drop-one-proprio-token
    proprio_token_drop_prob: float = 0.0

    dropout_seed: int = 7

    rast: RastConfig = default(RastConfig(cams=_CAMS))


def _pin(name: str, version: str) -> None:
    ds = DataSource.REGISTRY[name]
    ds.version = version
    ds.builder = ArrayRecordBuilder(name=ds.name, version=version, branch=ds.branch)


def main(cfg: Config):
    _pin("xgym_sweep_single", cfg.xgym_sweep_single_version)
    mano_version = cfg.sweep_mano_version or _resolve_version(None, dataset_name="sweep_mano")
    _pin("sweep_mano", mano_version)
    print(f"  xgym_sweep_single: {cfg.xgym_sweep_single_version}")
    print(f"  sweep_mano:        {mano_version}")
    print(
        f"  dropout: patch={cfg.patch_prob} view={cfg.view_drop_prob} "
        f"prop_sample={cfg.proprio_sample_drop_prob} prop_token={cfg.proprio_token_drop_prob}"
    )

    rng = np.random.default_rng(cfg.dropout_seed)
    orig = xflow.flatten_obs

    def flatten_with_dropout(obs, obs_keys):
        obs = patch_occlude(obs, rng, cfg.patch_prob, cfg.patch_count, cfg.patch_max_frac)
        obs = image_view_drop(obs, rng, cfg.view_drop_prob, cfg.always_keep_key)
        obs = image_key_shuffle(obs, rng, cfg.image_key_shuffle_prob)
        obs = proprio_sample_drop(obs, rng, cfg.proprio_sample_drop_prob)
        obs = proprio_token_drop(obs, rng, cfg.proprio_token_drop_prob)
        return orig(obs, obs_keys)

    xflow.flatten_obs = flatten_with_dropout
    xflow.main(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
