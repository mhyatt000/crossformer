"""XFlow training with image / proprio dropout experiments.

Mirrors scripts/train/xflow.py exactly; toggles experiments via flags:

    uv run scripts/debug/dropout_xflow.py --image-dropout
    uv run scripts/debug/dropout_xflow.py --proprio-dropout
    uv run scripts/debug/dropout_xflow.py --image-dropout --proprio-dropout

Image dropout: zeros random spatial blocks per (sample, time, image_key).
Proprio dropout: per-batch coin flip zeros all proprio (lowdim) keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro

from scripts.train import xflow
from scripts.train.xflow import Config as XFlowConfig

# -- dropout transforms -------------------------------------------------------


def _image_block_dropout(img: np.ndarray, rng: np.random.Generator, block_frac: float, n_blocks: int) -> np.ndarray:
    """Zero `n_blocks` random rectangular patches of fractional size `block_frac` per sample/time."""
    arr = np.asarray(img)
    if arr.ndim < 3:
        return arr
    *lead, h, w, _c = arr.shape
    bh = max(1, int(h * block_frac))
    bw = max(1, int(w * block_frac))
    out = arr.copy()
    flat = out.reshape(-1, h, w, out.shape[-1])
    for i in range(flat.shape[0]):
        for _ in range(n_blocks):
            y = int(rng.integers(0, max(1, h - bh + 1)))
            x = int(rng.integers(0, max(1, w - bw + 1)))
            flat[i, y : y + bh, x : x + bw, :] = 0
    return flat.reshape(*lead, h, w, out.shape[-1])


def make_normalize_obs(orig_fn, *, image_dropout: bool, proprio_dropout: bool, cfg: Config):
    image_pats = tuple(cfg.image_keys)
    rng = np.random.default_rng(cfg.dropout_seed)

    def wrapped(obs, obs_keys):
        out = orig_fn(obs, obs_keys)
        if image_dropout:
            for k in list(out):
                if any(pat in k for pat in ("image", "depth")) or k in image_pats:
                    out[k] = _image_block_dropout(out[k], rng, cfg.image_block_frac, cfg.image_n_blocks)
        if proprio_dropout and rng.random() < cfg.proprio_drop_prob:
            for k in obs_keys:
                if k in out:
                    out[k] = np.zeros_like(out[k])
        return out

    return wrapped


# -- config -------------------------------------------------------------------


@dataclass
class Config(XFlowConfig):
    image_dropout: bool = False
    proprio_dropout: bool = False
    image_block_frac: float = 0.25  # fraction of H/W per masked block
    image_n_blocks: int = 2  # number of blocks per (sample, time, key)
    proprio_drop_prob: float = 0.5  # per-batch prob to zero all proprio keys
    dropout_seed: int = 7


def main(cfg: Config):
    if not (cfg.image_dropout or cfg.proprio_dropout):
        print("[warn] neither --image-dropout nor --proprio-dropout set; running plain xflow")
    xflow.normalize_obs = make_normalize_obs(
        xflow.normalize_obs,
        image_dropout=cfg.image_dropout,
        proprio_dropout=cfg.proprio_dropout,
        cfg=cfg,
    )
    xflow.main(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
