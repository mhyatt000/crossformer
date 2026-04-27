from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    p_proprio_drop: float = (
        0.0  # probability of dropping all proprio for some samples (zeroing pixels and flipping pmd)
    )
    p_im_drop: float = (
        0.0  # probability of dropping entire camera views for some samples (zeroing pixels and flipping pmd)
    )
    p_im_shuffle: float = 0.0  # probability of shuffling camera views across view slots for some samples
    p_im_occlude: float = 0.0
    occlude_size: list[float] = (0.05, 0.5)  # min and max fraction of image area to occlude

    def aug_im_occlude(self, step: dict, rng) -> dict:
        return patch_occlude(
            step,
            rng,
            self.p_occlude,
            min_frac=self.occlude_size[0],
            max_frac=self.occlude_size[1],
        )

    def aug_im_view_drop(self, step: dict, rng) -> dict:
        return image_view_drop(step, rng, self.image_view_drop_prob)

    def aug_im_key_shuffle(self, step: dict, rng) -> dict:
        return image_key_shuffle(step, rng, self.image_key_shuffle_prob)

    def aug_proprio_drop(self, step: dict, rng) -> dict:
        return proprio_sample_drop(step, rng, self.proprio_drop_prob)


def patch_occlude(step: dict, rng, prob: float, min_frac: float = 0.05, max_frac: float = 0.5) -> dict:
    """Randomly zero out a rectangular region in each image view.

    For each (sample, timestep, view), with probability `prob` a rectangle of
    random area (uniform in [min_frac, max_frac] of image area) and aspect
    ratio (uniform in [0.5, 2.0]) is zeroed. Views are occluded independently.

    Does not modify pad_mask_dict: an occluded image is still "present", just
    corrupted. Use image_view_drop for whole-view removal.
    """
    images = step["observation"]["image"]

    def patch_one(a):
        squeezed = a.ndim == 4
        if squeezed:
            a = a[:, None]  # add time dim
        b, t, h, w = a.shape[:4]  # potentially assert dim = 5
        hit = rng.random((b, t)) < prob
        frac = rng.uniform(min_frac, max_frac, size=(b, t))
        aspect = rng.uniform(0.5, 2.0, size=(b, t))
        ph = np.clip(np.sqrt(frac * h * w * aspect), 1, h).astype(int)  # patch height
        pw = np.clip(np.sqrt(frac * h * w / aspect), 1, w).astype(int)  # patch width
        y0 = rng.integers(0, np.maximum(h - ph, 1))  # top edge of patch
        x0 = rng.integers(0, np.maximum(w - pw, 1))
        ys = np.arange(h)  # row indexes
        xs = np.arange(w)  # col indexes
        y_in = (ys >= y0[..., None]) & (ys < (y0 + ph)[..., None])
        x_in = (xs >= x0[..., None]) & (xs < (x0 + pw)[..., None])
        patch = hit[..., None, None] & y_in[..., None] & x_in[..., None, :]
        out = np.where(patch[..., None], 0, a)
        return out[:, 0] if squeezed else out

    new_images = {v: patch_one(images[v]) for v in images}
    return {**step, "observation": {**step["observation"], "image": new_images}}


def image_view_drop(step: dict, rng, prob: float) -> dict:
    """
    Randomly drop entire image views for some batch samples.

    For each (sample, view) with probability 'prob' across
    all timesteps, the view is zeroed across all timesteps
    and pmd entry flipped to False.
    Views drop independently.
    """
    # raise RuntimeError("check keys for image_view_drop")
    images = step["observation"]["image"]
    pmd = step["observation"]["pad_mask_dict"]
    pmd_img = pmd["image"]
    views = list(images.keys())
    b = images[views[0]].shape[0]
    # given view independence, small chance for all views to drop
    # can add rescue if needed
    drop = {v: rng.random(b) < prob for v in views}

    def zero_view(img, d):
        return np.where(d.reshape((b,) + (1,) * (img.ndim - 1)), 0, img)

    new_images = {v: zero_view(images[v], drop[v]) for v in views}

    def new_mask(v):
        existing = np.asarray(pmd_img[v], dtype=bool)
        d = drop[v].reshape((b,) + (1,) * (existing.ndim - 1))
        return existing & ~d

    new_pmd_img = {v: new_mask(v) for v in views}

    return {
        **step,
        "observation": {
            **step["observation"],
            "image": new_images,
            "pad_mask_dict": {**pmd, "image": new_pmd_img},
        },
    }


def image_key_shuffle(step: dict, rng, prob: float) -> dict:
    """
    Randomly shuffle camera views across camera slots for some samples.

    For each sample, with probability 'prob', permutes which camera appears
    in each image_* slot. pad_mask_dict travels with their view.
    """
    images = step["observation"]["image"]
    pmd = step["observation"]["pad_mask_dict"]["image"]
    views = list(images.keys())
    b, K = images[views[0]].shape[0], len(views)

    shuffled = rng.random(b) < prob
    random_perm = np.argsort(rng.random((b, K)), axis=1)  # indirect sorting
    perm = np.where(shuffled[:, None], random_perm, np.arange(K))

    imgs = np.stack([images[v] for v in views], axis=1)
    masks = np.stack([np.asarray(pmd[v]) for v in views], axis=1)

    bi = np.arange(b)[:, None]  # batch indices
    new_imgs = imgs[bi, perm]
    new_masks = masks[bi, perm]

    new_images = {v: new_imgs[:, i] for i, v in enumerate(views)}
    new_pmd = {v: new_masks[:, i] for i, v in enumerate(views)}

    return {
        **step,
        "observation": {
            **step["observation"],
            "image": new_images,
            "pad_mask_dict": {**step["observation"]["pad_mask_dict"], "image": new_pmd},
        },
    }


def proprio_sample_drop(step: dict, rng, prob: float) -> dict:
    """
    Randomly zero all proprio for some samples (mimics total proprio sensor loss).

    One Bernoulli hit per sample with probability 'prob'. On hit, every
    proprio subkey is zeroed and every proprio pmd entry is flipped to False
    across all timesteps. Subkeys move together - per-subkey independence
    is handled later by proprio_token_drop.
    """
    proprio = step["observation"]["proprio"]
    pmd = step["observation"]["pad_mask_dict"]
    pmd_p = pmd["proprio"]
    keys = list(proprio.keys())
    b = proprio[keys[0]].shape[0]

    drop = rng.random(b) < prob

    def zero(x):
        return np.where(drop.reshape((b,) + (1,) * (x.ndim - 1)), 0, x)

    new_proprio = {k: zero(proprio[k]) for k in keys}

    def new_mask(k):
        existing = np.asarray(pmd_p[k], dtype=bool)
        d = drop.reshape((b,) + (1,) * (existing.ndim - 1))
        return existing & ~d

    new_pmd_p = {k: new_mask(k) for k in keys}

    return {
        **step,
        "observation": {
            **step["observation"],
            "proprio": new_proprio,
            "pad_mask_dict": {**pmd, "proprio": new_pmd_p},
        },
    }
