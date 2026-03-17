from __future__ import annotations

import numpy as np


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (N,D), got {x.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = x / np.maximum(std, 1e-6)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    basis = vh[:2].T
    z = x @ basis
    if z.shape[1] < 2:
        z = np.pad(z, [(0, 0), (0, 2 - z.shape[1])])
    return z


def direction_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    da = np.diff(a, axis=0)
    db = np.diff(b, axis=0)
    n = min(da.shape[0], db.shape[0])
    da = da[:n]
    db = db[:n]
    na = np.linalg.norm(da, axis=1)
    nb = np.linalg.norm(db, axis=1)
    denom = np.maximum(na * nb, 1e-8)
    cos = np.sum(da * db, axis=1) / denom
    return float(np.mean(cos))
