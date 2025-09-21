from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


def make_synthetic_trajectory(length: int, *, language: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    observation = {
        "img1": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "img2": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "img_wrist": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "state": {
            "cartesian": rng.normal(size=(length, 6)).astype(np.float32),
            "joints": rng.normal(size=(length, 7)).astype(np.float32),
            "gripper": rng.normal(size=(length, 1)).astype(np.float32),
        },
        "language_embedding": rng.normal(size=(length, 8)).astype(np.float32),
    }
    return {
        "observation": observation,
        "action": rng.normal(size=(length, 4)).astype(np.float32),
        "language_instruction": np.array([language] * length, dtype=object),
    }


def standardize_synthetic(traj: dict) -> dict:
    obs = dict(traj["observation"])
    state = obs.pop("state")
    for key, value in state.items():
        obs[f"state_{key}"] = value
    traj = dict(traj)
    traj["observation"] = obs
    return traj


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
