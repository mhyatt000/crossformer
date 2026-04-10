"""
Restructure raw trajectories into a unified format for training.
"""

from __future__ import annotations

import jax
import numpy as np

from crossformer.utils.jax_utils import str2jax


def _restructure_trajectory(
    step: dict,
    *,
    name: str,
    lang_key: str | None = None,
) -> dict:
    info = step["info"]
    if "id" in info:
        info = info | info["id"]  # flatten id into info for backward compatibility
    sid, eid = np.array(info["step"]).reshape(-1), np.array(info["episode"]).reshape(-1)
    step["info"]["id"] = {"step": sid, "episode": eid}  # patch
    step["observation"]["timestep"] = sid

    task = {}
    # PATCH 0.5.2 to 0.5.3
    # task[lang_key] = step[lang_key]  # simple
    if "pose" in step["observation"]["proprio"]:
        step["observation"]["proprio"].pop("pose")

    return {
        "observation": step["observation"],
        "task": task,
        "action": step["action"],  #  action,
        "dataset_name": str2jax(name),
        "info": {
            "dataset_name": str2jax(name),
            "id": {k.replace("_id", ""): np.array([v]).reshape(-1) for k, v in step.items() if "_id" in k},
        }
        | step.get("info", {}),
    }


def _restructure_step_mano(x: dict, *, name: str, lang_key: str) -> dict:
    task = {}
    task[lang_key] = x.pop(lang_key)  # simple
    x["task"] = task

    x["observation"]["timestep"] = x["info"]["id"]["step"]

    # k3ds: (H, 21, 4) → strip homogeneous coord → (H, 21, 3)
    # derive cart_pos from palm keypoint (index 0) before flatten
    k3ds = np.array(x["action"]["k3ds"])  # (H, 21, 4)
    k3ds = k3ds[..., :3]  # (H, 21, 3) drop homogeneous w
    x["action"]["position"] = k3ds[:, 0, :]  # (H, 3) palm = cart_pos
    x["action"]["k3ds"] = k3ds.reshape(k3ds.shape[0], -1)  # (H, 63)
    x["action"].pop("k3ds")

    x["observation"]["proprio"] = jax.tree.map(lambda y: y[0], x["action"])

    x = jax.tree.map(lambda y: np.array(y), x)  # ensure numpy arrays
    return x
