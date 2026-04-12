"""Debug VizCallback on a dummy mixed robot/human batch.

Usage:
    uv run python scripts/debug/debug_viz_callback.py
    uv run python scripts/debug/debug_viz_callback.py --save-gif /tmp/viz.gif
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich import print
import tyro

from crossformer.data.grain.metadata import ArrayStatistics, normalize_tree
from crossformer.embody import DOF, HUMAN_SINGLE, SINGLE
from crossformer.utils.callbacks.viz import VizCallback
from crossformer.utils.slot import split_by_bodypart
from crossformer.utils.spec import spec
from crossformer.utils.tree.core import flat, unflat

EMBODIMENTS = (SINGLE, HUMAN_SINGLE)
PART_DOF_IDS = {p.name: np.array(p.dof_ids) for e in EMBODIMENTS for p in e.parts}
NICKNAMES = {"arm_7dof": "joints", "cart_pos": "position", "cart_ori": "orientation", "gripper": "gripper"}


def nickname(d: dict) -> dict:
    return {NICKNAMES.get(k, k): v for k, v in d.items()}


def fake_stats(tree: dict) -> dict:
    def _one(arr: np.ndarray) -> ArrayStatistics:
        d = arr.shape[-1]
        return ArrayStatistics(
            mean=np.full(d, 0.5, dtype=np.float32),
            std=np.full(d, 2.0, dtype=np.float32),
            maximum=np.ones(d, dtype=np.float32),
            minimum=-np.ones(d, dtype=np.float32),
            mask=np.ones(d, dtype=bool),
        )

    return {k: _one(v) for k, v in tree.items()}


JOINT_IDS = tuple(DOF[f"j{i}"] for i in range(7))
POS_IDS = (DOF["ee_x"], DOF["ee_y"], DOF["ee_z"])


@dataclass
class Config:
    batch_size: int = 6
    window: int = 1
    horizon: int = 50
    flow_steps: int = 50
    save_gif: Path | None = None
    seed: int = 0


def fake_fk(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 1:
        q = q[None]
    xyz = np.zeros((q.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = q[:, 0] + 0.25 * q[:, 3]
    xyz[:, 1] = q[:, 1] - 0.20 * q[:, 4]
    xyz[:, 2] = q[:, 2] + 0.15 * q[:, 5] - 0.10 * q[:, 6]
    return xyz


def make_dummy_batch(cfg: Config) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    b, w, h, f = cfg.batch_size, cfg.window, cfg.horizon, cfg.flow_steps
    max_a = SINGLE.action_dim

    act_base = np.zeros((b, w, h, max_a), dtype=np.float32)
    act_id = np.zeros((b, max_a), dtype=np.int32)
    mask_single = np.zeros((b, 1), dtype=np.bool_)
    mask_human = np.zeros((b, 1), dtype=np.bool_)

    half = b // 2
    for bi in range(b):
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)
        phase = 0.3 * bi
        if bi < half:
            mask_single[bi, 0] = True
            act_id[bi, :10] = np.array([*JOINT_IDS, *POS_IDS], dtype=np.int32)
            joints = np.stack([np.sin(t * 2.0 + phase + 0.2 * j) for j in range(7)], axis=-1).astype(np.float32)
            xyz = fake_fk(joints)
            act_base[bi, 0, :, :7] = joints
            act_base[bi, 0, :, 7:10] = xyz
        else:
            mask_human[bi, 0] = True
            act_id[bi, :3] = np.array(POS_IDS, dtype=np.int32)
            xyz = np.stack(
                [
                    0.8 * np.cos(t * 2.4 + phase),
                    0.6 * np.sin(t * 1.7 + phase),
                    0.4 * np.cos(t * 3.1 + 0.5 * phase),
                ],
                axis=-1,
            ).astype(np.float32)
            act_base[bi, 0, :, :3] = xyz

    noise = rng.standard_normal((f, b, w, h, max_a)).astype(np.float32)
    flow = np.empty_like(noise)
    alpha = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None, None, None, None]
    flow[:] = (1.0 - alpha) * noise + alpha * act_base[None]

    mask_bodypart = {name: np.isin(act_id, ids).any(axis=-1, keepdims=True) for name, ids in PART_DOF_IDS.items()}
    batch = unflat(
        {
            "act.base": act_base,
            "act.id": act_id,
            "mask.embodiment.single": mask_single,
            "mask.embodiment.human_single": mask_human,
            **{f"mask.bodypart.{k}": v for k, v in mask_bodypart.items()},
        }
    )
    return batch, flow


def main(cfg: Config) -> None:
    batch, flow = make_dummy_batch(cfg)
    print(spec(batch))
    print(spec(flow))

    f = flat(batch)
    act_id = f["act.id"]
    actions = nickname(split_by_bodypart(f["act.base"], act_id, EMBODIMENTS))
    pred = nickname(split_by_bodypart(flow, act_id[None], EMBODIMENTS))

    stats = fake_stats(actions)
    actions = normalize_tree(actions, stats, inv=True)
    pred = normalize_tree(pred, stats, inv=True)

    f.pop("act.base")
    for k, v in actions.items():
        f[f"actions.{k}"] = v
    for k, v in pred.items():
        f[f"pred.{k}"] = v
    batch = unflat(f)
    print(spec(batch))

    cb = VizCallback()
    cb._fk_fn = fake_fk
    frames = cb._call_v2(batch)

    print("frames:", frames.shape, frames.dtype)

    if cfg.save_gif is not None:
        out = cb.save(frames, cfg.save_gif)
        print("saved:", out)


if __name__ == "__main__":
    main(tyro.cli(Config))
