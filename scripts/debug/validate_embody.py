"""Validate embodiment action transform on real data.

Usage:
    uv run scripts/debug/validate_embody.py
    uv run scripts/debug/validate_embody.py --mix xgym_lift_single
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
import numpy as np
from rich import print
from rich.table import Table
import tyro

from crossformer import cn
from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.embody import build_action_norm_mask
from crossformer.data.grain.loader import GrainDataFactory
from crossformer.embody import DOF, MASK_ID

logging.basicConfig(level=logging.WARNING)

# reverse DOF lookup: id → name
ID_TO_NAME = {v: k for k, v in DOF.items()}


def norm_mask_from_ids(ids: np.ndarray, stats) -> np.ndarray:
    """Map per-key stats masks onto the embodied slot order."""
    out = np.zeros_like(ids, dtype=bool)
    stats_masks = {key: np.array(stat.mask).astype(bool).reshape(-1) for key, stat in stats.action.items()}
    key2mask = {
        "gripper": stats_masks.get("gripper", np.ones(1, dtype=bool)),
    }

    for i, did in enumerate(ids):
        if did == MASK_ID:
            continue
        name = ID_TO_NAME.get(int(did), "")
        if name in key2mask:
            out[i] = bool(key2mask[name][0])
            continue
        out[i] = True
    return out


@dataclass
class Config(cn.Train):
    mix: str = "xgym_sweep_single"
    n_batches: int = 3


def main(cfg: Config) -> None:
    cfg.data.mix = type(cfg.data.mix)[cfg.mix]
    arec = Arec.from_name(cfg.mix)

    mesh = Mesh(jax.devices(), axis_names="batch")
    shard = lambda b: multihost_utils.host_local_array_to_global_array(b, mesh, PartitionSpec("batch"))

    factory = GrainDataFactory()
    ds = factory.make(cfg, shard_fn=shard, train=True)
    it = iter(ds.dataset)
    stats = factory.stats[cfg.mix]
    expected_mask = build_action_norm_mask(
        {key: np.zeros(stat.mean.shape[-1], dtype=np.float32) for key, stat in stats.action.items()},
        arec.embodiment,
    )

    print("\n[bold]Dataset stats masks[/bold]")
    tbl = Table(show_lines=False)
    tbl.add_column("key")
    tbl.add_column("expected")
    tbl.add_column("stats")
    for key, mask in expected_mask.items():
        stats_mask = np.array(stats.action[key].mask)
        while stats_mask.ndim > 1:
            stats_mask = np.all(stats_mask, axis=0)
        assert np.array_equal(stats_mask, mask), f"{key}: stats mask mismatch"
        tbl.add_row(key, str(mask.astype(int).tolist()), str(stats_mask.astype(int).tolist()))
    print(tbl)

    for i in range(cfg.n_batches):
        batch = next(it)
        act_base = np.array(batch["act"]["base"])  # (B, H, max_a)
        act_id = np.array(batch["act"]["id"])  # (B, max_a)
        mask_act = np.array(batch["mask"]["act"])  # (B, max_a)

        B, _, _ = act_base.shape
        print(f"\n[bold]Batch {i}[/bold]  shape: act.base={act_base.shape}  act.id={act_id.shape}")

        for b in range(B):
            ids = act_id[b]
            names = [ID_TO_NAME.get(int(x), f"?{x}") for x in ids]

            tbl = Table(title=f"sample {b}", show_lines=False)
            tbl.add_column("slot", style="dim")
            tbl.add_column("dof_id", justify="right")
            tbl.add_column("name")
            tbl.add_column("mask.act", justify="center")
            tbl.add_column("norm", justify="center")
            tbl.add_column("act.base[0]", justify="right")

            norm_mask = norm_mask_from_ids(ids, stats)

            for slot, (did, name, m, n, val) in enumerate(zip(ids, names, mask_act[b], norm_mask, act_base[b, 0])):
                style = "red" if did == MASK_ID else "green"
                tbl.add_row(
                    str(slot),
                    str(int(did)),
                    f"[{style}]{name}[/{style}]",
                    str(bool(m)),
                    str(bool(n)),
                    f"{val:.4f}",
                )
            print(tbl)

            # assertions
            masked = ids == MASK_ID
            assert np.all(act_base[b, :, masked] == 0), f"batch {i} sample {b}: masked slots non-zero"
            assert np.array_equal(mask_act[b], ids != MASK_ID), f"batch {i} sample {b}: mask.act mismatch"

        print(f"[green]batch {i}: all assertions passed[/green]")

    print("\n[bold green]all batches valid[/bold green]")


if __name__ == "__main__":
    main(tyro.cli(Config))
