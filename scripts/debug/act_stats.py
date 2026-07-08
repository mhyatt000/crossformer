"""Per-body-part stats of bundled action targets, straight from the dataloader.

No model, no forward pass, no GPU: act.base / act.id / mask.act come out of the
grain embody pipeline. Finds degenerate normalization (huge or saturated
targets) that produces exploding gradients — the head clips actions at
+-max_action (5.0), so values beyond that train against a saturated constant.

Usage (safe to run while GPUs are busy — forces JAX to CPU):
    uv run scripts/debug/act_stats.py --mix xgym_lift
    uv run scripts/debug/act_stats.py --mix xgym_lift --batches 4 --batch-size 64
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rich import print
from rich.table import Table
import tyro


@dataclass
class Config:
    """Action-target stats probe."""

    mix: str = "xgym_lift"
    batch_size: int = 32
    batches: int = 2  # batches to accumulate
    clip: float = 5.0  # head max_action; count |v| >= clip as saturated
    seed: int = 42


def _part_by_dof_id() -> dict[int, str]:
    """Map DOF vocab id -> body part name (first part claiming the id)."""
    from crossformer.model.components.heads.loss_terms import _bodyparts_by_name

    out: dict[int, str] = {}
    for name, part in _bodyparts_by_name().items():
        try:
            dof_ids = part.dof_ids
        except KeyError:  # legacy parts whose dof names never entered the vocab
            continue
        for i in dof_ids:
            out.setdefault(i, name)
    return out


def main(cfg: Config) -> None:
    import crossformer.cn as cn
    from crossformer.cn.dataset import DataSourceE
    from crossformer.cn.dataset.dataset import Loader
    from crossformer.data.grain.embody import decode_embody_name
    from crossformer.data.grain.loader import GrainDataFactory

    train_cfg = cn.Train(
        data=cn.Dataset(mix=DataSourceE[cfg.mix], loader=Loader(use_grain=True, global_batch_size=cfg.batch_size)),
        seed=cfg.seed,
        verbosity=0,
    )
    factory = GrainDataFactory(mp=0, imaug=False)
    dataset = factory.make(train_cfg, shard_fn=lambda x: x, train=True)
    it = iter(dataset.dataset)

    id2part = _part_by_dof_id()
    # (embodiment, part) -> list of value arrays; separate mask bookkeeping
    vals: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    n_masked: dict[tuple[str, str], int] = defaultdict(int)
    n_total: dict[tuple[str, str], int] = defaultdict(int)

    for _ in range(cfg.batches):
        batch = next(it)
        base = np.asarray(batch["act"]["base"])  # (B, [W,] H, A)
        if base.ndim == 3:
            base = base[:, None]
        dof_ids = np.asarray(batch["act"]["id"])  # (B, A)
        mask_act = batch.get("mask", {}).get("act")
        mask_act = np.ones_like(dof_ids, dtype=bool) if mask_act is None else np.asarray(mask_act, dtype=bool)
        # mask.horizon marks valid future chunk steps (folded into chunk_steps
        # during training) — exclude padded steps here too, like the loss does.
        h_mask = batch.get("mask", {}).get("horizon")
        if h_mask is not None:
            h_mask = np.asarray(h_mask, dtype=bool)
            if h_mask.ndim == 3:  # (B, W, H) -> (B, H)
                h_mask = h_mask[:, 0]
        else:
            h_mask = np.ones((base.shape[0], base.shape[2]), dtype=bool)
        embody = batch.get("act", {}).get("embody")
        names = (
            [decode_embody_name(np.asarray(embody)[i]) for i in range(base.shape[0])]
            if embody is not None
            else ["?"] * base.shape[0]
        )

        for i in range(base.shape[0]):
            for a in range(dof_ids.shape[1]):
                did = int(dof_ids[i, a])
                if did == 0:  # MASK/pad slot
                    continue
                key = (names[i], id2part.get(did, f"dof_{did}"))
                n_total[key] += 1
                if not mask_act[i, a]:
                    n_masked[key] += 1
                    continue
                vals[key].append(base[i, :, h_mask[i], a].ravel())

    table = Table(title=f"act.base stats — {cfg.mix} ({cfg.batches}x{cfg.batch_size} samples, clip={cfg.clip})")
    for col in ("embodiment", "part", "slots", "masked%", "std", "min", "max", f"sat@{cfg.clip:g}%"):
        table.add_column(col, justify="right")

    worst: list[tuple[float, str]] = []
    for key in sorted(set(n_total)):
        emb, part = key
        masked_pct = 100.0 * n_masked[key] / n_total[key]
        if key not in vals:
            table.add_row(emb, part, str(n_total[key]), f"{masked_pct:.0f}", "-", "-", "-", "-")
            continue
        v = np.concatenate(vals[key])
        sat = 100.0 * float((np.abs(v) >= cfg.clip).mean())
        table.add_row(
            emb, part, str(n_total[key]), f"{masked_pct:.0f}",
            f"{v.std():.2f}", f"{v.min():.1f}", f"{v.max():.1f}", f"{sat:.1f}",
        )
        if sat > 0.5 or v.std() > 3.0 or np.abs(v).max() > 3 * cfg.clip:
            worst.append((sat, f"{emb}/{part}: std={v.std():.1f} range=[{v.min():.1f},{v.max():.1f}] sat={sat:.1f}%"))

    print(table)
    if worst:
        print("[bold red]suspect parts (saturated or huge targets):[/]")
        for _, msg in sorted(worst, reverse=True):
            print(f"  {msg}")
    else:
        print("[bold green]no saturated/degenerate parts found — targets look sane[/]")


if __name__ == "__main__":
    main(tyro.cli(Config))
