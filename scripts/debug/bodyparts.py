"""Print every embodiment's expanded body parts and slot layout.

Shows where each action dim comes from: per-part width, slot range, dof-id
range, and per-view expansion — i.e. why human_single is 192 wide.

Usage:
    uv run scripts/debug/bodyparts.py
    uv run scripts/debug/bodyparts.py --embodiment human_single
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import print
from rich.table import Table
import tyro

from crossformer.embody import Embodiment


@dataclass
class Config:
    embodiment: str | None = None  # filter by name; None = all


def part_table(emb: Embodiment) -> Table:
    tbl = Table(title=f"{emb.name} — action_dim={emb.action_dim}")
    tbl.add_column("slot part", style="bold")
    tbl.add_column("view", justify="center")
    tbl.add_column("dims", justify="right")
    tbl.add_column("slots", justify="center")
    tbl.add_column("dof ids", justify="center")
    tbl.add_column("frame")
    tbl.add_column("kind")

    offset = 0
    for p in emb.expanded:
        ids = p.dof_ids
        tbl.add_row(
            p.name if p.view == 0 else f"{p.name}_v{p.view}",
            "-" if p.view == 0 else str(p.view),
            str(p.action_dim),
            f"[{offset}:{offset + p.action_dim})",
            f"{min(ids)}..{max(ids)}",
            str(p.frame),
            str(p.kind),
        )
        offset += p.action_dim
    return tbl


def main(cfg: Config) -> None:
    registry = Embodiment.REGISTRY
    if cfg.embodiment is not None:
        registry = {cfg.embodiment: registry[cfg.embodiment]}

    for emb in registry.values():
        print(part_table(emb))
        print()

    summary = Table(title="embodiments (max_a over a mix = max of its action_dims)")
    summary.add_column("embodiment", style="bold")
    summary.add_column("catalog parts")
    summary.add_column("expanded parts", justify="right")
    summary.add_column("action_dim", justify="right")
    for emb in Embodiment.REGISTRY.values():
        summary.add_row(
            emb.name,
            "+".join(emb.part_names),
            str(len(emb.expanded)),
            str(emb.action_dim),
        )
    print(summary)


if __name__ == "__main__":
    main(tyro.cli(Config))
