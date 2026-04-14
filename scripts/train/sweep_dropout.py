"""Sweep all 16 combinations of the 4 dropout methods in dropout_xflow.py.

Each combination runs for --steps (default 100k). Runs are launched as
subprocesses so they are fully isolated (fresh JAX process, clean wandb run).

Usage:
    uv run scripts/train/sweep_dropout.py
    uv run scripts/train/sweep_dropout.py --steps 100000 --dry-run
    uv run scripts/train/sweep_dropout.py --only 0,5,15   # by combo index
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import subprocess
import sys

import tyro

METHODS = ("patch", "view", "prop_sample", "prop_token")

PROB_FLAGS = {
    "patch": ("--patch-prob", 0.5),
    "view": ("--view-drop-prob", 0.3),
    "prop_sample": ("--proprio-sample-drop-prob", 0.2),
    "prop_token": ("--proprio-token-drop-prob", 0.3),
}


@dataclass
class Args:
    steps: int = 100_000
    batch_size: int = 256
    only: str = ""  # comma-separated combo indices (0..15); empty = all
    dry_run: bool = False
    wandb_group: str = "dropout_sweep"
    extra: str = ""  # extra args appended to every invocation, e.g. "--mp 8"


def _combo_name(flags: tuple[bool, ...]) -> str:
    on = [m for m, f in zip(METHODS, flags) if f]
    return "none" if not on else "+".join(on)


def main(args: Args):
    combos = list(product([False, True], repeat=len(METHODS)))
    assert len(combos) == 16
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else set(range(16))

    for i, flags in enumerate(combos):
        if i not in only:
            continue
        name = _combo_name(flags)
        cmd = [
            "uv",
            "run",
            "scripts/train/dropout_xflow.py",
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--name",
            f"drop_{i:02d}_{name}",
            "--wandb.group",
            args.wandb_group,
        ]
        for method, on in zip(METHODS, flags):
            flag, default_prob = PROB_FLAGS[method]
            cmd += [flag, str(default_prob if on else 0.0)]
        if args.extra:
            cmd += args.extra.split()

        print(f"[{i:2d}/16] {name}")
        print(" ", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  !! exited {rc}; continuing to next combo", file=sys.stderr)


if __name__ == "__main__":
    main(tyro.cli(Args))
