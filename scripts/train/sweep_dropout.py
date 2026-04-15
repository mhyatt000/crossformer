"""Run a targeted set of dropout configurations from dropout_xflow.py.

Each run is launched as its own subprocess for isolation.

Configs:
  0. none                — all dropouts off, no shuffle
  1. patch               — patch occlusion only
  2. view                — per-sample image-view drop only
  3. prop_sample         — per-sample proprio drop only
  4. prop_token          — per-sample proprio-token drop only
  5. all4                — all 4 dropouts on
  6. shuffle_only        — no dropout, image-key shuffle on

Usage:
    uv run scripts/train/sweep_dropout.py
    uv run scripts/train/sweep_dropout.py --steps 100000 --dry-run
    uv run scripts/train/sweep_dropout.py --only 0,5   # by run index
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys

import tyro

METHODS = ("patch", "view", "key_shuf", "prop_sample", "prop_token")

PROB_FLAGS = {
    "patch": ("--patch-prob", 0.5),
    "view": ("--view-drop-prob", 0.3),
    "key_shuf": ("--image-key-shuffle-prob", 0.3),
    "prop_sample": ("--proprio-sample-drop-prob", 0.2),
    "prop_token": ("--proprio-token-drop-prob", 0.3),
}

# (name, set of methods to enable)
RUNS: list[tuple[str, set[str]]] = [
    ("none", set()),
    ("patch", {"patch"}),
    ("view", {"view"}),
    ("prop_sample", {"prop_sample"}),
    ("prop_token", {"prop_token"}),
    ("all4", {"patch", "view", "prop_sample", "prop_token"}),
    ("shuffle_only", {"key_shuf"}),
]


@dataclass
class Args:
    steps: int = 100_000
    batch_size: int = 256
    only: str = ""  # comma-separated run indices; empty = all
    dry_run: bool = False
    wandb_group: str = "dropout_sweep"
    extra: str = ""  # extra args appended to every invocation, e.g. "--mp 8"


def main(args: Args):
    n = len(RUNS)
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else set(range(n))

    for i, (name, on_set) in enumerate(RUNS):
        if i not in only:
            continue
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
        for method in METHODS:
            flag, default_prob = PROB_FLAGS[method]
            cmd += [flag, str(default_prob if method in on_set else 0.0)]
        if args.extra:
            cmd += args.extra.split()

        print(f"[{i:2d}/{n}] {name}")
        print(" ", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  !! exited {rc}; continuing to next run", file=sys.stderr)


if __name__ == "__main__":
    main(tyro.cli(Args))
