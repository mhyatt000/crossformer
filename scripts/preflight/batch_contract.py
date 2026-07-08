"""Preflight: validate one real loader batch against the config-derived oracle.

Builds the grain loader exactly as finetune does, takes the first batch, and
diffs it against ``crossformer.contract.oracle.expected_batch_spec``. Fails
loudly (exit 1) on any contract violation; prints unmodeled keys for
visibility. Opt-in — nothing in the training path calls this.

Run (mirrors finetune flags):
    uv run scripts/preflight/batch_contract.py --data.mix xgym_sweep \
        --data.loader.global-batch-size 4 --wandb.no-use
"""

from __future__ import annotations

import logging
import sys

from rich import print
import tyro

from crossformer import cn
from crossformer.cn.dataset.mix import Arec
from crossformer.contract.oracle import check_batch, Dims
from crossformer.data.grain.loader import GrainDataFactory

log = logging.getLogger(__name__)


def main(cfg: cn.Train) -> None:
    names = [name for name, _ in cfg.data.mix.value.flatten()]
    sources = [Arec.from_name(n) for n in names]

    factory = GrainDataFactory()
    dims = Dims.from_config(cfg, sources, img_size=factory.resize or (64, 64))
    print("resolved dims:", dims)

    loader = factory.make(cfg, train=True)
    batch = next(iter(loader.dataset))

    report = check_batch(batch, dims)
    if report.unmodeled:
        print(f"[yellow]unmodeled keys ({len(report.unmodeled)}) — axis rules passed:[/yellow]")
        for k in sorted(report.unmodeled):
            print(f"  {k}")
    if report.errors:
        print(f"[red]contract violations ({len(report.errors)}):[/red]")
        for e in report.errors:
            print(f"  [red]-[/red] {e}")
        sys.exit(1)
    print("[green]batch contract OK[/green]")


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
