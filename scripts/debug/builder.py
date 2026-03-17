from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

from rich.pretty import pprint
from tqdm import tqdm
import tyro

from crossformer.data.arec.generate import Builder

log = logging.getLogger(__name__)


@dataclass
class DebugBuilder(Builder):
    root: Path
    name: str = ""
    version: str = ""

    threshold: float = 1e-3
    workers: int = 32
    limit: int | None = None


@dataclass
class Config:
    dir: Path
    name: str = "my_dataset"
    workers: int = 16
    verbose: bool = False
    version: str = "0.5.0"
    limit: int | None = None
    route: Literal["build", "clean"] = "build"
    dry: bool = True  # dry run for cleaning (don't actually delete files)
    by_step: bool = False


def _first(ep: list[dict], by_step: bool):
    if by_step:
        return ep[0] if ep else None
    return ep


def _iter_episodes(builder: DebugBuilder) -> Iterable[list[dict]]:
    for ep in builder.build():
        if ep is not None:
            yield ep


def main(cfg: Config) -> None:
    if not cfg.dir.is_dir():
        raise SystemExit(f"Directory not found: {cfg.dir}")
    if cfg.verbose:
        logging.basicConfig(level=logging.DEBUG)

    builder = DebugBuilder(
        root=cfg.dir,
        name=cfg.name,
        version=cfg.version,
        workers=cfg.workers,
        limit=cfg.limit,
    )

    if cfg.route == "clean":
        builder.clean(dry=cfg.dry)
        return

    total_eps = 0
    total_steps = 0
    first = None

    for ep in tqdm(_iter_episodes(builder), desc="episodes"):
        total_eps += 1
        total_steps += len(ep)
        if first is None:
            first = _first(ep, cfg.by_step)

    pprint(
        {
            "name": builder.name,
            "dir": str(builder.root),
            "workers": builder.workers,
            "episodes": total_eps,
            "steps": total_steps,
        }
    )
    if first is not None:
        pprint(builder.spec(first))


if __name__ == "__main__":
    main(tyro.cli(Config))
