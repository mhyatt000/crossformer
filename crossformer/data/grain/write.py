"""tools for writing raw data as grain arec"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from crossformer.data.grain.utils import traj_len


def add_traj_len(x: dict) -> dict:
    n = int(traj_len(x))
    x["info"]["len"] = np.full((n,), n)
    return x


def add_step_id(x: dict) -> dict:
    n = int(traj_len(x))
    x["info"]["id"]["step"] = np.arange(n)
    return x


def add_episode_id(i, x: dict) -> dict:
    x["info"]["id"]["episode"] = np.full((len(x["info"]["id"]["step"]),), i)
    return x


def init_info(x: dict) -> dict:
    x.setdefault("info", {})
    x["info"].setdefault("id", {})
    return x


@dataclass
class BuildMGR:
    name: str
    version: str
    shard_size: int = 1000

    def __post_init__(self):
        # assert self.version.startswith("v"), "Version should start with 'v'"
        assert self.version.count(".") == 2, "Version should be in the format 'vX.Y.Z'"

    def build(self, fn):
        print(self)
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            shard_size=self.shard_size,
        )
        print(self.builder.root)

        self.builder.prepare(fn)

    def progress(total: int):
        """a mappable progress bar. example usage: ds.map(cfg.progress(total=1000))"""
        bar = tqdm(total=total, desc="Building dataset")

        def build_progress(x: dict) -> dict:
            bar.update(1)
            return x

    def yield_from_ds(ds) -> Callable[Iterator]:
        """a helper to yield from a dataset. example usage: cfg.build(cfg.yield_from_ds(ds))"""

        def yield_from(ds) -> Iterator[dict]:
            yield from ds

        return yield_from(ds)
