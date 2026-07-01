"""tools for writing raw data as grain arec"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from crossformer.data.arec.arec import ArrayRecordBuilder, WriterSpec
from crossformer.data.grain.utils import traj_len


def add_traj_len(x: dict[str, Any]) -> dict[str, Any]:
    n = int(traj_len(x))
    x["info"]["len"] = np.full((n,), n)
    return x


def add_step_id(x: dict[str, Any]) -> dict[str, Any]:
    n = int(traj_len(x))
    x["info"]["id"]["step"] = np.arange(n)
    return x


def add_episode_id(i: int, x: dict[str, Any]) -> dict[str, Any]:
    x["info"]["id"]["episode"] = np.full((len(x["info"]["id"]["step"]),), i)
    return x


def init_info(x: dict[str, Any]) -> dict[str, Any]:
    x.setdefault("info", {})
    x["info"].setdefault("id", {})
    return x


def make_writers(writer: Literal["source", "multisource"]) -> WriterSpec:
    if writer == "source":
        return {"data": ["*"]}
    return {
        "image": (["images"], {"options": "group_size:1"}),
        "proprio": (["proprio", "info", "state", "mask"], {"options": "group_size:32"}),
    }


@dataclass
class BuildMGR:
    name: str
    version: str
    shard_size: int | None = None

    def __post_init__(self) -> None:
        # assert self.version.startswith("v"), "Version should start with 'v'"
        assert self.version.count(".") == 2, "Version should be in the format 'vX.Y.Z'"

    def build(self, fn: Callable[[], Iterable[Any]]) -> None:
        print(self)
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            **{"shard_size": self.shard_size} if self.shard_size is not None else {},
        )
        print(self.builder.root)

        self.builder.prepare(fn)

    def progress(self, total: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """a mappable progress bar. example usage: ds.map(cfg.progress(total=1000))"""
        bar = tqdm(total=total, desc="Building dataset")

        def build_progress(x: dict[str, Any]) -> dict[str, Any]:
            bar.update(1)
            return x

        return build_progress

    def yield_from_ds(self, ds: Iterable[dict[str, Any]]) -> Callable[[], Iterator[dict[str, Any]]]:
        """a helper to yield from a dataset. example usage: cfg.build(cfg.yield_from_ds(ds))"""

        def yield_from() -> Iterator[dict[str, Any]]:
            yield from ds

        return yield_from  # returns generator object
