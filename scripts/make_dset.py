from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client
from xclients.cli.preprocess import Episode, extract_kp3d_step, make_intrinsics, PrepConfig

from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import add_episode_id, add_step_id, add_traj_len, BuildMGR, init_info


@dataclass
class MyBuildMGR(BuildMGR):
    prep: PrepConfig

    name: str
    version: str
    shard_size: int = 1000

    fn: Callable = field(init=False)
    builder: ArrayRecordBuilder = field(init=False)

    take: int | None = None  # debug. take n steps


def main(cfg: MyBuildMGR):
    client = Client(host=cfg.prep.host, port=cfg.prep.port)

    ds = grain.MapDataset.source(list(cfg.prep.dir.glob("ep*.npz")))
    ds = ds.map(lambda f: Episode.from_npz(f).data).map(lambda x: {"observation": x})
    ds = ds.map(init_info).map(add_traj_len).map(add_step_id).map_with_index(add_episode_id)

    # materialize to compute total steps for progress bar
    total, n = sum([x["info"]["len"][0] for x in tqdm(ds, desc="compute total")]), len(ds)
    print(f"total steps: {total} across {n} episodes")

    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))

    if cfg.take:  # debug
        dsit = iter(ds)
        ds = grain.MapDataset.source([next(dsit) for _ in range(cfg.take)])

    # force wilor runs serially
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)

    K = make_intrinsics(w=640, h=480, fx=515.0, fy=515.0)
    _extract = partial(extract_kp3d_step, client=client, K=K)
    # _extract = partial(extract_kp3d_ep, client=client)
    ds = ds.map(lambda x: x | {"observation": _extract(x["observation"])})

    ds = ds.map(cfg.progress(total=total))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
