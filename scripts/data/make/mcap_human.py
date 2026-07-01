from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client
from xclients.cli.preprocess import extract_kp3d_step, make_intrinsics

from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import add_episode_id, add_step_id, add_traj_len, BuildMGR, init_info
from crossformer.data.mcap import decode_cameras, McapLoader


@dataclass
class Endpoint:
    host: str = "localhost"
    port: int = 8080


@dataclass(kw_only=True)
class MyBuildMGR(BuildMGR):
    path: Path
    recursive: bool = True
    read_threads: int = 8
    prefetch_buffer_size: int = 8
    vbar: bool = False

    fxy: float = 515.0
    wilor: Endpoint = field(default_factory=Endpoint)
    shard_size: int | None = 1000
    preview: int | None = None  # debug. take n steps


def mcap_to_episode(tree: dict) -> dict:
    return {"observation": decode_cameras(tree)}


def make_episode_dataset(loader: McapLoader, cfg: MyBuildMGR):
    ds = loader.iter_dataset(
        read_threads=cfg.read_threads,
        prefetch_buffer_size=min(cfg.prefetch_buffer_size, len(loader)),
        show_message_progress=cfg.vbar,
    )
    ds = ds.map(mcap_to_episode)
    ds = ds.map(init_info).map(add_traj_len).map(add_step_id).map_with_index(add_episode_id)
    return ds


def count_steps(loader: McapLoader, cfg: MyBuildMGR) -> tuple[int, int]:
    total = 0
    n = 0
    for x in tqdm(make_episode_dataset(loader, cfg), total=len(loader), desc="compute total"):
        total += int(x["info"]["len"][0])
        n += 1
    return total, n


def main(cfg: MyBuildMGR):
    loader = McapLoader(
        cfg.path,
        recursive=cfg.recursive,
    )
    print(f"root: {loader.path}")
    print(f"episodes: {len(loader)}")

    total, n = count_steps(loader, cfg)
    print(f"total steps: {total} across {n} episodes")

    ds = make_episode_dataset(loader, cfg)
    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))

    # force wilor runs serially
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)

    client = Client(host=cfg.wilor.host, port=cfg.wilor.port)
    K = make_intrinsics(w=640, h=480, fx=cfg.fxy, fy=cfg.fxy)
    _extract = partial(extract_kp3d_step, client=client, K=K)
    ds = ds.map(lambda x: x | {"observation": _extract(x["observation"])})

    if cfg.preview:  # debug
        total = min(total, cfg.preview)
        ds = grain.MapDataset.source(list(islice(iter(ds), cfg.preview)))
        for x in ds:
            print(spec(x))
            quit()

    ds = ds.map(cfg.progress(total=total))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
