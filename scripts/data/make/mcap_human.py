from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Literal

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
import jax
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client
from xclients.cli.preprocess import extract_kp3d_step, make_intrinsics

from crossformer.data.arec.arec import ArrayRecordBuilder, WriterSpec
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import add_episode_id, add_step_id, add_traj_len, BuildMGR, init_info
from crossformer.data.mcap import decode_cameras, McapLoader
from crossformer.utils.spec import spec


@dataclass
class Endpoint:
    host: str = "localhost"
    port: int = 8080


def make_writers(writer: Literal["source", "multisource"]) -> WriterSpec:
    if writer == "source":
        return {"data": ["*"]}
    return {
        "image": (["images"], {"options": "group_size:1"}),
        "proprio": (["proprio", "info", "mask"], {"options": "group_size:32"}),
    }


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

    take: int | None = None  # debug. take n steps
    preview: bool = False  # debug. preview dataset and exit

    branch: str = "main"
    writer: Literal["source", "multisource"] = "multisource"

    def build(self, fn) -> None:
        print(self)
        kwargs = {} if self.shard_size is None else {"shard_size": self.shard_size}
        builder: ArrayRecordBuilder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            branch=self.branch,
            writers=make_writers(self.writer),
            **kwargs,
        )
        builder.prepare(fn)


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


def count_steps(ds, cfg: MyBuildMGR) -> tuple[int, int]:
    total = 0
    n = 0
    for x in tqdm(make_episode_dataset(ds, cfg), total=len(ds), desc="compute total"):
        total += int(x["info"]["len"][0])
        n += 1
    return total, n


def main(cfg: MyBuildMGR):
    client = Client(host=cfg.wilor.host, port=cfg.wilor.port)
    loader = McapLoader(
        cfg.path,
        recursive=cfg.recursive,
    )
    print(f"root: {loader.path}")
    print(f"episodes: {len(loader)}")

    ds = make_episode_dataset(loader, cfg)

    # force wilor runs serially
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)

    print("extracting keypoints with wilor...")
    K = make_intrinsics(w=640, h=480, fx=cfg.fxy, fy=cfg.fxy)
    _extract_one = partial(extract_kp3d_step, client=client, K=K)

    def _extract_one_join(x: dict) -> dict:
        x = x | {"observation": _extract_one(x["observation"])}

        # prepare the output for the multisource writer
        out = {}
        obs = x.pop("observation")
        images = [v for k, v in obs.items() if k.startswith("cam_")]
        out["images"] = np.stack(images, axis=0)
        out["proprio"] = {
            "kp2d_hand": np.stack(list(obs.pop("kp2d").values()), axis=0),
            "kp3dc_hand": np.stack(list(obs.pop("kp3d_cam").values()), axis=0),
        }
        out["info"] = x.pop("info")
        visible = np.stack(list(obs.pop("visible").values()), axis=0)
        out["mask"] = {"proprio": {"kp3dc_hand": visible}}
        return out

    total, n = 0, len(loader)

    def extract(x: dict) -> dict:
        tree_get = lambda d, i: jax.tree.map(lambda v: v[i], d)
        bar = partial(tqdm, total=len(x["info"]["len"]), desc="extracting keypoints", leave=False)
        results = [_extract_one_join(tree_get(x, i)) for i in bar(range(len(x["info"]["len"])))]

        stacked = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *results)
        return stacked

    # ds = ds.map(extract) # if extracting before flatten

    # total, n = count_steps(loader, cfg)
    # print(f"total steps: {total} across {n} episodes")

    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))
    ds = ds.map(_extract_one_join)  # extract after flatten

    if cfg.take:  # debug
        # total = min(total, cfg.preview)
        ds = grain.MapDataset.source(list(islice(iter(ds), cfg.take)))
    if cfg.preview:
        for x in ds:
            print(spec(x))
            quit()

    ds = ds.map(cfg.progress(total=None))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
