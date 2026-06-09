from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client
from xclients.cli.preprocess import Episode, extract_kp3d_step, make_intrinsics, PrepConfig

from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import add_episode_id, add_step_id, add_traj_len, BuildMGR, init_info


# Multisource layout mirroring the robot build (from_zarr.make_writers): the
# image writer holds the per-frame views, the proprio writer holds the palm +
# visibility + info. Stacking the proprio writer over a window is what gives
# the loader a horizon, so the per-frame palm MUST live in proprio.
MANO_WRITERS = {
    "image": (["image"], {"options": "group_size:1"}),
    "proprio": (["proprio", "info"], {"options": "group_size:32"}),
}


def to_multisource(step: dict) -> dict:
    """Reshape one extracted frame into the image/proprio multisource writers.

    Keeps EVERY camera (no view is chosen here) so the loader can pick a camera
    per-sample at train time. palm = wrist keypoint (joint 0) of each camera's
    single-view PnP lift, in that camera's own frame. visible rides in proprio
    so it stacks per-step and can drive the per-timestep loss mask.
    """
    obs = step["observation"]
    cams = list(obs["kp3d_cam"].keys())
    return {
        "image": {c: np.asarray(obs[c]) for c in cams},  # (480,640,3) uint8
        "proprio": {
            "position": {c: np.asarray(obs["kp3d_cam"][c][0], dtype=np.float32) for c in cams},  # (3,)
            "visible": {c: np.asarray(obs["visible"][c], dtype=bool) for c in cams},  # ()
        },
        "info": step["info"],
    }


@dataclass(kw_only=True)
class MyBuildMGR(BuildMGR):
    prep: PrepConfig

    name: str
    version: str
    shard_size: int = 1000

    fn: Callable = field(init=False, default=None)
    builder: ArrayRecordBuilder = field(init=False, default=None)

    take: int | None = None  # debug. take n steps

    def build(self, fn):
        # Override BuildMGR.build to write the multisource {image, proprio}
        # layout instead of the single "data" writer default.
        print(self)
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            shard_size=self.shard_size,
            writers=MANO_WRITERS,
        )
        print(self.builder.root)
        self.builder.prepare(fn)


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

    # reshape into the image/proprio multisource writers (all cameras kept)
    ds = ds.map(to_multisource)

    cfg.build(lambda: tqdm(ds, desc="building"))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
