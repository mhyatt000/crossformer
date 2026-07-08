"""Build an arec dataset from sim MCAP episodes with ground-truth calibration.

Sim recordings carry ``foxglove.CameraCalibration`` (per-camera intrinsics) and
``foxglove.FrameTransform`` (camera extrinsics) topics, so no DREAM/SAM/DR
services are needed. Exocentric cameras are published as ``base -> cam_optical``
(static w2c repeated per step); the wrist camera is ``link_tcp -> cam_optical``
and its w2c is composed with the recorded TCP pose every step.

Output spec matches ``from_mcap.py`` builds (e.g. xgym_lift_single).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import tyro

from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.loader import _apply_fd_limit
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import BuildMGR, make_writers
from crossformer.data.mcap import (
    add_episode_info,
    filter_noops,
    map_topic_payloads,
    McapLoader,
    truncate_to_shortest_topic,
)
from crossformer.run.dream import robot_keypoints_in_cameras
from crossformer.utils.spec import spec


@dataclass(kw_only=True)
class SimBuildMGR(BuildMGR):
    path: Path
    mode: Literal["preview", "build"] = "preview"
    preview: int | None = None  # n preview
    recursive: bool = True
    max_messages_per_topic: int | None = None
    read_threads: int = 32
    prefetch_buffer_size: int = 32
    threshold: float = 1e-3

    verbose: bool = False
    vbar: bool = False  # show message-level progress bars

    mp: int = 4
    mp_buf: int = 4  # per worker buffer size

    urdf: Path = Path("xarm7_standalone.urdf")
    mesh_dir: Path | None = Path("assets")

    branch: str = "main"
    writer: Literal["source", "multisource"] = "multisource"

    def build(self, fn: Callable[[], Iterable[Any]]) -> None:
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


def _tf_to_mat(data: dict, i: int) -> np.ndarray:
    """FrameTransform message i -> parent-from-child homogeneous matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat([data["rotation"][key][i] for key in "xyzw"]).as_matrix()
    mat[:3, 3] = [data["translation"][key][i] for key in "xyz"]
    return mat


def extract_calibration(tree: dict) -> dict:
    """Pop calibration topics into ``calib`` before truncation.

    The static topics (1 camera_info per camera, 1 tf per camera) would
    otherwise collapse the episode in ``truncate_to_shortest_topic``. Views are
    ordered by sorted image-topic name to match ``map_topic_payloads``.
    """
    infos = {}
    tf = None
    remaining = {}
    for key, value in tree["topics"].items():
        if value["schema"] == "foxglove.CameraCalibration":
            infos[key.removesuffix("/camera_info")] = value["data"]
        elif value["schema"] == "foxglove.FrameTransform":
            tf = value["data"]
        else:
            remaining[key] = value
    if tf is None or not infos:
        raise ValueError(f"episode missing calibration topics: {sorted(tree['topics'])}")

    frames = {
        str(child): (str(parent), _tf_to_mat(tf, i))
        for i, (parent, child) in enumerate(zip(tf["parent_frame_id"], tf["child_frame_id"]))
    }

    ks, w2c, static, tcp_cam = [], [], [], []
    nan = np.full((4, 4), np.nan, dtype=np.float32)
    for topic in sorted(key for key, value in remaining.items() if value["schema"] == "foxglove.RawImage"):
        info = infos[topic.split("/image_raw")[0]]
        ks.append(np.asarray(info["K"][0], dtype=np.float32).reshape(3, 3))
        parent, mat = frames[str(info["frame_id"][0])]
        if parent == "base":
            w2c.append(np.linalg.inv(mat))
            static.append(True)
            tcp_cam.append(nan)
        elif parent == "link_tcp":
            w2c.append(nan)
            static.append(False)
            tcp_cam.append(mat)
        else:
            raise ValueError(f"unsupported tf parent frame: {parent}")

    out = dict(tree)
    out["topics"] = remaining
    out["calib"] = {
        "K": np.stack(ks),
        "w2c": np.stack(w2c),
        "static": np.asarray(static),
        "tcp_cam": np.stack(tcp_cam),
    }
    return out


def _tcp_to_mat(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Recorded TCP pose (meters, euler xyz) -> base-from-tcp matrices (t, 4, 4)."""
    mats = np.repeat(np.eye(4, dtype=np.float32)[None], len(position), axis=0)
    mats[:, :3, :3] = Rotation.from_euler("xyz", orientation).as_matrix()
    mats[:, :3, 3] = position
    return mats


def sim_calibrate(x: dict, cfg: SimBuildMGR) -> dict:
    """Ground-truth intrinsics/extrinsics from the calibration topics."""
    calib = x.pop("calib")
    t = len(x["images"])
    v = len(calib["K"])

    w2c = np.repeat(calib["w2c"][None], t, axis=0)
    if not calib["static"].all():
        base_tcp = _tcp_to_mat(x["proprio"]["position"], x["proprio"]["orientation"])
        for view in np.flatnonzero(~calib["static"]):
            w2c[:, view] = np.linalg.inv(base_tcp @ calib["tcp_cam"][view])

    x["state"] = {
        "intr": {"K": np.repeat(calib["K"][None], t, axis=0)},
        "extr": {"w2c": w2c},
    }
    x["mask"] = {"state": {"extr": {"w2c": np.ones((t, v), dtype=bool)}}}
    x.setdefault("info", {}).setdefault("reg", {})["iou"] = np.ones((t, v), dtype=np.float32)

    kp3dw, kp3dc, kp3dc_mask = robot_keypoints_in_cameras(
        x["proprio"]["joints"],
        x["proprio"]["gripper"],
        w2c,
        urdf_path=cfg.urdf,
        mesh_dir=cfg.mesh_dir,
    )
    x["proprio"]["kp3dw_robot"] = kp3dw
    x["proprio"]["kp3dc_robot"] = kp3dc
    x["mask"].setdefault("proprio", {})["kp3dw_robot"] = np.ones(kp3dw.shape[:-1], dtype=bool)
    x["mask"]["proprio"]["kp3dc_robot"] = kp3dc_mask
    return x


def make_dataset(loader: McapLoader, cfg: SimBuildMGR, stop: int | None = None) -> grain.IterDataset:
    ds = loader.iter_dataset(
        read_threads=cfg.read_threads,
        prefetch_buffer_size=min(cfg.prefetch_buffer_size, stop or len(loader)),
        stop=stop,
        show_message_progress=cfg.vbar,
    )
    ds = ds.map(extract_calibration)
    ds = ds.map(truncate_to_shortest_topic)
    ds = ds.map(map_topic_payloads)
    ds = ds.map(lambda x: filter_noops(x, threshold=cfg.threshold))

    _apply_fd_limit(512**2)
    ds = ds.mp_prefetch(
        grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_buf),
    )
    return add_episode_info(ds)


def main(cfg: SimBuildMGR) -> None:
    loader = McapLoader(
        cfg.path,
        recursive=cfg.recursive,
        max_messages_per_topic=cfg.max_messages_per_topic,
    )
    print(f"Root: {loader.path}")
    print(f"n_episodes={len(loader)}")

    if cfg.mode == "preview":
        n = len(loader) if cfg.preview in (None, -1) else min(cfg.preview, len(loader))
        ds = make_dataset(loader, cfg, stop=n)
        ds = ds.map(lambda x: sim_calibrate(x, cfg))
        for i, x in enumerate(tqdm(ds, total=n, desc="Loading episodes", position=0)):
            if cfg.verbose:
                print(f"\n[bold]episode={i}[/bold]")
            print(spec(x))
        return

    total = sum(
        int(x["info"]["len"][0]) for x in tqdm(make_dataset(loader, cfg), total=len(loader), desc="Counting steps")
    )
    print(f"total_steps={total}")

    ds = make_dataset(loader, cfg)
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)
    ds = ds.map(lambda x: sim_calibrate(x, cfg))
    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))
    ds = ds.map(cfg.progress(total))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(SimBuildMGR))
