from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import cv2
import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
import jax
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import tyro
from webpolicy.client import Client

from crossformer.data.arec.arec import ArrayRecordBuilder, WriterSpec
from crossformer.data.grain.loader import _apply_fd_limit
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import BuildMGR
from crossformer.data.mcap import McapLoader
from crossformer.data.utils.trajectory import scan_noop
from crossformer.run.dream import (
    dream_w2c_cv_to_roboreg_ht,
    filter_w2c_by_iou,
    make_intr,
    mean_extr,
    roboreg_ht_to_dream_w2c_cv,
    robot_keypoints_in_cameras,
)
from crossformer.utils.spec import diff, SimpleSpec, spec
from crossformer.utils.tree import flat


def make_writers(writer: Literal["source", "multisource"]) -> WriterSpec:
    if writer == "source":
        return {"data": ["*"]}
    return {
        "image": (["images"], {"options": "group_size:1"}),
        "proprio": (["proprio", "info", "state", "mask"], {"options": "group_size:32"}),
    }


@dataclass
class Endpoint:
    host: str
    port: int


@dataclass
class Sam(Endpoint):
    prompt: str = "robot"
    confidence: float = 0.3
    close_kernel_size: int = 3
    min_component_area: int = 16


@dataclass
class Roboreg(Endpoint):
    pass


@dataclass
class Dream(Endpoint):
    units: Literal["deg", "rad"] = "deg"


class ClientLike(Protocol):
    def step(self, payload: dict) -> dict: ...


@dataclass(kw_only=True)
class MyBuildMGR(BuildMGR):
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

    n: int = 32  # number of frames to use for registration
    min_iou: float = 0.75  # keep calibrated w2c only above this registration IoU
    urdf: Path = Path("xarm7_standalone.urdf")
    mesh_dir: Path | None = Path("assets")

    image_size: int = 200  # square size for SAM, DREAM, and DR
    fxy: float = 515.0  # focal length for DR depth-to-3D conversion
    mask_area: list[float] = field(default_factory=lambda: [0.005, 0.8])

    dr: bool = True
    sam: Sam = field(default_factory=lambda: Sam(host="localhost", port=8080))
    reg: Roboreg = field(default_factory=lambda: Roboreg(host="localhost", port=8081))
    dream: Dream = field(default_factory=lambda: Dream(host="localhost", port=8082))

    branch: str = "main"
    writer: Literal["source", "multisource"] = "multisource"

    def __post_init__(self) -> None:
        super().__post_init__()

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


def first_spec_match():
    """Compare each item spec with the first."""
    first = None

    def match(item) -> tuple[bool, dict]:
        nonlocal first
        step = jax.tree.map(lambda x: x[0] if getattr(x, "ndim", 0) else x, item)
        current = {key: value for key, value in spec(flat(step)).items() if isinstance(value, SimpleSpec)}
        if first is None:
            first = current
            return True, {}
        delta = diff(first, current)
        return (True, {}) if not any(delta.values()) else (False, delta)

    return match


def _constant(x, name: str):
    x = np.asarray(x)
    if not len(x) or not np.all(x == x[0]):
        raise ValueError(f"RawImage {name} must be constant")
    return x[0].item()


def _raw_images(value: dict) -> np.ndarray:
    data = value["data"]
    h = _constant(data["height"], "height")
    w = _constant(data["width"], "width")
    step = _constant(data["step"], "step")
    if step % w:
        raise ValueError(f"RawImage step={step} is not divisible by width={w}")
    c = step // w
    images = data["data"]
    if images.shape[1] != h * step:
        raise ValueError(f"RawImage data width={images.shape[1]} does not match height x step={h * step}")
    images = images.reshape(len(images), h, w, c)
    if c == 2:
        images = np.stack([cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2) for image in images])
    return images


def truncate_to_shortest_topic(tree: dict) -> dict:
    """Truncate every topic to the shortest topic length."""
    topics = tree["topics"]
    if not topics:
        raise ValueError("episode has no topics")
    lengths = {key: len(value["log_time"]) for key, value in topics.items()}
    n = min(lengths.values())
    if not n:
        raise ValueError(f"episode has an empty topic: {lengths}")

    out = dict(tree)
    out["topics"] = {
        key: jax.tree.map(
            lambda x: x[:n] if isinstance(x, np.ndarray) and x.ndim else x,
            value,
        )
        for key, value in topics.items()
    }
    return out


def _pose(value: dict) -> tuple[np.ndarray, np.ndarray]:
    data = value["data"]
    position = np.stack([data["position"][key] for key in ("x", "y", "z")], axis=-1)
    quaternion = np.stack([data["orientation"][key] for key in ("x", "y", "z", "w")], axis=-1)
    position = (position / 1e3).astype(np.float32)
    orientation = Rotation.from_quat(quaternion).as_euler("xyz").astype(np.float32)
    return position, orientation


def map_topic_payloads(tree: dict) -> dict:
    """Group known MCAP topics into images and proprio."""
    images = []
    proprio = {}
    other = {}
    for key, value in tree["topics"].items():
        schema = value["schema"]
        if schema == "foxglove.JointStates":
            proprio["joints"] = value["data"]["joints"]["position"]
            continue
        if schema == "foxglove.Pose":
            proprio["position"], proprio["orientation"] = _pose(value)
            continue
        if schema == "xclients.Gripper":
            proprio["gripper"] = np.asarray(value["data"]["norm"], dtype=np.float32)[:, None]
            continue
        if schema == "foxglove.RawImage":
            images.append(_raw_images(value))
            continue
        other[key] = value

    if not images:
        raise ValueError("episode has no RawImage topics")
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"RawImage topics must share shape, got {sorted(shapes)}")

    out = {key: value for key, value in tree.items() if key != "topics"}
    out["images"] = np.stack(images, axis=1)
    out["proprio"] = proprio
    if other:
        out["topics"] = other
    return out


def filter_noops(tree: dict, threshold: float = 1e-3) -> dict:
    """Remove steps that are no-ops in both Cartesian and joint space."""
    proprio = tree["proprio"]
    gripper = proprio["gripper"]
    pos = np.concatenate((proprio["position"], gripper), axis=-1)
    jpos = np.concatenate((proprio["joints"], gripper), axis=-1)
    mask = np.logical_and(
        ~np.asarray(scan_noop(pos, threshold=threshold)),
        ~np.asarray(scan_noop(jpos, threshold=threshold)),
    )
    n = len(mask)
    print(f"mask | keep={sum(mask)} / total={n}")
    return jax.tree.map(
        lambda x: x[mask] if isinstance(x, np.ndarray) and x.ndim and len(x) == n else x,
        tree,
    )


def add_episode_info(ds):
    """Add contiguous IDs after filtering."""
    global_step = 0
    episode = 0

    def add(tree: dict) -> dict:
        nonlocal episode, global_step
        n = len(tree["images"])
        step = np.arange(n, dtype=np.int64)
        info = tree["info"]
        info.pop("episode", None)
        info.pop("path", None)
        info["id"] = {
            "episode": np.full(n, episode, dtype=np.int64),
            "step": step,
            "global": step + global_step,
        }
        info["len"] = np.full(n, n, dtype=np.int64)
        global_step += n
        episode += 1
        return tree

    return ds.map(add)


class SamClientWrapper:
    def __init__(self, client: ClientLike, cfg: Sam):
        self.client = client
        self.cfg = cfg

    def step(self, image: np.ndarray) -> dict:
        h, w, _c = image.shape
        payload = {
            "image": image,
            "type": "image",
            "text": self.cfg.prompt,
            "confidence": self.cfg.confidence,
        }
        out = self.client.step(payload)
        valid = np.prod(out["masks"].shape) > 0
        seg = out["masks"].any(axis=0).reshape(h, w, 1) if valid else np.zeros((h, w, 1), dtype=bool)
        return {"seg": seg, "valid": valid}


def _tree_stack(xs: list[dict]) -> dict:
    return jax.tree.map(lambda *ys: np.stack(ys), *xs)


def select_registration_frames(x: dict, n: int) -> list[dict]:
    """Select episode frames for calibration, preserving the camera axis."""
    t = len(x["info"]["id"]["episode"])
    idx = np.linspace(0, t - 1, min(n, t), dtype=np.int32)
    return [jax.tree.map(lambda y: y[i], x) for i in idx]


def dream_calibrate_step(x: dict, dream: ClientLike, cfg: MyBuildMGR) -> dict:
    images = np.asarray(x["images"])
    h, w = images.shape[1:3]
    k = make_intr(fx=cfg.fxy, fy=cfg.fxy, w=w, h=h)
    q = np.asarray(x["proprio"]["joints"], dtype=np.float32)
    q = np.rad2deg(q) if cfg.dream.units == "deg" else q
    payload = {
        "image": images,
        "K": k,
        "q": q,
        "type": "image",
        "calibrate": True,
    }
    out = dream.step(payload)
    w2c = np.asarray(out["w2c"], dtype=np.float32).copy()
    valid = (
        np.asarray(out["pnp_success"], dtype=bool)
        & np.isfinite(w2c).all(axis=(1, 2))
        & (np.asarray(out["pnp_valid"]).sum(axis=1) >= 5)
        & np.isfinite(out["pnp_reproj_px"])
        & (np.asarray(out["pnp_reproj_px"]) < 20.0)
    )
    w2c[~valid] = np.nan
    x["state"] = {
        "intr": {"K": np.repeat(k[None], images.shape[0], axis=0)},
        "extr": {"w2c": w2c},
    }
    x["mask"] = {"state": {"extr": {"w2c": valid}}}
    return x


def segment_step(x: dict, sam: ClientLike) -> dict:
    valid = np.asarray(x["mask"]["state"]["extr"]["w2c"], dtype=bool)

    def dummy(image: np.ndarray) -> dict:
        h, w, _c = image.shape
        return {"seg": np.zeros((h, w, 1), dtype=bool), "valid": False}

    seg = [sam.step(image) if ok else dummy(image) for image, ok in zip(x["images"], valid)]
    seg = _tree_stack(seg)
    x["seg"] = seg["seg"]
    x.setdefault("mask", {}).setdefault("obs", {})["seg"] = seg["valid"]
    return x


def batch_registration(x: dict, roboreg: ClientLike, cfg: MyBuildMGR) -> dict:
    w2cs = []
    ious = []
    images = x["images"]
    for view in range(images.shape[1]):
        img = images[:, view]
        seg = x["seg"][:, view]
        extr = x["state"]["extr"]["w2c"][:, view]
        valid = x["mask"]["state"]["extr"]["w2c"][:, view]

        if not valid.any():
            print(f"skipping registration for view={view} due to invalid DREAM w2c")
            w2cs.append(np.full((4, 4), np.nan, dtype=np.float32))
            ious.append(0.0)
            continue

        img = img[valid]
        seg = seg[valid]
        extr = extr[valid]
        joints = x["proprio"]["joints"][valid]
        _t, h, w, _c = img.shape
        payload = {
            "depth": img[..., 0],
            "joints": joints,
            "mask": seg.reshape(len(seg), h, w).astype(int) * 255,
            "intrinsics": make_intr(fx=cfg.fxy, fy=cfg.fxy, w=w, h=h),
            "HT": dream_w2c_cv_to_roboreg_ht(mean_extr(extr)),
            "mode": "dr",
        }
        out = roboreg.step(payload)
        iou = float(out["iou"])
        if iou > cfg.min_iou:
            w2c = roboreg_ht_to_dream_w2c_cv(np.asarray(out["HT"], dtype=np.float32))
        else:
            w2c = np.full((4, 4), np.nan, dtype=np.float32)
        w2cs.append(w2c)
        ious.append(iou)

    w2c, keep = filter_w2c_by_iou(np.stack(w2cs), np.asarray(ious), threshold=cfg.min_iou)
    return {"w2c": w2c, "iou": np.asarray(ious, dtype=np.float32), "valid": keep}


def add_robot_keypoints(x: dict, cfg: MyBuildMGR) -> dict:
    kp3dw, kp3dc, kp3dc_mask = robot_keypoints_in_cameras(
        x["proprio"]["joints"],
        x["proprio"]["gripper"],
        x["state"]["extr"]["w2c"],
        urdf_path=cfg.urdf,
        mesh_dir=cfg.mesh_dir,
    )
    x["proprio"]["kp3dw_robot"] = kp3dw
    x["proprio"]["kp3dc_robot"] = kp3dc
    x.setdefault("mask", {}).setdefault("proprio", {})["kp3dw_robot"] = np.ones(kp3dw.shape[:-1], dtype=bool)
    x["mask"]["proprio"]["kp3dc_robot"] = kp3dc_mask
    return x


def calibrate_extrinsics(x: dict, sam: ClientLike, dream: ClientLike, roboreg: ClientLike, cfg: MyBuildMGR) -> dict:
    reg = select_registration_frames(x, n=cfg.n)
    reg = [dream_calibrate_step(step, dream, cfg) for step in tqdm(reg, desc="DREAM calibration")]
    reg = [segment_step(step, sam) for step in tqdm(reg, desc="SAM segmentation")]
    registration = batch_registration(_tree_stack(reg), roboreg, cfg)

    t = len(x["info"]["id"]["episode"])
    v, h, w = x["images"].shape[1:4]
    k = make_intr(fx=cfg.fxy, fy=cfg.fxy, w=w, h=h)
    x["state"] = {
        "intr": {"K": np.repeat(k[None, None], t * v, axis=0).reshape(t, v, 3, 3)},
        "extr": {"w2c": np.repeat(registration["w2c"][None], t, axis=0)},
    }
    x["mask"] = {
        "state": {"extr": {"w2c": np.repeat(registration["valid"][None], t, axis=0)}},
    }
    x.setdefault("info", {}).setdefault("reg", {})["iou"] = np.repeat(registration["iou"][None], t, axis=0)
    return add_robot_keypoints(x, cfg)


def make_dataset(loader: McapLoader, cfg: MyBuildMGR, stop: int | None = None):
    ds = loader.iter_dataset(
        read_threads=cfg.read_threads,
        prefetch_buffer_size=min(cfg.prefetch_buffer_size, stop or len(loader)),
        stop=stop,
        show_message_progress=cfg.vbar,
    )
    ds = ds.map(truncate_to_shortest_topic)
    ds = ds.map(map_topic_payloads)
    ds = ds.map(lambda x: filter_noops(x, threshold=cfg.threshold))

    lim = _apply_fd_limit(512**2)
    ds = ds.mp_prefetch(
        grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_buf),
    )
    return add_episode_info(ds)


def main(cfg: MyBuildMGR) -> None:
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
        for i, x in enumerate(tqdm(ds, total=n, desc="Loading episodes", position=0)):
            if cfg.verbose:
                print(f"\n[bold]episode={i}[/bold]")
            print(spec(x))
        return

    total = sum(
        int(x["info"]["len"][0]) for x in tqdm(make_dataset(loader, cfg), total=len(loader), desc="Counting steps")
    )
    print(f"total_steps={total}")

    sam = SamClientWrapper(Client(host=cfg.sam.host, port=cfg.sam.port), cfg=cfg.sam)
    roboreg = Client(host=cfg.reg.host, port=cfg.reg.port)
    dream = Client(host=cfg.dream.host, port=cfg.dream.port)

    ds = make_dataset(loader, cfg)
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)
    ds = ds.map(lambda x: calibrate_extrinsics(x, sam=sam, dream=dream, roboreg=roboreg, cfg=cfg))
    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))
    ds = ds.map(cfg.progress(total))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
