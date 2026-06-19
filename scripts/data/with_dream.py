"""
uv run scripts/data/with_dream.py \
--sam.port 8080 --reg.port 8022 --dream.port 8085 --name xgym_lift_single --version 0.5.12 --sam.confidence 0.3 --n 32
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import time
from typing import Literal, Protocol

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
from grain.experimental import ThreadPrefetchIterDataset
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyroki as pk
from rich import print
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import tyro
from webpolicy import msgpack_numpy
from webpolicy.client import Client
import websockets.sync.client
import yourdfpy

from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.datasets import EpisodeArrayRecordSource, stack
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.write import BuildMGR
from crossformer.utils.autobox import Box
from crossformer.utils.spec import spec
import wandb


@dataclass
class Endpoint:
    host: str
    port: int


@dataclass
class Sam(Endpoint):
    prompt: str = "robot"  # SAM3 text prompt
    confidence: float = 0.3  # SAM3 confidence threshold

    # raw_webpolicy: bool = True  # Send raw payloads for older SAM3 webpolicy servers

    close_kernel_size: int = 3  # Morphological close kernel; 0 disables
    min_component_area: int = 16  # Remove tiny mask islands; 0 disables


@dataclass
class Roboreg(Endpoint):
    pass


@dataclass
class Dream(Endpoint):
    units: Literal["deg", "rad"] = "deg"  # DREAM server converts deg to rad internally


def make_intr(fx, fy, w, h):
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])


@dataclass
class MyBuildMGR(BuildMGR):
    take: int | None = None  # debug. take n steps
    n: int = 32  # number of frames to use for registration
    min_iou: float = 0.75  # keep calibrated w2c only above this registration IoU
    urdf: Path = Path("xarm7_standalone.urdf")
    mesh_dir: Path | None = Path("assets")

    image_size: int = 200  # Square size for SAM, Dream, and DR
    fxy: float = 515.0  # Focal length for DR depth-to-3D conversion
    mask_area: list[float] = default([0.005, 0.8])  # Reject tiny/large masks

    dr: bool = True  # whether to use sam+roboreg or only dream

    sam: Sam = default(Sam(host="localhost", port=8080))
    reg: Roboreg = default(Roboreg(host="localhost", port=8081))
    dream: Dream = default(Dream(host="localhost", port=8082))

    def __post_init__(self):
        super().__post_init__()
        assert self.name in Arec.REGISTRY, f"mix should be one of {list(Arec.REGISTRY.keys())}"

    @property
    def mix(self):
        return Arec.REGISTRY[self.name]


class ClientLike(Protocol):
    def step(self, payload: dict) -> dict: ...


def wandb_image(x):
    x = np.asarray(x)
    if x.dtype == bool:
        x = x.astype(np.uint8) * 255
    elif x.dtype != np.uint8:
        hi = float(np.nanmax(x)) if x.size else 0.0
        x = (x * 255.0 if hi <= 1.0 else x).clip(0, 255).astype(np.uint8)
    return wandb.Image(x)


def image_u8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    hi = float(np.nanmax(x)) if x.size else 0.0
    return (x * 255.0 if hi <= 1.0 else x).clip(0, 255).astype(np.uint8)


def overlay_mask(image, mask):
    image = np.asarray(image)
    mask = np.asarray(mask).reshape(image.shape[:2]).astype(bool)
    out = image.copy()
    out[mask] = (0.45 * out[mask] + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
    return out


def maybe_log_registration(view: int, out, *, step: int = 0, limit: int = 4):
    if wandb.run is None:
        return
    # if "render_overlays" not in out:
    # return
    log = {}
    for k in out:
        xs = np.asarray(out[k])
        for i in range(min(len(xs), limit)):
            log[f"registration/cam_{view}/{k}_{i:02d}"] = wandb_image(xs[i])
        wandb.log(log, step=step)


def maybe_log_registration_triplet(view: int, out, *, step: int = 0, limit: int = 9):
    if wandb.run is None:
        return

    panels = []
    if "first" in out and "both" in out.first:
        panels.append(np.asarray(out.first.both))
    if "rast" in out:
        panels.append(np.asarray(out.rast))
    elif "first" in out and "rast" in out.first:
        panels.append(np.asarray(out.first.rast))
    if "both" in out:
        panels.append(np.asarray(out.both))
    if not panels:
        return

    # write the iou on each panel if available
    def write_iou(panel):
        if "iou" not in out:
            return panel
        iou = float(out.iou)
        panel = panel.copy()
        for i in range(len(panel)):
            im = Image.fromarray(panel[i])
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            draw.text((10, 10), f"IoU: {iou:.2f}", fill=(255, 0, 0), font=font)
            panel[i] = np.asarray(im)
        return panel

    panels = [write_iou(panel) for panel in panels]

    rows = [
        np.concatenate([panel[i] for panel in panels], axis=1)
        for i in range(min(limit, *(len(panel) for panel in panels)))
    ]
    if rows:
        panel = np.concatenate(rows, axis=0)
        wandb.log({f"registration/cam_{view}/first_both_rast_both": wandb_image(panel)}, step=step)


def cam_to_px(kp3dc: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kp3dc = np.asarray(kp3dc, dtype=np.float32)
    z = kp3dc[..., 2]
    valid = np.isfinite(kp3dc).all(axis=-1) & (z > 1e-6)
    uv = np.full((*kp3dc.shape[:-1], 2), np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv[..., 0] = K[0, 0] * kp3dc[..., 0] / z + K[0, 2]
        uv[..., 1] = K[1, 1] * kp3dc[..., 1] / z + K[1, 2]
    uv[~valid] = np.nan
    return uv, valid


def draw_kp2d(image: np.ndarray, kp2d: np.ndarray, valid: np.ndarray, *, radius: int = 3) -> np.ndarray:
    colors = [
        (255, 64, 64),
        (64, 220, 64),
        (64, 128, 255),
        (255, 220, 64),
        (255, 64, 220),
        (64, 220, 220),
    ]
    out = image_u8(image).copy()
    h, w = out.shape[:2]
    im = Image.fromarray(out)
    draw = ImageDraw.Draw(im)
    for i, (u, v) in enumerate(kp2d):
        if not valid[i] or not np.isfinite([u, v]).all():
            continue
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        color = colors[i % len(colors)]
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(0, 0, 0))
    return np.asarray(im)


def maybe_log_kp3dc_projection(x, cfg: MyBuildMGR, *, step: int = 0, limit: int = 9):
    if wandb.run is None:
        print("skip kp3dc projection log: wandb.run is None")
        return
    x = Box(x)
    image = np.asarray(x.image)
    kp3dc = np.asarray(x.proprio.kp3dc_robot)
    h, w = image.shape[2:4]
    K = make_intr(fx=cfg.fxy, fy=cfg.fxy, w=w, h=h)
    kp2dc, valid = cam_to_px(kp3dc, K)

    n = min(limit, image.shape[0])
    rows = []
    for t in range(n):
        panels = [draw_kp2d(image[t, v], kp2dc[t, v], valid[t, v]) for v in range(image.shape[1])]
        rows.append(np.concatenate(panels, axis=1))
    if rows:
        panel = np.concatenate(rows, axis=0)
        print(f"log kp3dc projection panel: step={step} shape={panel.shape}")
        wandb.log({"registration/kp3dc_robot_projection": wandb_image(panel)}, step=step)


def episode_step(x) -> int:
    return int(np.asarray(x.info.id.episode).reshape(-1)[0])


def filter_w2c_by_iou(w2c, iou, *, threshold: float):
    w2c = np.asarray(w2c, dtype=np.float32)
    iou = np.asarray(iou, dtype=np.float32)
    keep = iou >= threshold
    filtered = w2c.copy()
    filtered[~keep] = np.nan
    return filtered, keep


def slerp_quat(q0, q1, t: float):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = (1.0 - t) * q0 + t * q1
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta


def mean_extr(extr):
    extr = np.asarray(extr, dtype=np.float64)
    q = Rotation.from_matrix(extr[:, :3, :3]).as_quat()
    q_mean = q[0]
    for i in range(1, len(q)):
        q_mean = slerp_quat(q_mean, q[i], 1.0 / (i + 1))

    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rotation.from_quat(q_mean).as_matrix().astype(np.float32)
    out[:3, 3] = extr[:, :3, 3].mean(axis=0).astype(np.float32)
    return out


class SamClientWrapper:
    def __init__(self, client: Client, cfg: Sam):
        self.client = client
        self.cfg = cfg

    def step(self, image: np.ndarray) -> np.ndarray:
        """Segment the image using SAM and return a mask."""
        h, w, c = image.shape  # noqa
        payload = {
            "image": image,
            "type": "image",
            "text": self.cfg.prompt,
            "confidence": self.cfg.confidence,
        }
        out = self.client.step(payload)

        valid = np.prod(out["masks"].shape) > 0
        return {
            "seg": out["masks"].any(axis=0).reshape(h, w, 1) if valid else np.zeros((h, w, 1)).astype(bool),
            "valid": valid,
        }


def do_segmentation(x: dict, sam: ClientLike, valid: bool | list[bool] = True):
    """Run SAM segmentation on all images"""
    valid = np.full((len(x["image"]),), valid, dtype=bool) if isinstance(valid, bool) else np.asarray(valid, dtype=bool)

    def dummy(im: np.ndarray) -> dict:
        h, w, c = im.shape  # noqa
        return {"seg": np.zeros((h, w, 1)).astype(bool), "valid": False}

    seg = [sam.step(im) if v else dummy(im) for im, v in zip(x["image"], valid)]
    seg = stack(seg)  # check all seg outputs have same shape

    # some masks are not valid and have shape (0,*) but this is handled in part by the client wrapper
    x["seg"] = seg["seg"]
    x = Box(x)
    # x['mask']['obs']['seg'] = seg['valid']  # add seg validity to mask obs
    x.auto().mask.obs.seg = seg["valid"]  # add seg validity to mask obs

    return x


class TimeoutClient(Client):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        *,
        open_timeout: float | None = 10,
        ping_interval: float | None = 20,
        ping_timeout: float | None = 20,
        close_timeout: float | None = 10,
    ) -> None:
        self.open_timeout = open_timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        super().__init__(host=host, port=port)

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=self.open_timeout,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=self.close_timeout,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)


# long = TimeoutClient( host=cfg.reg.host, port=cfg.reg.port, ping_interval=60, ping_timeout=300,)


def do_dream(x: dict, dream: ClientLike, cfg: MyBuildMGR):
    x = Box(x)
    K = make_intr(fx=cfg.fxy, fy=cfg.fxy, w=x.image.shape[2], h=x.image.shape[1])
    # TODO resize images according to cfg

    # image = [_shrink_crop_image_np(i, cfg.image_size, cfg.image_size, Image.BILINEAR) for i in x.image]
    # image = np.stack(image, axis=1)
    q = np.asarray(x.proprio.joints, dtype=np.float32)
    q = np.rad2deg(q) if cfg.dream.units == "deg" else q

    payload = {
        "image": x.image,  # expects list of images
        "K": K,  # camera intrinsics for depth-to-3D conversion
        "q": q,  # DREAM server units are controlled by cfg.dream.units
        "type": "image",
        "calibrate": True,  # ???
    }
    out = Box(dream.step(payload))

    valid = (
        out.pnp_success
        & np.isfinite(out.w2c).all(axis=(1, 2))
        & (out.pnp_valid.sum(axis=1) >= 5)
        & np.isfinite(out.pnp_reproj_px)
        & (out.pnp_reproj_px < 20.0)  # maybe 30.0 if noisy
    )
    nan = np.full((4, 4), np.nan, dtype=np.float32)
    out.w2c = out.w2c.copy()
    out.w2c[~valid] = nan

    x.auto().extr.w2c = out.w2c
    x.auto().extr.w2c_cv = out.w2c  # TODO im assuming its cv convention. need to verify
    x.auto().mask.extr.w2c = valid
    return x


"""
  The _T_ notation and a2b notation name the same transform from opposite grammatical angles.

  A_T_B:  destination_T_source
  B2A:    source-to-destination

  A_T_B maps coordinates expressed in B into coordinates expressed in A
  a2b maps from A to B

  base_T_camera means:
  a transform that takes coordinates in camera frame and maps into base frame
  base_T_camera == c2w camera_T_base == w2c
"""


def roboreg_link_T_optical(dtype):
    # roboreg's camera_link_T_optical / ht_optical
    return np.array(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )


def dream_w2c_cv_to_roboreg_ht(w2c_cv):
    link_T_optical = roboreg_link_T_optical(w2c_cv.dtype)

    # Dream: optical_T_base (OpenCV w2c)
    # Roboreg wants: base_T_camera_link
    return np.linalg.inv(w2c_cv) @ np.linalg.inv(link_T_optical)


def roboreg_ht_to_dream_w2c_cv(HT):
    link_T_optical = roboreg_link_T_optical(HT.dtype)

    # Roboreg: base_T_camera_link
    # Dream: optical_T_base (OpenCV w2c)
    return np.linalg.inv(link_T_optical) @ np.linalg.inv(HT)


keypoint2id: dict[str, int] = {
    "link_base": 0,
    "link1": 1,
    "link2": 2,
    "link3": 3,
    "link4": 4,
    "link5": 5,
    "link6": 6,
    "link7": 7,
    "link_eef": 8,
    "link_tcp": 9,
    "left_finger_joint": 10,
    "right_finger_joint": 11,
    "left_finger_tip": 12,
    "right_finger_tip": 13,
}

keypoint2link: dict[str, str] = {
    **{k: k for k in keypoint2id if k.startswith("link")},
    "left_finger_joint": "left_finger",
    "right_finger_joint": "right_finger",
    "left_finger_tip": "left_finger",
    "right_finger_tip": "right_finger",
}

keypoint_offsets: dict[str, np.ndarray] = {
    "left_finger_tip": np.array([0.000133, -0.023070, 0.059904], dtype=np.float32),
    "right_finger_tip": np.array([-0.000207, 0.023057, 0.059904], dtype=np.float32),
}


def _quat_to_rotmat(q: jax.Array) -> jax.Array:
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return jnp.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        axis=-1,
    ).reshape((*q.shape[:-1], 3, 3))


def _poses_to_mats(poses: jax.Array) -> jax.Array:
    eye = jnp.broadcast_to(jnp.eye(4, dtype=poses.dtype), (*poses.shape[:-1], 4, 4)).copy()
    eye = eye.at[..., :3, :3].set(_quat_to_rotmat(poses[..., :4]))
    eye = eye.at[..., :3, 3].set(poses[..., 4:])
    return eye


class RobotKeypoints:
    def __init__(self, urdf_path: Path, mesh_dir: Path | None):
        mesh_dir = urdf_path.parent if mesh_dir is None else mesh_dir
        self.urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.link_index = {n: i for i, n in enumerate(self.robot.links.names)}
        self.actuated = len(self.urdf.actuated_joints)
        self.drive_joint_index = next(
            (i for i, j in enumerate(self.urdf.actuated_joints) if j.name == "drive_joint"),
            None,
        )
        self.drive_joint_limits = self._drive_joint_limits()
        self._fk = jax.jit(self.robot.forward_kinematics)

    def _drive_joint_limits(self) -> tuple[float, float] | None:
        if self.drive_joint_index is None:
            return None
        limit = self.urdf.actuated_joints[self.drive_joint_index].limit
        return float(limit.lower), float(limit.upper)

    def _drive_from_gripper(self, gripper: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if self.drive_joint_limits is None:
            raise ValueError("gripper supplied but URDF has no drive_joint")
        lo, hi = self.drive_joint_limits
        g = np.asarray(gripper, dtype=np.float32)
        if g.ndim > 0 and g.shape[-1] == 1:
            g = np.squeeze(g, axis=-1)
        g = np.clip(g, 0.0, 1.0)
        drive = lo + (1.0 - g) * (hi - lo)
        return np.broadcast_to(drive, shape).astype(np.float32)

    def _q(self, joints: np.ndarray, gripper: np.ndarray | None = None) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float32).copy()
        if q.shape[-1] > self.actuated:
            raise ValueError(f"joint dim {q.shape[-1]} > actuated {self.actuated}")
        if q.shape[-1] < self.actuated:
            pad = [(0, 0)] * q.ndim
            pad[-1] = (0, self.actuated - q.shape[-1])
            q = np.pad(q, pad)
        if gripper is not None:
            if self.drive_joint_index is None:
                raise ValueError("gripper supplied but URDF has no drive_joint")
            q[..., self.drive_joint_index] = self._drive_from_gripper(gripper, q.shape[:-1])
        return q

    def fk(self, joints: np.ndarray, gripper: np.ndarray | None = None) -> np.ndarray:
        q = self._q(joints, gripper)
        shape = q.shape[:-1]
        q = q.reshape(-1, q.shape[-1])
        poses = self._fk(jnp.asarray(q))
        mats = np.asarray(_poses_to_mats(poses), dtype=np.float32).reshape(*shape, -1, 4, 4)
        pts = []
        for name in sorted(keypoint2id, key=keypoint2id.__getitem__):
            mat = mats[..., self.link_index[keypoint2link[name]], :, :]
            offset = keypoint_offsets.get(name)
            if offset is None:
                pts.append(mat[..., :3, 3])
            else:
                tip = np.einsum("...ij,j->...i", mat[..., :3, :3], offset)
                pts.append(tip + mat[..., :3, 3])
        return np.stack(pts, axis=-2)


def project_world_to_cam(kp3dw: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.asarray(w2c, dtype=np.float32)
    valid = np.isfinite(w2c).all(axis=(-2, -1))
    ones = np.ones((*kp3dw.shape[:-1], 1), dtype=kp3dw.dtype)
    kp3dw_h = np.concatenate([kp3dw, ones], axis=-1)
    kp3dc = np.einsum("tvij,tkj->tvki", w2c, kp3dw_h)[..., :3].astype(np.float32)
    kp3dc[~valid] = np.nan
    mask = np.repeat(valid[..., None], kp3dw.shape[-2], axis=-1)
    return kp3dc, mask


def apply_extrinsics(x, cfg: MyBuildMGR):
    x = Box(x)
    robot = RobotKeypoints(cfg.urdf, cfg.mesh_dir)
    kp3dw = robot.fk(
        np.asarray(x.proprio.joints, dtype=np.float32),
        gripper=np.asarray(x.proprio.gripper, dtype=np.float32),
    )
    kp3dc, kp3dc_mask = project_world_to_cam(kp3dw, np.asarray(x.extr.w2c, dtype=np.float32))

    x.auto().proprio.kp3dw_robot = kp3dw
    x.auto().proprio.kp3dc_robot = kp3dc
    x.auto().mask.proprio.kp3dw_robot = np.ones(kp3dw.shape[:-1], dtype=bool)
    x.auto().mask.proprio.kp3dc_robot = kp3dc_mask
    return x


def batch_registration(x: dict, roboreg: ClientLike, cfg: MyBuildMGR):
    # TODO filter out bad SAM masks according to cfg.mask_area before registration

    # pick_best_w2c
    # filter_dr_frames

    w2cs = []
    ious = []
    for i in range(x.image.shape[1]):  # loop over views in T,V,HWC
        img, seg, extr = x.image[:, i], x.seg[:, i], x.extr.w2c[:, i]

        if not x.mask.extr.w2c[:, i].any():  # skip if not any valid DREAM w2c
            print(f"skipping registration for frame {i} due to invalid DREAM w2c")
            w2cs.append(np.full((4, 4), np.nan, dtype=np.float32))
            ious.append(0.0)
            continue
        else:
            valid = x.mask.extr.w2c[:, i]
            img, seg, extr, joints = img[valid], seg[valid], extr[valid], x.proprio.joints[valid]

        # use T,HW not T,HW1
        T, H, W, _C = img.shape
        seg = seg.reshape(T, H, W).astype(int) * 255

        payload = {
            "depth": img[..., 0],  # dummy
            "joints": joints,
            "mask": seg,
            "intrinsics": make_intr(fx=cfg.fxy, fy=cfg.fxy, w=W, h=H),
            "HT": dream_w2c_cv_to_roboreg_ht(mean_extr(extr)),
            "mode": "dr",  # Literal['icp', 'dr', 'both']
        }
        # print(spec(payload))
        print(spec(payload))
        # print(seg.mean(), seg.dtype, seg.max(), seg.min())
        raw_out = roboreg.step(payload)
        # roboreg returns base_T_cam_link; DREAM/OpenCV wants cam_optical_T_base

        print(f"roboreg response keys: {sorted(raw_out)}")

        out = Box(raw_out)
        print(spec(out))
        print(out.keys())
        print(out.iou)
        if out.iou > cfg.min_iou:
            # maybe_log_registration(i, {k:v for k,v in out.items() if isinstance(v, np.ndarray) and len(v.shape)==4})
            maybe_log_registration_triplet(i, out)
            w2c = roboreg_ht_to_dream_w2c_cv(np.asarray(out.HT, dtype=np.float32))
        else:
            w2c = np.full((4, 4), np.nan, dtype=np.float32)
            print(w2c)

        w2cs.append(w2c)
        ious.append(float(out.iou))

    w2c, keep = filter_w2c_by_iou(np.stack(w2cs), np.asarray(ious), threshold=cfg.min_iou)
    return {"w2c": w2c, "iou": np.asarray(ious, dtype=np.float32), "valid": keep}


def select_registration_frames(x: dict, n: int = 32) -> list[dict]:
    """Select episode frames for calibration, preserving the camera axis."""
    t = len(x["info"]["id"]["episode"])
    k = min(n, t)
    idx = np.linspace(0, t - 1, k, dtype=np.int32)
    print(idx)
    return [jax.tree.map(lambda y: y[i], x) for i in idx]


def calibrate_extr(x, sam, dream, roboreg, cfg):
    reg = select_registration_frames(x, n=cfg.n)
    reg = [do_dream(reg, dream, cfg) for reg in tqdm(reg, desc="DREAM calibration")]
    reg = [do_segmentation(reg, sam, reg.mask.extr.w2c) for reg in tqdm(reg, desc="SAM segmentation")]
    reg = Box(stack([r.dict for r in reg]))
    # print(spec(reg.dict))
    registration = batch_registration(reg, roboreg, cfg)
    print(registration)

    t = len(x["info"]["id"]["episode"])
    x = Box(x)
    x.auto().extr.w2c = np.repeat(registration["w2c"][None], t, axis=0)
    x.auto().mask.extr.w2c = np.repeat(registration["valid"][None], t, axis=0)
    x.auto().info.reg.iou = np.repeat(registration["iou"][None], t, axis=0)
    x = apply_extrinsics(x, cfg)
    x["info"].pop("image_keys")
    maybe_log_kp3dc_projection(x, cfg, step=episode_step(x))

    print(spec(x.dict))
    # quit()
    return x.dict


def main(cfg: MyBuildMGR):
    wandb.init(
        project="with-dream", name=f"{cfg.name}-{cfg.version}", config={"name": cfg.name, "version": cfg.version}
    )

    sam = SamClientWrapper(Client(host=cfg.sam.host, port=cfg.sam.port), cfg=cfg.sam)
    roboreg = Client(host=cfg.reg.host, port=cfg.reg.port)
    dream = Client(host=cfg.dream.host, port=cfg.dream.port)

    eps = EpisodeArrayRecordSource.from_mix(cfg.mix)
    ds = grain.MapDataset.source(eps)

    # these already have eid
    # ds = ds.map(init_info).map(add_traj_len).map(add_step_id).map_with_index(add_episode_id)

    # materialize to compute total steps for progress bar
    # total, n = sum([x["info"]["len"][0] for x in tqdm(ds, desc="compute total")]), len(ds)
    total = len(cfg.mix.source)
    # print(f"total steps: {total} across {n} episodes")

    if cfg.take:  # debug
        dsit = iter(ds)
        ds = grain.MapDataset.source([next(dsit) for _ in range(cfg.take)])

    # force clients runs serially
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)

    ds = ds.map(partial(calibrate_extr, sam=sam, dream=dream, roboreg=roboreg, cfg=cfg))

    # ds = ds.map(partial(do_segmentation, sam=sam))
    # ds = ds.map(partial(do_dream, dream=dream, cfg=cfg))
    # ds = ds.map(partial(do_registration, roboreg=roboreg, cfg=cfg)) if cfg.dr else ds

    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))

    # dsit = iter(ds)
    # valids = []
    # for i in tqdm(range(5000)):
    # x = next(dsit)
    # print()

    ds = ds.map(cfg.progress(total))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
