from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
from scipy.spatial.transform import Rotation
import yourdfpy


def make_intr(fx: float, fy: float, w: int, h: int) -> np.ndarray:
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)


def filter_w2c_by_iou(w2c: np.ndarray, iou: np.ndarray, *, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.asarray(w2c, dtype=np.float32)
    iou = np.asarray(iou, dtype=np.float32)
    keep = iou >= threshold
    filtered = w2c.copy()
    filtered[~keep] = np.nan
    return filtered, keep


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
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


def mean_extr(extr: np.ndarray) -> np.ndarray:
    extr = np.asarray(extr, dtype=np.float64)
    q = Rotation.from_matrix(extr[:, :3, :3]).as_quat()
    q_mean = q[0]
    for i in range(1, len(q)):
        q_mean = slerp_quat(q_mean, q[i], 1.0 / (i + 1))

    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rotation.from_quat(q_mean).as_matrix().astype(np.float32)
    out[:3, 3] = extr[:, :3, 3].mean(axis=0).astype(np.float32)
    return out


def roboreg_link_T_optical(dtype: np.dtype) -> np.ndarray:
    return np.array(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )


def dream_w2c_cv_to_roboreg_ht(w2c_cv: np.ndarray) -> np.ndarray:
    link_T_optical = roboreg_link_T_optical(w2c_cv.dtype)
    return np.linalg.inv(w2c_cv) @ np.linalg.inv(link_T_optical)


def roboreg_ht_to_dream_w2c_cv(ht: np.ndarray) -> np.ndarray:
    link_T_optical = roboreg_link_T_optical(ht.dtype)
    return np.linalg.inv(link_T_optical) @ np.linalg.inv(ht)


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
    **{key: key for key in keypoint2id if key.startswith("link")},
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
        self.link_index = {name: i for i, name in enumerate(self.robot.links.names)}
        self.actuated = len(self.urdf.actuated_joints)
        self.drive_joint_index = next(
            (i for i, joint in enumerate(self.urdf.actuated_joints) if joint.name == "drive_joint"),
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


def robot_keypoints_in_cameras(
    joints: np.ndarray,
    gripper: np.ndarray,
    w2c: np.ndarray,
    *,
    urdf_path: Path,
    mesh_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = RobotKeypoints(urdf_path, mesh_dir)
    kp3dw = robot.fk(np.asarray(joints, dtype=np.float32), gripper=np.asarray(gripper, dtype=np.float32))
    kp3dc, kp3dc_mask = project_world_to_cam(kp3dw, np.asarray(w2c, dtype=np.float32))
    return kp3dw, kp3dc, kp3dc_mask
