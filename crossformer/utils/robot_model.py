"""Lightweight xArm robot mesh and FK helpers."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

LANDMARK_LINKS = (
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "link_eef",
    "link_tcp",
)


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


def poses_to_mats(poses: jax.Array) -> jax.Array:
    """qw,qx,qy,qz,tx,ty,tz -> 4x4 homogeneous matrices."""
    eye = jnp.broadcast_to(jnp.eye(4, dtype=poses.dtype), (*poses.shape[:-1], 4, 4)).copy()
    eye = eye.at[..., :3, :3].set(_quat_to_rotmat(poses[..., :4]))
    eye = eye.at[..., :3, 3].set(poses[..., 4:])
    return eye


class RobotMesh:
    """Pre-computed mesh buffers and batched FK via pyroki."""

    def __init__(self, urdf_path: Path, mesh_dir: Path | None = None):
        mesh_dir = mesh_dir if mesh_dir is not None else urdf_path.parent
        self.urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.link_index = {n: i for i, n in enumerate(self.robot.links.names)}
        self.joint_names = tuple(j.name for j in self.urdf.actuated_joints)
        self.joint_limits = tuple((float(j.limit.lower), float(j.limit.upper)) for j in self.urdf.actuated_joints)
        self.actuated = len(self.urdf.actuated_joints)
        self.verts, self.faces, self.link_ids = self._extract_meshes()
        self._fk = jax.jit(self.robot.forward_kinematics)

    def _link_name(self, node: str) -> str:
        base = node.rsplit(".", 1)[0]
        if base in self.link_index:
            return base
        world_mesh = self.urdf.scene.graph.get(node)[0]
        return min(
            self.link_index,
            key=lambda n: np.abs(np.linalg.inv(self.urdf.scene.graph.get(n)[0]) @ world_mesh - np.eye(4)).sum(),
        )

    def _extract_meshes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        verts, faces, links = [], [], []
        offset = 0
        for node in self.urdf.scene.graph.nodes_geometry:
            link = self._link_name(node)
            geom_name = self.urdf.scene.graph[node][1]
            geom = self.urdf.scene.geometry[geom_name]

            world_mesh = self.urdf.scene.graph.get(node)[0]
            world_link = self.urdf.scene.graph.get(link)[0]
            link_to_mesh = np.linalg.inv(world_link) @ world_mesh
            v = np.asarray(geom.vertices, dtype=np.float32)
            v_h = np.concatenate([v, np.ones((len(v), 1), dtype=np.float32)], axis=1)
            v_link = (link_to_mesh @ v_h.T).T.astype(np.float32)
            verts.append(v_link)
            faces.append(np.asarray(geom.faces, dtype=np.int32) + offset)
            links.append(np.full(len(v_link), self.link_index[link], dtype=np.int32))
            offset += len(v_link)
        return (
            np.concatenate(verts).astype(np.float32),
            np.concatenate(faces).astype(np.int32),
            np.concatenate(links).astype(np.int32),
        )

    def pad_cfg(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        if q.shape[-1] == self.actuated:
            return q
        pad = self.actuated - q.shape[-1]
        if pad < 0:
            raise ValueError(f"joint dim {q.shape[-1]} > actuated {self.actuated}")
        return np.pad(q, ((0, 0), (0, pad)))

    def posed_verts(self, q: np.ndarray) -> np.ndarray:
        """(B, A) joints -> (B, V, 4) world-frame vertices."""
        poses = self._fk(jnp.asarray(self.pad_cfg(q)))
        mats = np.asarray(poses_to_mats(poses))
        link_mats = mats[:, self.link_ids]
        v = np.broadcast_to(self.verts[None], (len(q), *self.verts.shape))
        return np.einsum("bvij,bvj->bvi", link_mats, v)


def get_robot_mesh() -> RobotMesh:
    if not hasattr(get_robot_mesh, "_cache"):
        get_robot_mesh._cache = RobotMesh(Path("xarm7_standalone.urdf"), Path("assets"))
    return get_robot_mesh._cache


def fk_keypoints(joints_rad: np.ndarray, robot: RobotMesh | None = None) -> np.ndarray:
    """Run FK and extract 3D positions for the 10 landmarks."""
    robot = get_robot_mesh() if robot is None else robot
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    poses = robot._fk(jnp.asarray(q))
    mats = np.asarray(poses_to_mats(poses))[0]
    pts = [mats[robot.link_index[name], :3, 3] for name in LANDMARK_LINKS]
    return np.stack(pts)


__all__ = ["LANDMARK_LINKS", "RobotMesh", "fk_keypoints", "get_robot_mesh", "poses_to_mats"]
