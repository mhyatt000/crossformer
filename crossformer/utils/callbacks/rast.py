"""Callback that rasterizes a URDF robot from camera viewpoints.

Uses nvdiffrast for GPU-accelerated rasterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nvdiffrast.torch as dr
import pyroki as pk
import torch
import yourdfpy

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _quat_to_rotmat(q: jax.Array) -> jax.Array:
    """qw,qx,qy,qz --> 3x3 rotation matrix."""
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
    """qw,qx,qy,qz,tx,ty,tz --> 4x4 homogeneous matrices."""
    eye = jnp.broadcast_to(jnp.eye(4, dtype=poses.dtype), (*poses.shape[:-1], 4, 4)).copy()
    eye = eye.at[..., :3, :3].set(_quat_to_rotmat(poses[..., :4]))
    eye = eye.at[..., :3, 3].set(poses[..., 4:])
    return eye


# ---------------------------------------------------------------------------
# Static robot mesh extraction
# ---------------------------------------------------------------------------


class _RobotMesh:
    """Pre-computed mesh buffers + batched FK via pyroki."""

    def __init__(self, urdf_path: Path, mesh_dir: Path | None = None):
        mesh_dir = mesh_dir if mesh_dir is not None else urdf_path.parent
        self.urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.link_index = {n: i for i, n in enumerate(self.robot.links.names)}
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
        """(B, A) joints --> (B, V, 4) world-frame vertices."""
        poses = self._fk(jnp.asarray(self.pad_cfg(q)))
        mats = np.asarray(_poses_to_mats(poses))  # (B, L, 4, 4)
        link_mats = mats[:, self.link_ids]  # (B, V, 4, 4)
        v = np.broadcast_to(self.verts[None], (len(q), *self.verts.shape))
        posed = np.einsum("bvij,bvj->bvi", link_mats, v)
        return posed  # (B, V, 4) — homogeneous


# ---------------------------------------------------------------------------
# GPU rasterizer (nvdiffrast)
# ---------------------------------------------------------------------------


class _GpuRasterizer:
    """Batched GPU silhouette rasterizer via nvdiffrast."""

    def __init__(self, faces: np.ndarray, device: torch.device | None = None):
        self.device = device or torch.device("cuda")
        self.glctx = dr.RasterizeCudaContext(device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.int32, device=self.device)

    def render_masks(self, verts_clip: np.ndarray, width: int, height: int) -> np.ndarray:
        """Rasterize (B, V, 4) clip-space verts to (B, H, W) binary masks."""
        pos = torch.tensor(np.ascontiguousarray(verts_clip), dtype=torch.float32, device=self.device)
        rast_out, _ = dr.rasterize(self.glctx, pos, self.faces, resolution=[height, width])
        masks = (rast_out[:, :, :, 3] > 0).float()
        masks = torch.flip(masks, dims=[1])  # Y-flip to match image convention
        return masks.cpu().numpy()


def _perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (zfar + znear) / (znear - zfar)
    P[2, 3] = 2.0 * zfar * znear / (znear - zfar)
    P[3, 2] = -1.0
    return P


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@dataclass
class RastCallback:
    """Render URDF robot silhouettes from camera viewpoints.

    Uses nvdiffrast for GPU-accelerated rasterization.
    """

    urdf: Path
    cams: list[Path] | None = None
    mesh_dir: Path | None = None
    width: int = 256
    height: int = 256
    fovy: float = 45.0
    color: tuple[float, float, float] = (0.2, 0.4, 0.9)
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha: float = 0.6
    joint_dim: int = 7
    _robot: _RobotMesh | None = field(default=None, init=False, repr=False)
    _rasterizer: _GpuRasterizer | None = field(default=None, init=False, repr=False)
    _cam_mats: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _proj: np.ndarray | None = field(default=None, init=False, repr=False)

    def _ensure_init(self) -> None:
        if self._robot is not None:
            return
        self._robot = _RobotMesh(self.urdf, self.mesh_dir)
        self._rasterizer = _GpuRasterizer(self._robot.faces)
        self._proj = _perspective(self.fovy, self.width / self.height, 0.01, 10.0)
        if self.cams:
            self._cam_mats = [_load_cam(p) for p in self.cams]
        else:
            self._cam_mats = [self._default_cam()]

    def _default_cam(self) -> np.ndarray:
        az, el, r = np.deg2rad(135.0), np.deg2rad(25.0), 1.2
        eye = np.array(
            [
                r * np.cos(el) * np.cos(az),
                r * np.cos(el) * np.sin(az),
                r * np.sin(el),
            ],
            dtype=np.float32,
        )
        return _look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))

    def __call__(self, joints: np.ndarray) -> list[np.ndarray]:
        """Render silhouettes for (B, A) or (B, H, A) joint configs."""
        self._ensure_init()
        robot = self._robot
        proj = self._proj

        q = np.asarray(joints, dtype=np.float32)
        if q.ndim == 3:
            q = q[:, -1]
        q = q[..., : self.joint_dim]

        verts = robot.posed_verts(q)  # (B, V, 4)
        bg = np.asarray(self.bg, dtype=np.float32)
        color = np.asarray(self.color, dtype=np.float32)

        frames = []
        for cam in self._cam_mats:
            mvp = proj @ cam
            # Batched clip-space projection: (B, V, 4)
            clip = np.einsum("bvi,ji->bvj", verts, mvp)
            masks = self._rasterizer.render_masks(clip, self.width, self.height)  # (B, H, W)
            # Composite all batch items onto one frame
            frame = np.tile(bg[None, None, :], (self.height, self.width, 1))
            for b in range(len(q)):
                w = masks[b, :, :, None] * self.alpha
                frame = frame * (1.0 - w) + color[None, None, :] * w
            frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))
        return frames

    def render_trajectory(self, trajectory: np.ndarray) -> list[np.ndarray]:
        """Render trajectory as time-faded ghost overlay."""
        self._ensure_init()
        robot = self._robot
        proj = self._proj

        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim == 3:
            traj = traj[0]
        traj = traj[..., : self.joint_dim]

        verts = robot.posed_verts(traj)  # (T, V, 4)
        bg = np.asarray(self.bg, dtype=np.float32)
        color = np.asarray(self.color, dtype=np.float32)
        t_count = len(traj)

        frames = []
        for cam in self._cam_mats:
            mvp = proj @ cam
            # Batched clip-space projection for all timesteps
            clip = np.einsum("bvi,ji->bvj", verts, mvp)
            masks = self._rasterizer.render_masks(clip, self.width, self.height)  # (T, H, W)
            frame = np.tile(bg[None, None, :], (self.height, self.width, 1))
            for t in range(t_count):
                a = self.alpha * (0.3 + 0.7 * t / max(t_count - 1, 1))
                w = masks[t, :, :, None] * a
                frame = frame * (1.0 - w) + color[None, None, :] * w
            frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))
        return frames


def _load_cam(p: Path) -> np.ndarray:
    """Load a FLU extrinsic and convert to OpenGL view matrix.

    HT is T_base_cam in FLU convention (col0=forward, col1=left, col2=up).
    We extract eye + look direction and build a look_at view matrix.
    """
    p = Path(p)
    HT = np.load(str(p))["HT"].astype(np.float32) if p.suffix == ".npz" else np.load(str(p)).astype(np.float32)
    eye = HT[:3, 3]
    fwd = HT[:3, 0]
    up = HT[:3, 2]
    return _look_at(eye, eye + fwd, up)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack([x, y, z], axis=0)
    view[:3, 3] = -view[:3, :3] @ eye
    return view
