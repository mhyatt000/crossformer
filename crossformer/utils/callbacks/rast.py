"""Callback that rasterizes a URDF robot from camera viewpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
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
# Vectorised software rasterizer
# ---------------------------------------------------------------------------


def _perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (zfar + znear) / (znear - zfar)
    P[2, 3] = 2.0 * zfar * znear / (znear - zfar)
    P[3, 2] = -1.0
    return P


def _rasterize_mask(
    verts_clip: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Vectorised silhouette rasterizer — no Python triangle loop.

    Projects all triangles, bins pixels via edge functions in parallel.
    For silhouettes we only need a binary mask, no z-buffer needed.
    """
    w = verts_clip[:, 3]
    ok = np.abs(w) > 1e-7
    ndc = np.zeros((len(verts_clip), 3), dtype=np.float32)
    ndc[ok] = verts_clip[ok, :3] / w[ok, None]

    # NDC --> pixel coords for all vertices
    px = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)

    # Gather triangle vertex pixel coords: (F, 3) each
    x0, x1, x2 = px[faces[:, 0]], px[faces[:, 1]], px[faces[:, 2]]
    y0, y1, y2 = py[faces[:, 0]], py[faces[:, 1]], py[faces[:, 2]]

    # Filter degenerate and off-screen triangles
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    tri_xmin = np.minimum(np.minimum(x0, x1), x2)
    tri_xmax = np.maximum(np.maximum(x0, x1), x2)
    tri_ymin = np.minimum(np.minimum(y0, y1), y2)
    tri_ymax = np.maximum(np.maximum(y0, y1), y2)
    keep = (np.abs(area2) > 1e-6) & (tri_xmax >= 0) & (tri_xmin < width) & (tri_ymax >= 0) & (tri_ymin < height)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return np.zeros((height, width), dtype=np.float32)

    x0, x1, x2 = x0[idx], x1[idx], x2[idx]
    y0, y1, y2 = y0[idx], y1[idx], y2[idx]
    F = len(idx)

    # Process triangles in chunks to limit memory
    mask = np.zeros((height, width), dtype=np.float32)
    chunk = max(1, min(F, 500))
    for start in range(0, F, chunk):
        end = min(start + chunk, F)
        _rast_chunk(
            mask,
            x0[start:end],
            y0[start:end],
            x1[start:end],
            y1[start:end],
            x2[start:end],
            y2[start:end],
            width,
            height,
        )
    return mask


def _rast_chunk(
    mask: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Rasterize a chunk of triangles into mask (in-place)."""
    n = len(x0)
    # Per-triangle bounding box
    bx0 = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(np.int32), 0, width - 1)
    bx1 = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(np.int32), 0, width - 1)
    by0 = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(np.int32), 0, height - 1)
    by1 = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(np.int32), 0, height - 1)

    # Max bbox size across chunk --> build grid once
    max_w = int((bx1 - bx0).max()) + 2
    max_h = int((by1 - by0).max()) + 2

    # Pixel offsets within max bbox
    dx = np.arange(max_w, dtype=np.float32) + 0.5  # (W,)
    dy = np.arange(max_h, dtype=np.float32) + 0.5  # (H,)
    gx = bx0[:, None] + dx[None, :]  # (n, W)
    gy = by0[:, None] + dy[None, :]  # (n, H)

    # Edge function for each triangle x pixel: use broadcasting
    # pixel coords: (n, H, W) via outer product
    px = gx[:, None, :]  # (n, 1, W)
    py_grid = gy[:, :, None]  # (n, H, 1)

    # Edge functions (n, H, W)
    e0 = (x1 - x0)[:, None, None] * (py_grid - y0[:, None, None]) - (y1 - y0)[:, None, None] * (px - x0[:, None, None])
    e1 = (x2 - x1)[:, None, None] * (py_grid - y1[:, None, None]) - (y2 - y1)[:, None, None] * (px - x1[:, None, None])
    e2 = (x0 - x2)[:, None, None] * (py_grid - y2[:, None, None]) - (y0 - y2)[:, None, None] * (px - x2[:, None, None])

    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))

    # Clamp to image bounds
    x_valid = (px >= 0) & (px < width)  # (n, 1, W)
    y_valid = (py_grid >= 0) & (py_grid < height)  # (n, H, 1)
    valid = inside & x_valid & y_valid

    # Scatter into mask
    for i in range(n):
        ys, xs = np.where(valid[i])
        if len(ys) == 0:
            continue
        row = by0[i] + ys
        col = bx0[i] + xs
        mask[row, col] = 1.0


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@dataclass
class RastCallback:
    """Render URDF robot silhouettes from camera viewpoints.

    Meshes are decimated at load time for fast software rasterization.
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
    _cam_mats: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _proj: np.ndarray | None = field(default=None, init=False, repr=False)

    def _ensure_init(self) -> None:
        if self._robot is not None:
            return
        self._robot = _RobotMesh(self.urdf, self.mesh_dir)
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

        verts = robot.posed_verts(q)
        bg = np.asarray(self.bg, dtype=np.float32)
        color = np.asarray(self.color, dtype=np.float32)

        frames = []
        for cam in self._cam_mats:
            mvp = proj @ cam
            frame = np.tile(bg[None, None, :], (self.height, self.width, 1))
            for b in range(len(q)):
                clip = verts[b] @ mvp.T
                m = _rasterize_mask(clip, robot.faces, self.width, self.height)
                w = m[..., None] * self.alpha
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

        verts = robot.posed_verts(traj)
        bg = np.asarray(self.bg, dtype=np.float32)
        color = np.asarray(self.color, dtype=np.float32)
        t_count = len(traj)

        frames = []
        for cam in self._cam_mats:
            mvp = proj @ cam
            frame = np.tile(bg[None, None, :], (self.height, self.width, 1))
            for t in range(t_count):
                clip = verts[t] @ mvp.T
                m = _rasterize_mask(clip, robot.faces, self.width, self.height)
                a = self.alpha * (0.3 + 0.7 * t / max(t_count - 1, 1))
                w = m[..., None] * a
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
