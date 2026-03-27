"""Render a flowed robot trajectory with batched masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import torch
from tqdm.auto import tqdm
import tyro
import yourdfpy


@dataclass
class Config:
    """Config for the raster demo."""

    urdf: Path
    mesh_dir: Path | None = None
    out: Path = Path("tmp_rast.mp4")
    dim: int = 7
    fps: int = 10
    secs: float = 4.0
    seed: int = 7
    n: int = 5
    width: int = 768
    height: int = 768
    radius: float = 2.4
    elev: float = 25.0
    azim: float = 35.0
    scale: float = 1.0
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha: float = 0.3
    color: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tcp_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    line_color: tuple[float, float, float] = (0.0, 0.0, 0.0)


def sample_target(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Sample a known 7D joint distribution."""
    means = np.array(
        [
            [-1.1, -0.8, 0.5, -1.4, 0.3, 1.0, 0.2],
            [0.9, 0.6, -0.7, -0.2, -0.4, 0.8, -0.6],
            [-0.4, 1.0, 0.8, -0.9, 0.7, -0.5, 0.9],
        ],
        dtype=np.float32,
    )
    scales = np.array(
        [
            [0.18, 0.22, 0.20, 0.24, 0.19, 0.17, 0.15],
            [0.20, 0.18, 0.21, 0.17, 0.19, 0.16, 0.18],
            [0.16, 0.20, 0.17, 0.21, 0.18, 0.19, 0.16],
        ],
        dtype=np.float32,
    )
    idx = rng.choice(len(means), size=n, p=np.array([0.35, 0.4, 0.25]))
    eps = rng.normal(size=(n, dim)).astype(np.float32)
    q = means[idx] + scales[idx] * eps
    return np.clip(q, -1.8, 1.8)


def build_flow(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair noise with target samples."""
    rng = np.random.default_rng(cfg.seed)
    q0 = rng.normal(scale=0.55, size=(cfg.n, cfg.dim)).astype(np.float32)
    q1 = sample_target(rng, cfg.n, cfg.dim)
    steps = max(2, round(cfg.fps * cfg.secs))
    ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return q0, q1, ts


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a view matrix."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack([x, y, z], axis=0)
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Build a perspective matrix."""
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (zfar + znear) / (znear - zfar)
    proj[2, 3] = 2.0 * zfar * znear / (znear - zfar)
    proj[3, 2] = -1.0
    return proj


def quat_to_rotmat(q: jax.Array) -> jax.Array:
    """Convert qw,qx,qy,qz to a rotation matrix."""
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return jnp.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        axis=-1,
    ).reshape((*q.shape[:-1], 3, 3))


def poses_to_mats(poses: jax.Array) -> jax.Array:
    """Convert qw,qx,qy,qz,tx,ty,tz poses to 4x4 matrices."""
    mats = jnp.broadcast_to(jnp.eye(4, dtype=poses.dtype), (*poses.shape[:-1], 4, 4))
    mats = mats.at[..., :3, :3].set(quat_to_rotmat(poses[..., :4]))
    mats = mats.at[..., :3, 3].set(poses[..., 4:])
    return mats


class StaticRobot:
    """Static mesh tensors plus batched FK."""

    def __init__(self, cfg: Config):
        mesh_dir = cfg.mesh_dir if cfg.mesh_dir is not None else cfg.urdf.parent
        self.urdf = yourdfpy.URDF.load(cfg.urdf, mesh_dir=mesh_dir)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.link_index = {name: i for i, name in enumerate(self.robot.links.names)}
        self.tcp_link = self._find_tcp_link()
        self.actuated = len(self.urdf.actuated_joints)
        verts, faces, links = self._build_mesh_tensors()
        self.verts = verts
        self.faces = faces
        self.links = links
        self._fk = jax.jit(self.robot.forward_kinematics)

    def _find_tcp_link(self) -> str:
        for name in ("link_tcp", "link_eef", "xarm_gripper_base_link", "link7"):
            if name in self.link_index:
                return name
        return self.robot.links.names[-1]

    def _link_name(self, node: str) -> str:
        base = node.rsplit(".", 1)[0]
        if base in self.link_index:
            return base
        world_mesh = self.urdf.scene.graph.get(node)[0]
        best = min(
            self.link_index,
            key=lambda name: np.abs(np.linalg.inv(self.urdf.scene.graph.get(name)[0]) @ world_mesh - np.eye(4)).sum(),
        )
        return best

    def _build_mesh_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        verts: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        links: list[np.ndarray] = []
        offset = 0
        for node in self.urdf.scene.graph.nodes_geometry:
            link = self._link_name(node)
            geom_name = self.urdf.scene.graph[node][1]
            geom = self.urdf.scene.geometry[geom_name].copy()
            world_mesh = self.urdf.scene.graph.get(node)[0]
            world_link = self.urdf.scene.graph.get(link)[0]
            link_to_mesh = np.linalg.inv(world_link) @ world_mesh
            v = np.asarray(geom.vertices, dtype=np.float32)
            v_h = np.concatenate([v, np.ones((len(v), 1), dtype=np.float32)], axis=1)
            v_link = np.asarray((link_to_mesh @ v_h.T).T, dtype=np.float32)
            verts.append(v_link)
            faces.append(np.asarray(geom.faces, dtype=np.int32) + offset)
            links.append(np.full(len(v_link), self.link_index[link], dtype=np.int32))
            offset += len(v_link)
        return (
            np.concatenate(verts).astype(np.float32, copy=False),
            np.concatenate(faces).astype(np.int32, copy=False),
            np.concatenate(links).astype(np.int32, copy=False),
        )

    def pad_cfg(self, q: np.ndarray) -> np.ndarray:
        """Pad joints to the URDF's actuated count."""
        q = np.asarray(q, dtype=np.float32)
        if q.shape[-1] == self.actuated:
            return q
        pad = self.actuated - q.shape[-1]
        if pad < 0:
            raise ValueError(f"joint dim {q.shape[-1]} > actuated joints {self.actuated}")
        return np.pad(q, ((0, 0), (0, pad)))

    def link_mats(self, q: np.ndarray) -> np.ndarray:
        """Compute batched link transforms."""
        poses = self._fk(jnp.asarray(self.pad_cfg(q), dtype=jnp.float32))
        return np.asarray(poses_to_mats(poses), dtype=np.float32)

    def posed_verts(self, q: np.ndarray) -> np.ndarray:
        """Transform static vertices with batched FK."""
        mats = self.link_mats(q)[:, self.links]
        verts = np.broadcast_to(self.verts[None], (len(q), *self.verts.shape))
        posed = np.einsum("bvij,bvj->bvi", mats, verts, dtype=np.float32)
        return np.asarray(posed[..., :3], dtype=np.float32)

    def tcp_points(self, q: np.ndarray) -> np.ndarray:
        """Return batched tool center points in world space."""
        mats = self.link_mats(q)
        return np.asarray(mats[:, self.link_index[self.tcp_link], :3, 3], dtype=np.float32)

    def bounds(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute batched mesh bounds."""
        verts = self.posed_verts(q)
        return verts.min(axis=(0, 1)), verts.max(axis=(0, 1))


class NVDiffRasterizer:
    """Batched mask rasterizer."""

    def __init__(self, cfg: Config, center: np.ndarray, extent: float, faces: np.ndarray):
        try:
            import nvdiffrast.torch as dr
        except ModuleNotFoundError as e:
            raise RuntimeError("nvdiffrast is not installed") from e

        if not torch.cuda.is_available():
            raise RuntimeError("nvdiffrast rendering requires CUDA")

        self.cfg = cfg
        self.dr = dr
        self.device = torch.device("cuda")
        self.glctx = dr.RasterizeCudaContext(device=self.device)
        self.center = np.asarray(center, dtype=np.float32)
        self.extent = float(extent)
        self.mvp = torch.tensor(self._mvp(), device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.int32, device=self.device)

    def _mvp(self) -> np.ndarray:
        az = np.deg2rad(self.cfg.azim)
        el = np.deg2rad(self.cfg.elev)
        eye = np.array(
            [
                self.cfg.radius * self.extent * np.cos(el) * np.cos(az),
                self.cfg.radius * self.extent * np.cos(el) * np.sin(az),
                self.cfg.radius * self.extent * np.sin(el),
            ],
            dtype=np.float32,
        )
        eye = eye + self.center
        view = look_at(eye, self.center, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        proj = perspective(40.0, self.cfg.width / self.cfg.height, 0.01, 10.0)
        model = np.eye(4, dtype=np.float32)
        model[:3, :3] *= self.cfg.scale
        return proj @ view @ model

    def render(self, verts: np.ndarray) -> np.ndarray:
        """Rasterize masks for all robots in one call."""
        ones = np.ones((*verts.shape[:-1], 1), dtype=np.float32)
        verts_h = np.concatenate([verts, ones], axis=-1)
        pos = torch.tensor(verts_h, device=self.device)
        pos_clip = pos @ self.mvp.T
        rast, _ = self.dr.rasterize(
            self.glctx,
            pos_clip,
            self.faces,
            resolution=[self.cfg.height, self.cfg.width],
        )
        mask = torch.flip((rast[..., 3] > 0).float(), dims=[1])
        return mask.cpu().numpy()

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to image pixels."""
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=-1)
        clip = pts_h @ self.mvp.cpu().numpy().T
        w = clip[:, 3]
        ok = w > 1e-6
        ndc = np.zeros((len(pts), 3), dtype=np.float32)
        ndc[ok] = clip[ok, :3] / w[ok, None]
        x = (ndc[:, 0] * 0.5 + 0.5) * (self.cfg.width - 1)
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (self.cfg.height - 1)
        xy = np.stack([x, y], axis=-1)
        ok &= np.abs(ndc[:, 0]) <= 1.0
        ok &= np.abs(ndc[:, 1]) <= 1.0
        ok &= np.abs(ndc[:, 2]) <= 1.0
        return xy, ok


def fit_camera(scene: StaticRobot, q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit camera bounds from flow endpoints."""
    lo, hi = scene.bounds(np.concatenate([q0, q1], axis=0))
    center = 0.5 * (lo + hi)
    extent = max(float(np.max(hi - lo)), 0.5)
    return center, extent


def overlay_masks(masks: np.ndarray, color: np.ndarray, bg: np.ndarray, alpha: float) -> np.ndarray:
    """Composite same-color masks without a Python loop."""
    k = masks.sum(axis=0, dtype=np.float32)
    w = 1.0 - np.power(1.0 - alpha, k)
    return bg[None, None, :] * (1.0 - w[..., None]) + color[None, None, :] * w[..., None]


def fill_poly(frame: np.ndarray, pts: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Fill a polygon."""
    if len(pts) < 3:
        return frame
    pts = np.asarray(pts, dtype=np.float32)
    h, w = frame.shape[:2]
    x0 = max(int(np.floor(pts[:, 0].min())), 0)
    x1 = min(int(np.ceil(pts[:, 0].max())), w - 1)
    y0 = max(int(np.floor(pts[:, 1].min())), 0)
    y1 = min(int(np.ceil(pts[:, 1].max())), h - 1)
    if x0 > x1 or y0 > y1:
        return frame

    xx, yy = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.float32) + 0.5,
        np.arange(y0, y1 + 1, dtype=np.float32) + 0.5,
    )
    inside = np.zeros(xx.shape, dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        cross = (y[i] > yy) != (y[j] > yy)
        x_hit = (x[j] - x[i]) * (yy - y[i]) / (y[j] - y[i] + 1e-8) + x[i]
        inside ^= cross & (xx < x_hit)

    out = frame.copy()
    roi = out[y0 : y1 + 1, x0 : x1 + 1]
    a = 0.05
    roi[inside] = (1.0 - a) * roi[inside] + a * color
    return out


def draw_line(frame: np.ndarray, p0: np.ndarray, p1: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Draw a 1 px line segment."""
    out = frame.copy()
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]), 1))
    xs = np.rint(np.linspace(p0[0], p1[0], n + 1)).astype(np.int32)
    ys = np.rint(np.linspace(p0[1], p1[1], n + 1)).astype(np.int32)
    keep = (xs >= 0) & (xs < frame.shape[1]) & (ys >= 0) & (ys < frame.shape[0])
    out[ys[keep], xs[keep]] = color
    return out


def make_video(cfg: Config) -> Path:
    """Render the flowed trajectory."""
    q0, q1, ts = build_flow(cfg)
    scene = StaticRobot(cfg)
    center, extent = fit_camera(scene, q0, q1)
    rast = NVDiffRasterizer(cfg, center=center, extent=extent, faces=scene.faces)
    bg = np.asarray(cfg.bg, dtype=np.float32)
    color = np.asarray(cfg.color, dtype=np.float32)
    tcp_color = np.asarray(cfg.tcp_color, dtype=np.float32)
    line_color = np.asarray(cfg.line_color, dtype=np.float32)
    tcp_hist: list[np.ndarray] = []
    tcp_ok_hist: list[np.ndarray] = []

    with imageio.get_writer(cfg.out, fps=cfg.fps, codec="libx264") as writer:
        for t in tqdm(ts, desc="render frames"):
            q = (1.0 - t) * q0 + t * q1
            verts = scene.posed_verts(q)
            masks = rast.render(verts)
            frame = overlay_masks(masks, color=color, bg=bg, alpha=cfg.alpha)
            tcp = scene.tcp_points(q)
            tcp_px, ok = rast.project(tcp)
            tcp_hist.append(tcp_px.copy())
            tcp_ok_hist.append(ok.copy())
            for i in range(len(q)):
                if not ok[i]:
                    continue
                path = np.array(
                    [past[i] for past, past_ok in zip(tcp_hist, tcp_ok_hist) if past_ok[i]], dtype=np.float32
                )
                frame = fill_poly(frame, path, color=tcp_color)
                frame = draw_line(frame, path[0], tcp_px[i], color=line_color)
                for j in range(1, len(path)):
                    frame = draw_line(frame, path[j - 1], path[j], color=tcp_color)
            writer.append_data((255.0 * np.clip(frame, 0.0, 1.0)).astype(np.uint8))

    return cfg.out


def main() -> None:
    """Run the raster demo."""
    out = make_video(tyro.cli(Config))
    print(out.resolve())


if __name__ == "__main__":
    main()
