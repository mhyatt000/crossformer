"""Standalone synth viz using jaxrenderer (differentiable JAX rasterizer).

Mirrors scripts/debug/test_synth_viz.py but swaps nvdiffrast for
https://github.com/JoeyTeng/jaxrenderer. Saves outputs to /tmp/synth_viz_jax/.

Usage:
    uv run scripts/debug/synth_viz_jaxrenderer.py
    uv run scripts/debug/synth_viz_jaxrenderer.py --view 1000
"""

from __future__ import annotations

import contextlib

import jax as _jax

_orig_update = _jax.config.update


def _safe_update(name, value, *args, **kwargs):
    with contextlib.suppress(AttributeError):
        _orig_update(name, value, *args, **kwargs)


_jax.config.update = _safe_update

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import jax.numpy as jnp
import numpy as np
from PIL import Image
from renderer import (
    Camera,
    LightParameters,
    merge_objects,
    Model,
    Renderer,
    Scene,
)
from rich import print
import tyro

from crossformer.embody import KP2D_NAMES
from crossformer.utils.callbacks.rast import _RobotMesh
from crossformer.utils.callbacks.synth_viz import (
    build_K,
    fk_keypoints,
    solve_pnp,
)

DATA_DIR = Path("robot_vga_100k")
OUT_DIR = Path("/tmp/synth_viz_jax")

IMG_W, IMG_H = 640, 480

ROBOT_COLOR = np.array([0.2, 0.4, 0.9], dtype=np.float32)
ROBOT_ALPHA = 0.6
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)


@dataclass
class Config:
    view: int = 1000
    noise: float = 0.0
    out: Path = OUT_DIR


def _projection_from_intrinsics(K: np.ndarray, W: int, H: int, znear: float = 0.01, zfar: float = 10.0) -> np.ndarray:
    """OpenGL projection matrix built from OpenCV intrinsics."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = 1.0 - 2.0 * cx / W
    P[1, 2] = 2.0 * cy / H - 1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0
    return P


def _viewport_matrix(W: int, H: int, depth: float = 1.0) -> np.ndarray:
    """Map clip-space [-1, 1]^3 onto screen [0, W] x [0, H] x [0, depth]."""
    V = np.eye(4, dtype=np.float32)
    V[0, 0] = W / 2
    V[1, 1] = H / 2
    V[2, 2] = depth / 2
    V[0, 3] = W / 2
    V[1, 3] = H / 2
    V[2, 3] = depth / 2
    return V


def _posed_verts_cartesian(robot: _RobotMesh, joints_rad: np.ndarray) -> np.ndarray:
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    verts_h = robot.posed_verts(q)
    return np.asarray(verts_h[0, :, :3], dtype=np.float32)


def rasterize_robot_jr(
    robot: _RobotMesh,
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """Render robot silhouette with jaxrenderer. Returns (H, W) float mask in [0, 1]."""
    verts = _posed_verts_cartesian(robot, joints_rad)
    faces = robot.faces.astype(np.int32)

    norms = np.zeros_like(verts)
    norms[:, 2] = 1.0
    uvs = np.zeros((verts.shape[0], 2), dtype=np.float32)
    diffuse = np.full((2, 2, 3), ROBOT_COLOR, dtype=np.float32)

    model = Model.create(
        verts=jnp.asarray(verts, dtype=jnp.float32),
        norms=jnp.asarray(norms, dtype=jnp.float32),
        uvs=jnp.asarray(uvs, dtype=jnp.float32),
        faces=jnp.asarray(faces, dtype=jnp.int32),
        diffuse_map=jnp.asarray(diffuse, dtype=jnp.float32),
    )

    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    view = (flip @ w2c).astype(np.float32)
    P = _projection_from_intrinsics(K, W, H)
    VP = _viewport_matrix(W, H, 1.0)

    camera = Camera.create(
        view=jnp.asarray(view),
        projection=jnp.asarray(P),
        viewport=jnp.asarray(VP),
    )

    scene = Scene()
    scene, mid = scene.add_model(model)
    scene, oid = scene.add_object_instance(mid)
    merged = merge_objects([scene.objects[oid]])

    buffers = Renderer.create_buffers(
        W,
        H,
        colour_default=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        zbuffer_default=jnp.array(1.0, dtype=jnp.float32),
    )
    out = Renderer.render(
        model=merged,
        light=LightParameters(),
        camera=camera,
        buffers=buffers,
    )

    zbuf = np.asarray(out.zbuffer)  # (W, H), origin at bottom-left
    mask = (zbuf < 1.0 - 1e-6).astype(np.float32)
    return mask.T[::-1, :]


def composite_robot(img: np.ndarray, mask: np.ndarray, alpha: float = ROBOT_ALPHA) -> np.ndarray:
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_f = img.astype(np.float32) / 255.0
    w = mask[..., None] * alpha
    blended = img_f * (1.0 - w) + ROBOT_COLOR[None, None, :] * w
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _draw_kp2d(
    img: np.ndarray,
    kp2d: np.ndarray,
    W: int,
    H: int,
    color: tuple[int, int, int],
    radius: int = 4,
    thickness: int = 2,
    label: bool = True,
    skeleton: bool = True,
) -> np.ndarray:
    for i, (u, v) in enumerate(kp2d):
        if u == 0.0 and v == 0.0:
            continue
        px, py = int(u * W), int(v * H)
        cv2.circle(img, (px, py), radius, color, thickness)
        if label:
            cv2.putText(img, KP2D_NAMES[i], (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    if skeleton:
        for i in range(len(kp2d) - 1):
            u0, v0 = kp2d[i]
            u1, v1 = kp2d[i + 1]
            if (u0 == 0 and v0 == 0) or (u1 == 0 and v1 == 0):
                continue
            p0 = (int(u0 * W), int(v0 * H))
            p1 = (int(u1 * W), int(v1 * H))
            cv2.line(img, p0, p1, color, 1)
    return img


def main(cfg: Config):
    cfg.out.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / f"view_{cfg.view}.json"
    img_path = DATA_DIR / f"view_{cfg.view}.png"
    assert json_path.exists(), f"not found: {json_path}"
    assert img_path.exists(), f"not found: {img_path}"

    with open(json_path) as f:
        meta = json.load(f)
    img = np.array(Image.open(img_path))
    print(f"[bold]image:[/] {img_path} shape={img.shape}")

    kp2d_gt = np.zeros((10, 2), dtype=np.float32)
    for i, kp in enumerate(meta["keypoints"][:10]):
        if kp["visible"]:
            kp2d_gt[i, 0] = kp["pixel_xy"][0] / 640.0
            kp2d_gt[i, 1] = kp["pixel_xy"][1] / 480.0

    joints_deg = np.array([meta["joints"][f"joint{i}"] for i in range(1, 8)], dtype=np.float32)
    cam = meta["camera"]["intrinsics"]
    cam_px = np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], dtype=np.float64)
    K = build_K(cam_px)
    print(f"[bold]K:[/]\n{K}")

    joints_rad = np.deg2rad(joints_deg.astype(np.float64))

    robot = _RobotMesh(Path("xarm7_standalone.urdf"), Path("assets"))

    pts_3d = fk_keypoints(joints_rad, robot)
    pts_2d_px = kp2d_gt.copy()
    pts_2d_px[:, 0] *= IMG_W
    pts_2d_px[:, 1] *= IMG_H
    w2c = solve_pnp(pts_3d, pts_2d_px, K)
    assert w2c is not None, "PnP failed"

    gt_w2c = np.array(meta["camera"]["extrinsics"]["world_to_camera"], dtype=np.float64)
    print(f"[bold]w2c diff vs GT (Frobenius):[/] {np.linalg.norm(w2c - gt_w2c):.6f}")

    rng = np.random.default_rng(42)
    pred_kp2d = kp2d_gt + rng.normal(0, cfg.noise, kp2d_gt.shape).astype(np.float32) if cfg.noise > 0 else kp2d_gt

    print("[bold]rasterizing with jaxrenderer...[/]")
    mask = rasterize_robot_jr(robot, joints_rad, w2c, K, IMG_W, IMG_H)
    print(f"[bold]mask:[/] shape={mask.shape} nonzero={np.count_nonzero(mask)}")

    Image.fromarray((mask * 255).astype(np.uint8)).save(cfg.out / "mask.png")
    print(f"  saved {cfg.out / 'mask.png'}")

    img_rgb = img[..., :3] if img.shape[-1] == 4 else img
    img_resized = np.array(Image.fromarray(img_rgb).resize((IMG_W, IMG_H)))

    comp = composite_robot(img_resized, mask)
    Image.fromarray(comp).save(cfg.out / "composite.png")
    print(f"  saved {cfg.out / 'composite.png'}")

    full = _draw_kp2d(comp.copy(), kp2d_gt, IMG_W, IMG_H, GT_COLOR, 4, 2, label=True, skeleton=True)
    if cfg.noise > 0:
        full = _draw_kp2d(full, pred_kp2d, IMG_W, IMG_H, PRED_COLOR, 5, 1, label=False, skeleton=True)
    Image.fromarray(full).save(cfg.out / "full.png")
    print(f"  saved {cfg.out / 'full.png'}")

    print(f"\n[bold green]done![/] outputs in {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
