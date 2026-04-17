"""Standalone synth viz using softras (pure-JAX silhouette rasterizer).

Renders a URDF robot silhouette with a SoftRas-style pure-JAX rasterizer.
Fully differentiable; a final block sanity-checks grad flow from the mask
back to clip-space vertices using `jax.grad`.

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
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from rich import print
import tyro

from crossformer.embody import KP2D_NAMES
from crossformer.utils.callbacks.rast import _RobotMesh
from crossformer.utils.callbacks.synth_viz import (
    build_K,
    fk_keypoints,
    solve_pnp,
)
from crossformer.utils.softras import silhouette

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
    sigma: float | None = None  # None -> 1 pixel in NDC (2 / width)
    chunk: int = 32
    height: int = 480  # render / display height; intrinsics are rescaled to match
    width: int = 640  # render / display width


def _projection_from_intrinsics(K: np.ndarray, W: int, H: int, znear: float = 0.01, zfar: float = 10.0) -> np.ndarray:
    """OpenGL projection built from OpenCV intrinsics. y=+1 is the top row."""
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


def _world_to_clip(verts_world: np.ndarray, w2c: np.ndarray, K: np.ndarray, W: int, H: int) -> np.ndarray:
    """(V, 3) world -> (V, 4) OpenGL clip. diag(1,-1,-1,1) converts OpenCV camera."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    view = flip @ w2c.astype(np.float32)
    P = _projection_from_intrinsics(K, W, H)
    mvp = (P @ view).astype(np.float32)
    v_h = np.concatenate([verts_world, np.ones((len(verts_world), 1), dtype=np.float32)], axis=1)
    return (v_h @ mvp.T).astype(np.float32)


def _posed_verts_cartesian(robot: _RobotMesh, joints_rad: np.ndarray) -> np.ndarray:
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    verts_h = robot.posed_verts(q)
    return np.asarray(verts_h[0, :, :3], dtype=np.float32)


def rasterize_robot_softras(
    robot: _RobotMesh,
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    sigma: float,
    chunk: int,
) -> tuple[np.ndarray, jax.Array, jax.Array]:
    """Returns (mask_np, clip_jax, tris_jax). clip is kept around for grad check."""
    verts3 = _posed_verts_cartesian(robot, joints_rad)
    tris = jnp.asarray(robot.faces.astype(np.int32))
    clip = jnp.asarray(_world_to_clip(verts3, w2c, K, W, H))
    mask = silhouette(clip, tris, H, W, sigma=sigma, chunk=chunk)
    return np.asarray(mask), clip, tris


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


def _scale_intrinsics(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    sx, sy = dst_w / src_w, dst_h / src_h
    K2 = K.astype(np.float64, copy=True)
    K2[0, 0] *= sx  # fx
    K2[0, 2] *= sx  # cx
    K2[1, 1] *= sy  # fy
    K2[1, 2] *= sy  # cy
    return K2


def main(cfg: Config):
    cfg.out.mkdir(parents=True, exist_ok=True)
    SRC_W, SRC_H = IMG_W, IMG_H  # source image + intrinsics resolution
    DST_W, DST_H = cfg.width, cfg.height  # target render / display resolution

    json_path = DATA_DIR / f"view_{cfg.view}.json"
    img_path = DATA_DIR / f"view_{cfg.view}.png"
    assert json_path.exists(), f"not found: {json_path}"
    assert img_path.exists(), f"not found: {img_path}"

    with open(json_path) as f:
        meta = json.load(f)
    img = np.array(Image.open(img_path))
    print(f"[bold]image:[/] {img_path} shape={img.shape}  target=({DST_H}, {DST_W})")

    # Keypoints stored in normalized [0, 1] so they survive resize.
    kp2d_gt = np.zeros((10, 2), dtype=np.float32)
    for i, kp in enumerate(meta["keypoints"][:10]):
        if kp["visible"]:
            kp2d_gt[i, 0] = kp["pixel_xy"][0] / SRC_W
            kp2d_gt[i, 1] = kp["pixel_xy"][1] / SRC_H

    joints_deg = np.array([meta["joints"][f"joint{i}"] for i in range(1, 8)], dtype=np.float32)
    cam = meta["camera"]["intrinsics"]
    cam_px = np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], dtype=np.float64)
    K = build_K(cam_px)
    print(f"[bold]K:[/]\n{K}")

    joints_rad = np.deg2rad(joints_deg.astype(np.float64))

    robot = _RobotMesh(Path("xarm7_standalone.urdf"), Path("assets"))

    # PnP in source-resolution pixel coords with source K.
    pts_3d = fk_keypoints(joints_rad, robot)
    pts_2d_px = kp2d_gt.copy()
    pts_2d_px[:, 0] *= SRC_W
    pts_2d_px[:, 1] *= SRC_H
    w2c = solve_pnp(pts_3d, pts_2d_px, K)
    assert w2c is not None, "PnP failed"

    gt_w2c = np.array(meta["camera"]["extrinsics"]["world_to_camera"], dtype=np.float64)
    print(f"[bold]w2c diff vs GT (Frobenius):[/] {np.linalg.norm(w2c - gt_w2c):.6f}")

    rng = np.random.default_rng(42)
    pred_kp2d = kp2d_gt + rng.normal(0, cfg.noise, kp2d_gt.shape).astype(np.float32) if cfg.noise > 0 else kp2d_gt

    # Rescale intrinsics so the projection matches the target resolution.
    K_render = _scale_intrinsics(K, SRC_W, SRC_H, DST_W, DST_H)

    sigma = cfg.sigma if cfg.sigma is not None else 2.0 / DST_W
    print(
        f"[bold]rasterizing with softras[/] ({DST_H}x{DST_W}, sigma={sigma:.3e} = {sigma * DST_W / 2:.2f}px, chunk={cfg.chunk})..."
    )
    mask, clip, tris = rasterize_robot_softras(robot, joints_rad, w2c, K_render, DST_W, DST_H, sigma, cfg.chunk)
    print(f"[bold]mask:[/] shape={mask.shape} nonzero(>0.01)={int((mask > 0.01).sum())} max={float(mask.max()):.3f}")

    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(cfg.out / "mask.png")
    print(f"  saved {cfg.out / 'mask.png'}")

    img_rgb = img[..., :3] if img.shape[-1] == 4 else img
    img_resized = np.array(Image.fromarray(img_rgb).resize((DST_W, DST_H)))

    comp = composite_robot(img_resized, mask)
    Image.fromarray(comp).save(cfg.out / "composite.png")
    print(f"  saved {cfg.out / 'composite.png'}")

    full = _draw_kp2d(comp.copy(), kp2d_gt, DST_W, DST_H, GT_COLOR, 4, 2, label=True, skeleton=True)
    if cfg.noise > 0:
        full = _draw_kp2d(full, pred_kp2d, DST_W, DST_H, PRED_COLOR, 5, 1, label=False, skeleton=True)
    Image.fromarray(full).save(cfg.out / "full.png")
    print(f"  saved {cfg.out / 'full.png'}")

    # ---- JAX-native gradient sanity check ----
    print("[bold]grad check:[/] d(mask.sum()) / d(clip_verts)")

    def loss(vc: jax.Array) -> jax.Array:
        return silhouette(vc, tris, DST_H, DST_W, sigma=sigma, chunk=cfg.chunk).sum()

    g = jax.grad(loss)(clip)
    finite = bool(jnp.isfinite(g).all())
    gnorm = float(jnp.linalg.norm(g))
    nz = int((jnp.abs(g) > 1e-6).sum())
    print(f"  shape={g.shape} finite={finite} norm={gnorm:.3e} nonzero={nz}/{g.size}")

    print(f"\n[bold green]done![/] outputs in {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
