from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import imageio

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy


def make_robot_from_urdf(urdf_path: str | Path, mesh_dir: str | Path) -> pk.Robot:
    urdf_model = yourdfpy.URDF.load(str(urdf_path), mesh_dir=Path(mesh_dir))
    return pk.Robot.from_urdf(urdf_model)


def _normalize_q_steps(q_steps: np.ndarray, sample_index: int = 0) -> np.ndarray:
    """
    Accept:
      [S,D]      -> [S,1,D]
      [S,N,D]    -> [S,N,D]
    """
    q = np.asarray(q_steps, dtype=np.float32)
    if q.ndim == 2:
        q = q[:, None, :]
    elif q.ndim != 3:
        raise ValueError(f"q_steps must be [S,D] or [S,N,D], got {q.shape}")
    return q


def _fk_all_links(robot: pk.Robot, q: jax.Array, pad_gripper: bool = True) -> jax.Array:
    """
    q: [..., Dq]
    returns xyz: [..., L, 3]
    """
    if pad_gripper:
        pads = [(0, 0)] * (q.ndim - 1) + [(0, 1)]
        q = jnp.pad(q, pads, constant_values=0.0)
    T = jaxlie.SE3(robot.forward_kinematics(q)).translation()  # [..., L, 3]
    return T


def _topdown_project(xyz: np.ndarray, scale: float = 220.0, cx: float = 320, cy: float = 240) -> np.ndarray:
    """
    Simple top-down orthographic projection for quick rendering.
    xyz: [L,3] -> uv: [L,2]
    x-right, y-forward -> image plane
    """
    xy = xyz[:, :2]
    uv = np.empty((xy.shape[0], 2), dtype=np.float32)
    uv[:, 0] = cx + scale * xy[:, 0]
    uv[:, 1] = cy - scale * xy[:, 1]
    return uv


def _render_frame(
    uv: np.ndarray,                  # [L,2]
    links_xyz: np.ndarray,           # [L,3], for optional depth coloring later
    H: int = 480,
    W: int = 640,
    bg=(245, 245, 245),
    point_color=(40, 40, 220),
    trail_uv: Optional[np.ndarray] = None,   # [T,2] for ee trail
) -> np.ndarray:
    """
    Lightweight numpy frame renderer.
    Draws points + optional ee trail.
    """
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :] = np.array(bg, dtype=np.uint8)

    # draw ee trail (last link assumed ee)
    if trail_uv is not None and len(trail_uv) > 1:
        for i in range(1, len(trail_uv)):
            x0, y0 = trail_uv[i - 1]
            x1, y1 = trail_uv[i]
            _draw_line(frame, int(x0), int(y0), int(x1), int(y1), color=(220, 60, 60), thickness=1)

    # draw link points
    for p in uv:
        _draw_circle(frame, int(p[0]), int(p[1]), r=4, color=point_color)

    return frame


def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color=(0, 0, 255)):
    H, W = img.shape[:2]
    x0, x1 = max(0, cx - r), min(W - 1, cx + r)
    y0, y1 = max(0, cy - r), min(H - 1, cy + r)
    rr2 = r * r
    for y in range(y0, y1 + 1):
        dy2 = (y - cy) * (y - cy)
        for x in range(x0, x1 + 1):
            if (x - cx) * (x - cx) + dy2 <= rr2:
                img[y, x] = color


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=(255, 0, 0), thickness: int = 1):
    H, W = img.shape[:2]
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        for ty in range(-thickness, thickness + 1):
            for tx in range(-thickness, thickness + 1):
                xx, yy = x + tx, y + ty
                if 0 <= xx < W and 0 <= yy < H:
                    img[yy, xx] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def render_robot_q_flow_video(
    robot: pk.Robot,
    q_steps: np.ndarray,                 # [S,Dq] or [S,N,Dq]
    out_mp4: str | Path,
    out_png: str | Path,
    sample_index: int = 0,
    pad_gripper: bool = True,
    fps: int = 10,
    H: int = 480,
    W: int = 640,
    scale: float = 220.0,
):
    """
    Part C: Actual joints rendered on virtual robot (no PCA).
    """
    out_mp4 = Path(out_mp4)
    out_png = Path(out_png)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    q = _normalize_q_steps(q_steps)     # [S,N,D]
    S, N, D = q.shape
    sample_index = int(np.clip(sample_index, 0, N - 1))

    # choose one sample trajectory for part C video
    q_sel = q[:, sample_index, :]       # [S,D]

    xyz_all = np.asarray(_fk_all_links(robot, jnp.asarray(q_sel), pad_gripper=pad_gripper), dtype=np.float32)  # [S,L,3]
    S2, L, _ = xyz_all.shape
    assert S2 == S

    frames = []
    ee_trail = []

    for s in range(S):
        links_xyz = xyz_all[s]                      # [L,3]
        uv = _topdown_project(links_xyz, scale=scale, cx=W / 2, cy=H / 2)
        ee_uv = uv[-1]                              # assume last link ~ ee
        ee_trail.append(ee_uv.copy())

        frame = _render_frame(
            uv=uv,
            links_xyz=links_xyz,
            H=H,
            W=W,
            trail_uv=np.asarray(ee_trail),
        )

        # step text
        text = f"robot q flow step {s+1}/{S} (N={N}, sample={sample_index})"
        _put_text_block(frame, text, x=8, y=24)
        frames.append(frame)

    # save
    imageio.imwrite(out_png, frames[0])
    imageio.mimwrite(out_mp4, frames, fps=fps)

    return {
        "out_png": str(out_png),
        "out_mp4": str(out_mp4),
        "num_frames": S,
        "num_links": L,
        "num_samples": N,
    }


def _put_text_block(img: np.ndarray, txt: str, x: int = 8, y: int = 24):
    """
    Minimal placeholder text marker (no cv2 dependency).
    If you already use cv2/PIL, replace this with real text rendering.
    """
    # simple black strip to indicate status line region
    h = 24
    img[max(0, y - h):y + 4, max(0, x - 4):min(img.shape[1], x + 420)] = (30, 30, 30)
    # no font renderer here to keep dependencies minimal


# -------------------------
# tiny CLI for quick testing
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part C: render joint flow on virtual robot.")
    parser.add_argument("--q-npy", type=Path, required=True, help="Path to q steps (.npy or .npz)")
    parser.add_argument("--q-key", type=str, default="q_flow_steps", help="npz key")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/flow_viz_c"))
    parser.add_argument("--urdf-path", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.q_npy.suffix == ".npz":
        q_steps = np.load(args.q_npy)[args.q_key]
    else:
        q_steps = np.load(args.q_npy)

    robot = make_robot_from_urdf(args.urdf_path, args.mesh_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = render_robot_q_flow_video(
        robot=robot,
        q_steps=q_steps,
        out_mp4=args.out_dir / "part_c_robot.mp4",
        out_png=args.out_dir / "part_c_robot_step0.png",
        sample_index=args.sample_index,
        fps=args.fps,
    )
    print(out)
