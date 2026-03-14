from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio


def world_to_uv(xyz: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    xyz: [J,3] in world coordinates
    K: [3,3], R: [3,3], t: [3]
    returns uv: [J,2] pixel coordinates
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).reshape(3)

    cam = (R @ xyz.T).T + t[None, :]         # [J,3]
    z = np.clip(cam[:, 2:3], 1e-6, None)
    xy = cam[:, :2] / z
    uv_h = (K @ np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1).T).T
    uv = uv_h[:, :2]
    return uv


def render_xyz_overlay_video(
    image: np.ndarray,                 # [H,W,3]
    xyz_steps: np.ndarray,             # [S,J,3] or [S,N,J,3]
    out_mp4: Path | str,
    out_png: Path | str,
    K: np.ndarray,                     # [3,3]
    R: np.ndarray,                     # [3,3]
    t: np.ndarray,                     # [3]
    fps: int = 10,
    sample_index: int = 0,             # used if xyz_steps is [S,N,J,3]
    draw_history: bool = True,
):
    out_mp4 = Path(out_mp4)
    out_png = Path(out_png)

    img = np.asarray(image)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"image must be [H,W,3/4], got {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]

    xyz = np.asarray(xyz_steps, dtype=np.float32)
    if xyz.ndim == 4:
        xyz = xyz[:, sample_index]  # [S,J,3]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz_steps must be [S,J,3] or [S,N,J,3], got {xyz.shape}")

    S, J, _ = xyz.shape
    H, W = img.shape[:2]

    frames = []
    uv_hist = []

    for s in range(S):
        uv = world_to_uv(xyz[s], K=K, R=R, t=t)  # [J,2]
        uv_hist.append(uv)

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(img)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

        # history trails per joint
        if draw_history and len(uv_hist) > 1:
            # stack: [s+1, J, 2]
            h = np.stack(uv_hist, axis=0)
            for j in range(J):
                ax.plot(h[:, j, 0], h[:, j, 1], color="yellow", alpha=0.25, linewidth=1.0)

        # current joints
        ax.scatter(uv[:, 0], uv[:, 1], c="red", s=18, edgecolors="white", linewidths=0.5)
        ax.text(8, 20, f"step {s+1}/{S}", color="white", fontsize=10,
                bbox=dict(facecolor="black", alpha=0.45, pad=2))

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        plt.close(fig)

    # write outputs
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_mp4, frames, fps=fps)
    imageio.imwrite(out_png, frames[0])

    return {
        "num_frames": S,
        "num_joints": J,
        "out_mp4": str(out_mp4),
        "out_png": str(out_png),
    }
