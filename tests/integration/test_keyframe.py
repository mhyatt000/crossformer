"""
test_keyframe.py — visual validation of k3ds UV projection onto real images.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from crossformer.data.grain.datasets import _DecodedArrayRecord
from crossformer.utils.callbacks.flow_viz import (
    _DEFAULT_K,
    load_camera_extrinsics,
    world_to_uv,
)

shards = sorted(Path("~/data/arrayrecords/sweep_mano/0.0.2/to_step").expanduser().glob("*.arrayrecord"))

# Starting orientation: xyz WRT robot origin
START = 30
N = 100
K = _DEFAULT_K
ROS_TO_OPENCV = True


records = _DecodedArrayRecord(shards)
R_wc, t_wc = load_camera_extrinsics("low")

frames: list[np.ndarray] = []

for i in range(min(N, len(records))):
    step = records[START + i]
    xyz = step["observation"]["proprio"]["k3ds"][:, :3]
    img = step["observation"]["image"]["low"]
    H, W = img.shape[:2]

    uv = world_to_uv(xyz, R=R_wc, t=t_wc, K=K, ros_to_opencv=ROS_TO_OPENCV)

    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if i == 0:
        print(f"image shape: {img.shape}")
        print(f"uv u: [{uv[:,0].min():.1f}, {uv[:,0].max():.1f}]  (W={W})")
        print(f"uv v: [{uv[:,1].min():.1f}, {uv[:,1].max():.1f}]  (H={H})")
        print(f"in-frame: {in_frame.sum()}/{len(in_frame)} joints")

    fig, axes = plt.subplots(1, 2, figsize=(W * 2 / 100, H / 100))

    axes[0].imshow(img)
    axes[0].scatter(uv[:, 0], uv[:, 1], c="red", s=20, edgecolors="white", linewidths=0.5)
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)
    axes[0].axis("off")
    axes[0].set_title("image + projected k3ds")

    blank = np.ones_like(img) * 255
    axes[1].imshow(blank)
    axes[1].scatter(uv[:, 0], uv[:, 1], c="blue", s=20, edgecolors="white", linewidths=0.5)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].axis("off")
    axes[1].set_title("UV projection (image coords)")

    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

out_path = "/tmp/keypoints.gif"
imageio.mimwrite(out_path, frames, fps=10)
print(f"saved {len(frames)} frames to {out_path}")

