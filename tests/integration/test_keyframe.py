from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from crossformer.data.grain.datasets import _DecodedArrayRecord
from crossformer.utils.callbacks.flow_viz import _DEFAULT_K, load_camera_extrinsics, world_to_uv

shards = sorted(Path("~/.cache/arrayrecords/sweep_mano/0.0.2/to_step").expanduser().glob("*.arrayrecord"))
records = _DecodedArrayRecord(shards)
R, t = load_camera_extrinsics("low")

H, W = 480, 640

frames = []
for i in range(min(100, len(records))):
    step = records[i]
    xyz = step["observation"]["proprio"]["k3ds"][:, :3]
    img = step["observation"]["image"]["low"]
    uv = world_to_uv(xyz, R=R, t=t, K=_DEFAULT_K)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.scatter(uv[:, 0], uv[:, 1], c="red", s=20, edgecolors="white", linewidths=0.5)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(buf[:, :, :3].copy())
    plt.close(fig)

imageio.mimwrite("/tmp/keypoints.gif", frames, fps=10)
print(f"saved {len(frames)} frames to /tmp/keypoints.gif")
