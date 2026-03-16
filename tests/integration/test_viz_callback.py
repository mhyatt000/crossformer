from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import imageio
import numpy as np

from crossformer.data.grain.datasets import _DecodedArrayRecord
from crossformer.utils.callbacks.flow_viz import (
    _DEFAULT_K,
    FlowVisCallback,
    load_camera_extrinsics,
)

shards = sorted(Path("~/crossformer_data/").expanduser().glob("*.arrayrecord"))
records = _DecodedArrayRecord(shards)

START = 40  # change this to pick frames where hand is visiblek
B, ft = 4, 6
N = B + ft  # need enough steps to build ft-length trajectories per batch item

# Same data as test_keyframe.py: real images from "low" camera + actual k3ds
steps = [records[START + i] for i in range(N)]
k3ds = np.stack([s["observation"]["proprio"]["k3ds"][:, :3] for s in steps])  # (N, 21, 3)
imgs = np.stack([s["observation"]["image"]["low"] for s in steps])  # (N, H, W, 3)

print(f"image shape: {imgs.shape[1:]}")  # H, W, C
print(f"k3ds range: {k3ds.min():.3f} to {k3ds.max():.3f}")

# joints_ft[i] = actual consecutive steps i..i+ft (real future trajectory)
joints_ft = np.stack([k3ds[i : i + ft] for i in range(B)])  # (B, ft, 21, 3)
batch_imgs = imgs[:B]  # (B, H, W, 3)

R, t = load_camera_extrinsics("low")
K = _DEFAULT_K


def dummy_iter():
    while True:
        yield {"observation": {"image_primary": batch_imgs}}


def dummy_eval_step(state, batch):
    return {"text_conditioned": {"vis": {"joints_ft": joints_ft}}}


cb = object.__new__(FlowVisCallback)
cb.fps = 5
cb.max_videos = 2
cb.num_val_batches = 1
cb.camera_intrinsics = K
cb.camera_R = R
cb.camera_t = t
cb.use_rerun = True
cb.rerun_spawn = False
cb.rerun_path = "/home/nh/flow_viz.rrd"
cb.ros_to_opencv = True  # k3ds are in ROS frame; extrinsics assume OpenCV world convention
cb._rerun_initialized = False
cb.val_iterators = {"sweep_mano": dummy_iter()}
cb.eval_step = dummy_eval_step

out = cb(train_state=None, step=0)


def save(wandb_video, path):
    path = str(Path(path).expanduser())
    frames = wandb_video.data.transpose(0, 2, 3, 1)
    imageio.mimwrite(path, frames, fps=5)
    print(f"saved {path}")


save(out["flow_vis/sweep_mano/uv"][0], "~/uv.gif")
save(out["flow_vis/sweep_mano/joints_3d"][0], "~/joints.gif")
