from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np

from crossformer.data.grain.datasets import _DecodedArrayRecord
from crossformer.utils.callbacks.flow_viz import (
    _DEFAULT_K,
    FlowVisCallback,
    load_camera_extrinsics,
)

# Load first pass of dataset
shards = sorted(Path("~/.cache/arrayrecords/sweep_mano/0.0.2/to_step").expanduser().glob("*.arrayrecord"))
records = _DecodedArrayRecord(shards)

B, ft = 4, 6
rng = np.random.default_rng(0)

# Grab B steps, stack k3ds as fake joints_ft and repeat ft times with small noise
steps = [records[i] for i in range(B)]
k3ds = np.stack([s["observation"]["proprio"]["k3ds"][:, :3] for s in steps])
joints_ft = k3ds[:, None, :, :] + rng.standard_normal((B, ft, 21, 3)).astype(np.float32) * 0.01

R, t = load_camera_extrinsics("low")
K = _DEFAULT_K


def dummy_iter():
    while True:
        yield {}


def dummy_eval_step(state, batch):
    return {"text_conditioned": {"vis": {"joints_ft": joints_ft}}}


cb = object.__new__(FlowVisCallback)
cb.fps = 5
cb.max_videos = 2
cb.num_val_batches = 1
cb.camera_intrinsics = K
cb.camera_R = R
cb.camera_t = t
cb.val_iterators = {"sweep_mano": dummy_iter()}
cb.eval_step = dummy_eval_step

out = cb(train_state=None, step=0)


def save(wandb_video, path):
    frames = wandb_video.data.transpose(0, 2, 3, 1)
    imageio.mimwrite(path, frames, fps=5)


save(out["flow_vis/sweep_mano/uv"][0], "~/uv.gif")
save(out["flow_vis/sweep_mano/joints_3d"][0], "~/joints.gif")
