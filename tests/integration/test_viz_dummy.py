from __future__ import annotations
import numpy as np
import wandb

from crossformer.utils.callbacks.flow_viz import _DEFAULT_K, FlowVisCallback, load_camera_extrinsics

wandb.init(mode="disabled")

rng = np.random.default_rng(0)
B, ft, J = 2, 4, 21


def dummy_iter():
    while True:
        yield {}  # no intrinsics in batch — will use defaults


def dummy_eval_step(state, batch):
    return {
        "text_conditioned": {
            "vis": {
                "joints_ft": rng.standard_normal((B, ft, J, 3)).astype(np.float32),
                "xyz_ft": rng.standard_normal((B, ft, 50, 3)).astype(np.float32),
            }
        }
    }


cb = object.__new__(FlowVisCallback)
cb.fps = 5
cb.max_videos = 2
cb.num_val_batches = 1
cb.camera_intrinsics = _DEFAULT_K.copy()
cb.camera_R, cb.camera_t = load_camera_extrinsics("low")
cb.val_iterators = {"my_ds": dummy_iter()}
cb.eval_step = dummy_eval_step

out = cb(train_state=None, step=0)
Videos = out["flow_vis/my_ds/uv"]
joints = out["flow_vis/my_ds/joints_3d"]
arr = Videos[0].data

import imageio
import numpy as np


def save_video(wandb_video, path):
    arr = wandb_video.data
    frames = arr.transpose(0, 2, 3, 1)
    imageio.mimwrite(path, frames, fps=5)


save_video(out["flow_vis/my_ds/uv"][0], "~/uv.gif")
save_video(out["flow_vis/my_ds/joints_3d"][0], "~/joints.gif")
save_video(out["flow_vis/my_ds/xyz_3d"][0], "~/xyz.gif")
