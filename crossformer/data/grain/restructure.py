"""
Restructure raw trajectories into a unified format for training.
"""

from __future__ import annotations

import jax
import numpy as np

from crossformer.utils.jax_utils import str2np

ACTION_KEYS = ("joints", "gripper", "position", "orientation")
LOWDIM_PROPRIO_KEYS = ACTION_KEYS
KP_KEYS = ("kp3dc_robot", "kp3dw_robot")


def has_usable_keypoints(x: dict) -> bool:
    """False only when keypoint masks exist and all are empty."""
    masks = x.get("mask", {}).get("proprio", {})
    kp_masks = [np.asarray(masks[k]) for k in KP_KEYS if k in masks]
    if "kp3dc_robot" in masks:
        return bool(_valid_camera_mask(x).any())
    return True if not kp_masks else any(bool(m.any()) for m in kp_masks)


def _take_keys(x: dict, keys: tuple[str, ...]) -> dict:
    return {k: x[k] for k in keys if k in x}


def _as_action_tree(x: dict) -> dict:
    return {k: np.asarray(v).reshape(1, -1) for k, v in _take_keys(x, ACTION_KEYS).items()}


def _valid_camera_mask(x: dict) -> np.ndarray:
    masks = x.get("mask", {})
    proprio = masks.get("proprio", {})
    cam = np.asarray(proprio.get("kp3dc_robot", []), dtype=bool)
    if cam.ndim >= 2:
        valid = cam.reshape(cam.shape[0], -1).any(axis=-1)
    else:
        image = np.asarray(x.get("image", []))
        valid = np.ones(image.shape[:1], dtype=bool)

    extr = masks.get("extr", {})
    if "w2c" in extr:
        valid = np.logical_and(valid, np.asarray(extr["w2c"], dtype=bool))
    return valid


def _select_primary_image(x: dict) -> None:
    image = x.get("image")
    if image is None:
        return
    image = np.asarray(image)
    if image.ndim != 4:
        return
    valid = _valid_camera_mask(x)
    idx = int(np.flatnonzero(valid)[0])
    x["image"] = {"primary": image[idx]}


def _wrap_observation_record(x: dict) -> dict:
    """Wrap flat ARec observations into the trajectory schema."""
    x = dict(x)
    info = x.pop("info")
    info = dict(info)
    info.pop("reg", None)
    action = x.pop("action", None)
    _select_primary_image(x)
    x.pop("extr", None)
    x.pop("mask", None)
    if "proprio" in x:
        proprio = x["proprio"]
        x["proprio"] = _take_keys(proprio, LOWDIM_PROPRIO_KEYS)
        if action is None:
            action = _as_action_tree(proprio)
    return {
        "observation": x,
        "action": action,
        "info": info,
    }


def _restructure_trajectory(
    step: dict,
    *,
    name: str,
    lang_key: str | None = None,
) -> dict:
    if "observation" not in step:
        step = _wrap_observation_record(step)
    info = step["info"]
    if "id" in info:
        info = info | info["id"]  # flatten id into info for backward compatibility
    sid, eid = np.array(info["step"]).reshape(-1), np.array(info["episode"]).reshape(-1)
    step["info"]["id"] = {"step": sid, "episode": eid}  # patch
    step["observation"]["timestep"] = sid

    task = {}
    # PATCH 0.5.2 to 0.5.3
    # task[lang_key] = step[lang_key]  # simple
    if "pose" in step["observation"]["proprio"]:
        step["observation"]["proprio"].pop("pose")

    return {
        "observation": step["observation"],
        "task": task,
        "action": step["action"],  #  action,
        "dataset_name": str2np(name, length=32),
        "info": {
            "dataset_name": str2np(name, length=32),
            "id": {k.replace("_id", ""): np.array([v]).reshape(-1) for k, v in step.items() if "_id" in k},
        }
        | step.get("info", {}),
    }


def _restructure_step_mano(x: dict, *, name: str, lang_key: str) -> dict:
    # PATCH from <0.5.5
    lang_key = "language"
    task = {}
    task[lang_key] = x.pop(lang_key)  # simple
    x["task"] = task

    x["observation"]["timestep"] = x["info"]["id"]["step"]

    # k3ds: (H, 21, 4) → strip homogeneous coord → (H, 21, 3)
    # derive cart_pos from palm keypoint (index 0) before flatten
    k3ds = np.array(x["action"]["k3ds"])  # (H, 21, 4)
    k3ds = k3ds[..., :3]  # (H, 21, 3) drop homogeneous w
    x["action"]["position"] = k3ds[:, 0, :]  # (H, 3) palm = cart_pos
    x["action"]["k3ds"] = k3ds.reshape(k3ds.shape[0], -1)  # (H, 63)
    x["action"].pop("k3ds")

    x["observation"]["proprio"] = jax.tree.map(lambda y: y[0], x["action"])

    x = jax.tree.map(lambda y: np.array(y), x)  # ensure numpy arrays
    return x


def restructure_xarm_dream(step: dict, *, name: str, lang_key: str | None = None) -> dict:
    """Restructure single-step xarm_dream synthetic data.

    Raw keys: image, state.{joints, gripper, kp2d, kp3d_world, kp3d_camera},
              camera.{intr.{fx,fy,cx,cy,K}, extr.{c2w,w2c}}, info.*
    """
    state = step["state"]
    cam = step["camera"]["intr"]
    info = step.get("info", {})

    # kp2d: scale uv to [0,1] by image dims (640x480); append vis channel.
    # OOF joints get a neutral sentinel uv (0.5, 0.5) — loss is masked via vis.
    kp2d = np.array(state["kp2d"], dtype=np.float32)  # (10, 2) copy for mutation
    kp_vis = np.asarray(info.get("kp_visible", np.ones(10, dtype=bool)))  # (10,)
    kp2d[:, 0] /= 640.0  # u
    kp2d[:, 1] /= 480.0  # v
    kp2d[~kp_vis] = 0.5
    vis = kp_vis.astype(np.float32)[:, None]  # (10, 1)
    kp2d = np.concatenate([kp2d, vis], axis=1).reshape(-1)  # (30,) = u,v,vis per joint

    # cam_intr: min-max scale fx/fy to [0,1] with [450, 900] range
    FX_MIN, FX_MAX = 450.0, 900.0
    fx = np.clip((cam["fx"] - FX_MIN) / (FX_MAX - FX_MIN), 0.0, 1.0)
    fy = np.clip((cam["fy"] - FX_MIN) / (FX_MAX - FX_MIN), 0.0, 1.0)
    cx = cam["cx"] / 640.0  # scale by image width
    cy = cam["cy"] / 480.0  # scale by image height
    cam_intr = np.array([fx, fy, cx, cy], dtype=np.float32)

    # cam_extr: (tx,ty,tz) + Zhou 6D rotation. Pipeline normalizes translation;
    # 6D kept raw (lies on a manifold — mean/std would break orthogonality).
    # Convert row-vector Blender convention (p_cam = p_world @ w2c_raw, with
    # translation in the last ROW) to standard column-vector SE(3) form
    # ([[R | t]; [0 0 0 1]], translation in the last column) via transpose.
    w2c = np.asarray(step["camera"]["extr"]["w2c"], dtype=np.float32).T  # (4, 4)
    # Blender camera is y-up / z-back; convert to OpenCV (y-down / z-forward)
    # by flipping the y and z axes in camera frame. This matches what PnP /
    # the rasterizer expect, so the stored cam_extr is directly usable.
    FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R = FLIP @ w2c[:3, :3]
    t_xyz = FLIP @ w2c[:3, 3]
    # Zhou et al. 2019 ("On the Continuity of Rotation Representations in Neural
    # Networks") define the 6D rep as literally the first two columns of R.
    # The third column is recoverable at inference via Gram-Schmidt.
    r6d = np.concatenate([R[:, 0], R[:, 1]], axis=0)  # (6,)
    cam_extr = np.concatenate([t_xyz, r6d], axis=0).astype(np.float32)  # (9,)

    # horizon=1: action == proprio (model predicts current state)
    action = {
        "joints": np.asarray(state["joints"]),
        "gripper": np.asarray(state["gripper"], dtype=np.float32).reshape(1),
        "kp2d": kp2d,
        "cam_intr": cam_intr,
        "cam_extr": cam_extr,
    }
    proprio = action  # same values, (D,)
    action = jax.tree.map(lambda x: x[None], action)  # (1, D) for horizon

    # Per-DOF validity masks per body part. For kp2d, gate u/v by per-joint
    # visibility; the vis DOF itself is always supervised.
    kp2d_valid = np.stack([kp_vis, kp_vis, np.ones(10, dtype=bool)], axis=1).reshape(
        -1
    )  # (30,) matches u,v,vis DOF order
    act_mask = {
        "joints": np.ones(7, dtype=bool),
        "gripper": np.ones(1, dtype=bool),
        "kp2d": kp2d_valid,
        "cam_intr": np.ones(4, dtype=bool),
        "cam_extr": np.ones(9, dtype=bool),
    }

    sid = np.array(info.get("id", {}).get("step", 0)).reshape(-1)
    eid = np.array(info.get("id", {}).get("episode", 0)).reshape(-1)
    info["id"] = {"step": sid, "episode": eid}

    lang = step.get("language", {})

    return {
        "observation": {
            "image": {"low": step["image"]},
            "proprio": proprio,
            "timestep": sid,
        },
        "task": {},
        "action": action,
        "mask": {"act": act_mask},
        "dataset_name": str2np(name, length=32),
        "language.embedding": lang.get("embedding", np.zeros((512,), dtype=np.float32)),
        "info": info | {"dataset_name": str2np(name, length=32)},
        "aux": {
            "kp3d_world": np.asarray(state.get("kp3d_world", [])),
            "kp3d_camera": np.asarray(state.get("kp3d_camera", [])),
            "kp_visible": np.asarray(info.get("kp_visible", [])),
            "cam_extr": np.asarray(step.get("camera", {}).get("extr", {}).get("w2c", [])),
        },
    }
