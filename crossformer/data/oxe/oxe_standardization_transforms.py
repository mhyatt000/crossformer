from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING

from jax.scipy.spatial.transform import Rotation
import numpy as np
import tensorflow as tf

from crossformer.cn.dataset.types import ActionRep, ActionSpace

if TYPE_CHECKING:
    pass

METRIC_WAYPOINT_SPACING = {
    "cory_hall": 0.06,
    "go_stanford": 0.12,
    "recon": 0.25,
    "sacson": 0.255,
    "scand": 0.38,
    "seattle": 0.35,
    "tartan_drive": 0.72,
}


def xgym_mano_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """
    TODO make this dynamic depending on which MANO inference strategy we select
    """

    obs = trajectory.pop("observation")

    # frame is used for v1

    # obs.pop("proprio")

    # DMANO_PALM = 6
    kp = obs["keypoints_3d"]
    j = kp[:, 0]  # palm

    def grippiness(fingers):
        """grippiness is estimated as the mean dist 3d of all the fingers"""

        fingers = np.array(fingers)
        print(type(fingers))
        combos = itertools.product(fingers, fingers)
        dist = np.mean([np.linalg.norm(kp[a] - kp[b]) for a, b in combos])
        return dist

    #
    #     def pairwise_combinations(fingers):
    #         """
    #         Computes pairwise 3D distances for a 5x3 array (fingers) using NumPy broadcasting.
    #         """
    #
    #         fingers = np.array(fingers)
    #         # Use broadcasting to compute pairwise differences
    #         diff = fingers[:, np.newaxis, :] - fingers[np.newaxis, :, :]  # Shape: (5, 5, 3)
    #
    #         # Compute the Euclidean distances
    #         distances = np.linalg.norm(diff, axis=2)  # Shape: (5, 5)
    #
    #         # Get the upper triangle of the distance matrix (excluding diagonal)
    #         triu_indices = np.triu_indices(fingers.shape[0], k=1)
    #         pairwise_distances = distances[triu_indices]
    #
    #         return pairwise_distances
    #
    #     def pairwise_combinations(fingers):
    #         """
    #         Computes pairwise 3D distances for a TensorFlow tensor (fingers) using broadcasting.
    #         Compatible with symbolic execution.
    #         """
    #         # Expand dimensions to compute pairwise differences
    #         diff = tf.expand_dims(fingers, axis=1) - tf.expand_dims(fingers, axis=0)  # Shape: (N, N, 3)
    #
    #         # Compute the Euclidean distances
    #         distances = tf.norm(diff, axis=2)  # Shape: (N, N)
    #
    #         # Extract the upper triangle indices (excluding the diagonal)
    #         num_fingers = tf.shape(fingers)[0]
    #         row_indices, col_indices = tf.linalg.band_part(tf.ones((num_fingers, num_fingers)), 0, -1) - tf.eye(num_fingers)
    #         upper_triangle_mask = tf.where(row_indices > 0)
    #
    #         # Gather upper triangle distances
    #         pairwise_distances = tf.gather_nd(distances, upper_triangle_mask)
    #
    #         return pairwise_distances
    #

    def pairwise_combinations(fingers):
        """
        Computes pairwise 3D distances for a TensorFlow tensor (fingers) using broadcasting.
        Compatible with symbolic execution.
        """
        # Expand dimensions to compute pairwise differences
        diff = tf.expand_dims(fingers, axis=1) - tf.expand_dims(fingers, axis=0)  # Shape: (N, N, 3)

        # Compute the Euclidean distances
        distances = tf.norm(diff, axis=2)  # Shape: (N, N)

        # Create a mask for the upper triangle (excluding the diagonal)
        num_fingers = tf.shape(fingers)[0]
        row_indices, col_indices = tf.meshgrid(tf.range(num_fingers), tf.range(num_fingers), indexing="ij")
        upper_triangle_mask = tf.cast(row_indices < col_indices, tf.bool)

        # Apply the mask to get the upper triangle distances
        pairwise_distances = tf.boolean_mask(distances, upper_triangle_mask)
        return tf.reduce_mean(pairwise_distances)

    def mat2euler(mat):
        """helper to avoid np problems with tf symbolic/eager tensors"""
        rot = Rotation.from_matrix(mat)
        return rot.as_euler("xyz", degrees=False)

    rot = obs["mano.global_orient"][:, 0]
    rot = tf.numpy_function(mat2euler, [rot], tf.float32)

    loc = np.array([4, 8, 12, 20]).astype(np.uint8)

    # j = [j[:, x] for x in [0, 4, 8, 12, 16, 20]]
    fingers = tf.gather(kp, indices=[4, 8, 12, 16, 20], axis=1)
    grip = tf.map_fn(pairwise_combinations, fingers, fn_output_signature=tf.float32)
    grip = tf.expand_dims(grip, axis=-1)  # shape b => b,1

    state = tf.concat([j, rot, grip], axis=-1)  # w,7
    # state = tf.concat([j, rot], axis=-1)  # w,7

    deltas = state[1:] - state[:-1]
    deltas = tf.concat([deltas, tf.zeros_like(state[-1:])], axis=0)
    actions = deltas

    focal = obs["scaled_focal_length"]
    focal = tf.expand_dims(focal, axis=-1)  # shape b => b,1
    proprio = tf.concat([state, focal], axis=-1)

    trajectory["action"] = actions
    trajectory["observation"] = {
        "image": obs.pop("img"),
        "proprio": proprio,
    }
    return trajectory


def xgym_single_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    obs = trajectory.pop("observation")
    images = obs.pop("image")

    from crossformer.cn.dataset.action import DataSpec, XGYM

    xcfg = DataSpec.REGISTRY["xgym_stack_single"]
    assert isinstance(xcfg, XGYM)

    if use_ss := (xcfg.freq_data != xcfg.freq_train):
        subsample = xcfg.freq_data // xcfg.freq_train

    def recompute() -> tf.Tensor:
        """recompute the action representation on the fly
        because use_ss implies that the action representation
        is not directly available in the dataset
        """
        prop = obs["proprio"]
        out = {k: v for k, v in prop.items() if k != "gripper"}
        out["gripper"] = prop["gripper"][1:]

    def getrep() -> dict[str, tf.Tensor]:
        if use_ss:
            return recompute()
        if xcfg.action_rep == ActionRep.RELATIVE:
            return trajectory["action"]
        if xcfg.action_rep == ActionRep.ABSOLUTE:
            return obs["proprio"]
        raise ValueError(f"Unknown action representation {xcfg.action_rep}")

    # pprint(xcfg.action_encoding)
    match xcfg.action_encoding:
        case ActionSpace.JOINT:
            spacekey = "joints"
        case ActionSpace.POS_EULER:
            spacekey = "position"
        case _:
            raise ValueError(f"Unknown action encoding {xcfg.action_encoding}")

    _t = getrep()
    base = _t[spacekey]
    gripper = _t["gripper"]

    proprio = obs.pop("proprio")
    position = proprio["position"]
    joints = proprio["joints"]
    trajectory["observation"] = {
        **images,
        "proprio": tf.concat([position, joints, gripper], axis=-1),
    }

    trajectory["action"] = tf.concat([base, gripper], axis=-1)

    return trajectory


def oakink_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Applies to Oak-Ink dataset."""

    # take every 3rd frame... 30FPS => 10FPS
    # trajectory = jax.tree.map(lambda x: x[::3], trajectory)

    obs = trajectory.pop("observation")

    """
    # apply the oakink transform
    generator = torch.Generator()
    gstate = generator.get_state()
    results = []
    for i in range(len(obs["image"])):
        print(f'range: {len(obs["image"])}')
        g = torch.Generator()
        g.set_state(gstate)

        o = {k: v[i] for k, v in obs.items()}
        print(o.keys())
        print({k: v.shape for k, v in o.items()})
        # random crop on the center of the image
        o["bbox_center"] = o['cam_center']
        o["bbox_scale"] = min(np.array(o['raw_size']))

        image = o.pop("image")
        results.append(OI(image, o, generator=g))

    results = default_collate(results)
    obs.update(results)
    obs = {k: np.array(v) for k, v in obs.items() if isinstance(v, torch.Tensor)}
    """

    # xyz of palm
    j = obs["joints_3d"]
    j = [j[:, x] for x in [0, 4, 8, 12, 16, 20]]
    state = tf.reshape(j, [-1, 18])

    # flatten state keys into a single tensor
    proprio = tf.reshape(
        tf.concat(
            [
                tf.reshape(v, [tf.shape(v)[0], -1])
                for v in [
                    obs[f"{_v}"]
                    # obs[f"target_{_v}"]
                    for _v in ["joints_3d", "mano_pose", "cam_intr"]
                ]
            ],
            axis=-1,
        ),
        [-1, 120],
    )

    deltas = state[1:] - state[:-1]
    deltas = tf.concat([deltas, tf.zeros_like(state[-1:])], axis=0)

    actions = deltas
    # actions = tf.concat([deltas[:, :-9], state[:, -9:]], axis=-1)  # camera is absolute

    # roll the state by 1
    # actions = tf.roll(state, shift=-1, axis=0)
    # last = state[-2:-1] # if we are done then the absolute mesh is same as last
    # trajectory["action"] = tf.concat([actions[:-1], last], axis=0)
    trajectory["action"] = actions

    trajectory["observation"] = {
        "image": obs["image"],
        "proprio": proprio,
    }
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "xgym_lift_mano": xgym_mano_dataset_transform,
    "xgym_stack_mano": xgym_mano_dataset_transform,
    "xgym_duck_mano": xgym_mano_dataset_transform,
    "xgym_single": xgym_single_dataset_transform,
    "xgym_lift_single": xgym_single_dataset_transform,
    "xgym_duck_single": xgym_single_dataset_transform,
    "xgym_stack_single": xgym_single_dataset_transform,
    "xgym_sweep_single": xgym_single_dataset_transform,
    "xgym_play_single": xgym_single_dataset_transform,
    "rlds_oakink": oakink_dataset_transform,
}
