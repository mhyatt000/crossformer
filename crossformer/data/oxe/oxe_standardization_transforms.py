"""Open X-Embodiment Dataset Transforms

input: dict of features, each is batched, i.e. has leading time dimension
expected output:
step = {
    'observation': {
        <image_keys, depth_image_keys>
        state in chosen state representation
    },
    'action': action in chosen action representation,
    'language_instruction': str,
}
"""

import itertools
import random
from typing import Any, Dict

import jax
from jax.scipy.spatial.transform import Rotation
import numpy as np
import tensorflow as tf

from crossformer.data.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_actions,
)

METRIC_WAYPOINT_SPACING = {
    "cory_hall": 0.06,
    "go_stanford": 0.12,
    "recon": 0.25,
    "sacson": 0.255,
    "scand": 0.38,
    "seattle": 0.35,
    "tartan_drive": 0.72,
}


def xgym_mano_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
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

    def pairwise_combinations(fingers):
        """
        Computes pairwise 3D distances for a 5x3 array (fingers) using NumPy broadcasting.
        """

        fingers = np.array(fingers)
        # Use broadcasting to compute pairwise differences
        diff = fingers[:, np.newaxis, :] - fingers[np.newaxis, :, :]  # Shape: (5, 5, 3)

        # Compute the Euclidean distances
        distances = np.linalg.norm(diff, axis=2)  # Shape: (5, 5)

        # Get the upper triangle of the distance matrix (excluding diagonal)
        triu_indices = np.triu_indices(fingers.shape[0], k=1)
        pairwise_distances = distances[triu_indices]

        return pairwise_distances

    def pairwise_combinations(fingers):
        """
        Computes pairwise 3D distances for a TensorFlow tensor (fingers) using broadcasting.
        Compatible with symbolic execution.
        """
        # Expand dimensions to compute pairwise differences
        diff = tf.expand_dims(fingers, axis=1) - tf.expand_dims(
            fingers, axis=0
        )  # Shape: (N, N, 3)

        # Compute the Euclidean distances
        distances = tf.norm(diff, axis=2)  # Shape: (N, N)

        # Extract the upper triangle indices (excluding the diagonal)
        num_fingers = tf.shape(fingers)[0]
        row_indices, col_indices = tf.linalg.band_part(
            tf.ones((num_fingers, num_fingers)), 0, -1
        ) - tf.eye(num_fingers)
        upper_triangle_mask = tf.where(row_indices > 0)

        # Gather upper triangle distances
        pairwise_distances = tf.gather_nd(distances, upper_triangle_mask)

        return pairwise_distances

    def pairwise_combinations(fingers):
        """
        Computes pairwise 3D distances for a TensorFlow tensor (fingers) using broadcasting.
        Compatible with symbolic execution.
        """
        # Expand dimensions to compute pairwise differences
        diff = tf.expand_dims(fingers, axis=1) - tf.expand_dims(
            fingers, axis=0
        )  # Shape: (N, N, 3)

        # Compute the Euclidean distances
        distances = tf.norm(diff, axis=2)  # Shape: (N, N)

        # Create a mask for the upper triangle (excluding the diagonal)
        num_fingers = tf.shape(fingers)[0]
        row_indices, col_indices = tf.meshgrid(
            tf.range(num_fingers), tf.range(num_fingers), indexing="ij"
        )
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


from crossformer.cn.dataset.action import XGYM, ActionRep, ActionSpace


def xgym_single_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:

    obs = trajectory.pop("observation")
    images = obs.pop("image")

    x = XGYM()

    def getrep() -> dict[str, tf.Tensor]:
        if x.action_rep == ActionRep.RELATIVE:
            return trajectory["action"]
        if x.action_rep == ActionRep.ABSOLUTE:
            return obs["proprio"]
        raise ValueError(f"Unknown action representation {x.action_rep}")

    spacekey = None
    if x.action_encoding == ActionSpace.JOINT:
        spacekey = "joints"
    if x.action_encoding == ActionSpace.POS_EULER:
        spacekey = "position"
    if spacekey is None:
        raise ValueError(f"Unknown action encoding {x.action_encoding}")

    base = getrep()[spacekey]
    gripper = getrep()["gripper"]

    proprio = obs.pop("proprio")
    position = proprio["position"]
    joints = proprio["joints"]
    trajectory["observation"] = {**images, "proprio": position}

    trajectory["action"] = tf.concat([base, gripper], axis=-1)

    return trajectory


def oakink_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
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


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    gripper_value = tf.io.decode_compressed(
        trajectory["observation"]["gripper_closed"], compression_type="ZLIB"
    )
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["proprio"] = tf.concat(
        (
            tf.reshape(eef_value, (-1, 7)),
            tf.reshape(gripper_value, (-1, 1)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def taco_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["robot_obs"][:, :6],
            trajectory["observation"]["robot_obs"][:, 7:8],
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def berkeley_cable_routing_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1)
    )

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_states"],
            trajectory["observation"]["gripper_states"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def berkeley_autolab_ur5_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = trajectory["observation"].pop(
        "image_with_depth"
    )

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"][
        :, 6:14
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "effector_translation"
    ]
    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(
        instruction_bytes, output_encoding="UTF-8"
    )
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[
        :, :1
    ].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][
        ..., 0
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["ee_position"],
            trajectory["observation"]["ee_orientation"],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :7]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
            trajectory["observation"]["state"][:, -3:-2],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(
        trajectory["observation"]["depth"][..., 0], tf.float32
    )
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -6:]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_pose"],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :7]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(
                tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["present/xyz"],
            trajectory["observation"]["present/axis_angle"],
            trajectory["observation"]["present/sensed_close"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_pose"
    ]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
            trajectory["observation"]["state"][:, -1:],
        ),
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["pose"],
            tf.cast(trajectory["observation"]["gripper"], tf.float32)[:, None],
        ),
        axis=-1,
    )

    # invert gripper
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :-1],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ],
        axis=1,
    )

    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 30Hz to 10Hz
    factor = 3
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)

    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_pos"],
            tf.cast(trajectory["observation"]["gripper"], tf.float32)[:, None],
        ),
        axis=-1,
    )

    # recompute actions for downsampled sequence
    joint_actions = (
        trajectory["observation"]["joint_pos"][1:, :7]
        - trajectory["observation"]["joint_pos"][:-1, :7]
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)

    # recombine to get full actions, invert gripper
    traj_truncated["action"] = tf.concat(
        [joint_actions, invert_gripper_actions(trajectory["action"][:-1, -1:])],
        axis=1,
    )

    return traj_truncated


def kaist_nonprehensible_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -7:]
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
            trajectory["observation"]["end_effector_pose"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["ground_truth_states"]["EE"],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, 6:7],
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def omnimimic_gnm_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    traj_len = tf.shape(trajectory["action"])[0]
    action_horizon = 100

    # Pad trajectory states
    padding = tf.tile(trajectory["observation"]["state"][-1:, :], [action_horizon, 1])
    trajectory["observation"]["state"] = tf.concat(
        (trajectory["observation"]["state"], padding), axis=0
    )

    # Get next len_seq_pred indices
    indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(1, action_horizon + 1)
    global_waypoints = tf.gather(trajectory["observation"]["state"], indices)[:, :, :2]

    # Get current position indices
    curr_pos_indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(
        0, action_horizon
    )
    curr_pos = tf.gather(trajectory["observation"]["state"], curr_pos_indices)[
        :, :, :2
    ]  # delta waypoints

    global_waypoints -= curr_pos
    global_waypoints = tf.expand_dims(global_waypoints, 2)
    actions = tf.squeeze(
        tf.linalg.matmul(
            global_waypoints,
            tf.expand_dims(trajectory["observation"]["yaw_rotmat"][:, :2, :2], 1),
        ),
        2,
    )

    normalization_factor = 1.0
    for dataset_name, value in METRIC_WAYPOINT_SPACING.items():
        if tf.strings.regex_full_match(
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0],
            f".*{dataset_name}.*",
        ):
            normalization_factor = value
    normalization_factor = tf.cast(normalization_factor, tf.float64)
    actions = actions / normalization_factor

    trajectory["action"] = actions

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    return trajectory


def old_gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    traj_len = tf.shape(trajectory["action"])[0]
    action_horizon = 4

    # compute rot matrix
    yaw = trajectory["observation"]["yaw"]
    rot_mat = tf.convert_to_tensor(
        [
            [tf.cos(yaw), -tf.sin(yaw)],
            [tf.sin(yaw), tf.cos(yaw)],
        ]
    )
    rot_mat = tf.transpose(rot_mat, [3, 2, 0, 1])[0]

    # chunk actions and recompute as relative to the start of the chunk
    pos = trajectory["observation"]["position"]
    start = tf.broadcast_to(pos[:, None], [traj_len, action_horizon, 2])
    end_indices = tf.range(traj_len)[:, None] + tf.range(1, action_horizon + 1)
    end_indices = tf.minimum(end_indices, traj_len - 1)
    end = tf.gather(pos, end_indices)
    delta = end - start
    action = tf.matmul(delta[:, :, None], rot_mat[:, None])[:, :, 0]  # * scaling_factor

    # get normalization factor
    normalization_factor = 1.0
    for dataset_name, value in METRIC_WAYPOINT_SPACING.items():
        if tf.strings.regex_full_match(
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0],
            f".*{dataset_name}.*",
        ):
            normalization_factor = value
    normalization_factor = tf.cast(normalization_factor, tf.float64)
    action = action / normalization_factor

    trajectory["action"] = action

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    return trajectory


def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 50Hz to 10Hz
    factor = 5
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def mujoco_manip_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    gripper_action = invert_gripper_actions(trajectory["action"][:, -1:] / 255)
    trajectory["action"] = tf.concat(
        (trajectory["action"][:, :6], gripper_action), axis=-1
    )
    return trajectory


def go1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def go1_real_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def aloha_pen_uncap_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def aloha_dough_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["language_instruction"] = trajectory["global_instruction"]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action_dict"]["cartesian_velocity"],
            invert_gripper_actions(trajectory["action_dict"]["gripper_position"]),
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "xgym_lift_mano": xgym_mano_dataset_transform,
    "xgym_stack_mano": xgym_mano_dataset_transform,
    "xgym_duck_mano": xgym_mano_dataset_transform,
    "xgym_single": xgym_single_dataset_transform,
    "xgym_lift_single": xgym_single_dataset_transform,
    "xgym_duck_single": xgym_single_dataset_transform,
    "xgym_stack_single": xgym_single_dataset_transform,
    "xgym_play_single": xgym_single_dataset_transform,
    "rlds_oakink": oakink_dataset_transform,
    #
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_dataset_transform,
    "taco_extra": taco_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "omnimimic_gnm_dataset": omnimimic_gnm_transform,
    "aloha_dagger_dataset": aloha_dataset_transform,
    "aloha_mobile_dataset": aloha_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    "mujoco_manip": mujoco_manip_dataset_transform,
    "go1": go1_dataset_transform,
    "go1_real_dataset": go1_real_dataset_transform,
    "a1": go1_dataset_transform,
    "aloha_pen_uncap_diverse_dataset": aloha_pen_uncap_dataset_transform,
    "aloha_new_sushi_dataset": aloha_pen_uncap_dataset_transform,
    "aloha_dough_cut_dataset": aloha_dough_dataset_transform,
    "aloha_lucy_dataset": aloha_dough_dataset_transform,
    "aloha_drawer_dataset": aloha_dough_dataset_transform,
    "aloha_pick_place_dataset": aloha_dough_dataset_transform,
    "aloha_static_dataset": aloha_dough_dataset_transform,
    "aloha_sushi_cut_full_dataset": aloha_dough_dataset_transform,
    "droid": droid_dataset_transform,
    "droid_wipe": droid_dataset_transform,
    "droid_flip_pot_upright": droid_dataset_transform,
}
