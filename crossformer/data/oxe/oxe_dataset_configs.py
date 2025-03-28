"""Dataset kwargs for Open X-Embodiment datasets.

Target configuration:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB
    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth
    proprio_encoding: Type of proprio encoding used
    action_encoding: Type of action encoding used, e.g. EEF position vs joint position control
"""

from enum import IntEnum

from crossformer.utils.spec import ModuleSpec


class ProprioEncoding(IntEnum):
    """Defines supported proprio encoding schemes for different datasets."""

    NONE = -1  # no proprio provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # joint angles + gripper open/close
    JOINT_BIMANUAL = 4  # 2 x [6 x joint angles + gripper open/close]
    POS_NAV = 5  # XY + yaw
    QUADRUPED = 6

    # single arm
    MANO = 7  # Hand 3D Pose (21 x 3) + MANO Pose (16 x 3) + cam (3x3) = 120 total


class ActionEncoding(IntEnum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close
    JOINT_POS_BIMANUAL = 3  # 2 x [6 x joint pos + gripper]
    NAV_2D = 4  # [delta_x, delta_y] waypoint
    JOINT_POS_BIMANUAL_NAV = (
        5  # 2 x [6 x joint pos + gripper] + linear base vel + angular base vel
    )
    QUADRUPED = 6

    # single arm
    MANO = 7  # Hand 3D Pose (21 x 3) + MANO Pose (16 x 3) + cam (3x3) = 120 total


class ActionDim(IntEnum):

    # OTHER
    NAV_2D = 2
    JOINT_POS = 8
    JOINT_POS_BIMANUAL_NAV = 14
    QUADRUPED = 12

    # MAIN
    SINGLE = 7
    BIMANUAL = 14
    MANO = 120

    # DEBUG
    DMANO_6 = 6  # xyz&rot
    DMANO_7 = 7  # xyz&rot grip
    # DMANO_18 = 18  # 3 palm & 15 finger params
    DMANO_35 = 35  # xyz,rot, 6*3 major knuckles and thumb , 11*1 other knuckles
    DMANO_51 = 51  # 3 palm & 48 pose params
    DMANO_52 = 52  #
    # DMANO_XYZ = 63


class ProprioDim(IntEnum):

    POS_EULER = 7
    POS_QUAT = 8
    JOINT = 8
    BIMANUAL = 14
    POS_NAV = 3
    QUADRUPED = 46

    # +1 to account for focal length which is needed for perspective mat
    DMANO_6 = ActionDim.DMANO_6 + 1
    DMANO_7 = ActionDim.DMANO_7 + 1
    DMANO_51 = ActionDim.DMANO_51 + 1
    DMANO_52 = ActionDim.DMANO_52 + 1

    MANO = DMANO_7  # current setting


# clean up data spec
proprio = {}


class PreDict:
    # _TEMPLATE = {}

    def __init__(self, keys):
        """Generate a predefined dictionary with optional overrides."""

        # self._TEMPLATE.copy()
        self.TEMPLATE = {k: None for k in keys}

    def __call__(self, **kwargs):
        """Makes the class callable, delegating to the `generate` method."""
        return self.generate(**kwargs)

    def generate(self, **kwargs):
        """Generate a dictionary with optional overrides."""

        out = self.TEMPLATE.copy()
        for key, value in kwargs.items():
            if key in out:
                out[key] = value
            else:
                raise KeyError(f"'{key}' is not a valid key in the template.")
        return out

IMOBS = PreDict(["primary", "high", "side", "nav", "left_wrist", "right_wrist"])
DIMOBS = PreDict(["primary", "secondary", "wrist"])
POBS = PreDict(["single", "mano", "bimanual", "quadruped"])

xgym = {
    "image_obs_keys": {
        "primary": "worm",
        "high": "overhead",
        "side": "side",
        "nav": None,
        "left_wrist": "wrist",
        "right_wrist": None,
    },
    "depth_obs_keys": DIMOBS(),
    "state_obs_keys": [],
    "proprio_obs_keys": POBS(single="proprio"),
    "proprio_obs_dims": {
        "mano": ProprioDim.MANO,
        "single": ProprioDim.POS_EULER,
        "bimanual": ProprioDim.BIMANUAL,
        "quadruped": ProprioDim.QUADRUPED,
    },
    "proprio_encoding": ProprioEncoding.POS_EULER,  # roll-pitch-yaw + gripper open/close
    "action_encoding": ActionEncoding.JOINT_POS,  # EEF_POS,
}

#
# TODO use some sort of metaclass to handle this cleanly?
# it is a very complicated setup

mano = {
    "image_obs_keys": IMOBS(primary="image"),
    "depth_obs_keys": DIMOBS(),
    "state_obs_keys": [
        "cam_intr",
        "mano_pose",
        "mano_shape",
        "joints_3d",
        "joints_vis",
    ],
    "proprio_obs_keys": POBS(mano="proprio"),
    # what is significant about this?
    # re: i think its needed for datasets where there is no proprio so they can fill zeros
    "proprio_obs_dims": {
        "mano": ProprioDim.MANO,
        "single": ProprioDim.POS_EULER,
        "bimanual": ProprioDim.BIMANUAL,
        "quadruped": ProprioDim.QUADRUPED,
    },
    "proprio_encoding": ProprioEncoding.MANO,
    "action_encoding": ActionEncoding.MANO,
}

# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "xgym_lift_mano": mano,
    "xgym_stack_mano": mano,
    "xgym_stack_mano": mano,
    "xgym_duck_mano": mano,
    "xgym_duck_mano": mano,
    "xgym_lift_single": xgym,
    "xgym_duck_single": xgym,
    "xgym_stack_single": xgym,
    "xgym_play_single": xgym,
    "xgym_single": xgym,
    # "rlds_oakink": mano,  # OAK INK Dataset
    #
    #
    #
    "fractal20220817_data": {
        "image_obs_keys": IMOBS(primary="image"),
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "kuka": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    "bridge_dataset": {
        "image_obs_keys": {
            "primary": "image_0",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"mano": None, "bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "mano": ProprioDim.MANO,
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "taco_play": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "high": None,
            "nav": None,
            # "left_wrist": "rgb_gripper",
            # "right_wrist": "rgb_gripper",
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": {
            "primary": "depth_static",
            "secondary": None,
            "wrist": "depth_gripper",
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "taco_extra": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "high": None,
            "nav": None,
            # "left_wrist": "rgb_gripper",
            # "right_wrist": "rgb_gripper",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "jaco_play": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "image_wrist",
            # "right_wrist": "image_wrist",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist45_image",
            # "right_wrist": "wrist45_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "roboturk": {
        "image_obs_keys": {
            "primary": "front_rgb",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": {
            "primary": None,
            "high": None,
            "nav": None,
            "left_wrist": "image",
            "right_wrist": "image",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "viola": {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "high": None,
            "nav": None,
            # "left_wrist": "eye_in_hand_rgb",
            # "right_wrist": "eye_in_hand_rgb",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "hand_image",
            # "right_wrist": "hand_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "toto": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "language_table": {
        "image_obs_keys": {
            "primary": "rgb",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "depth_image", "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None,
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"mano": None, "bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "mano": ProprioDim.MANO,
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": "depth_additional_view",
            "wrist": None,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": None,
            "wrist": "wrist_depth",
        },
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "highres_image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"mano": None, "bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "mano": ProprioDim.MANO,
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"mano": None, "bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "mano": ProprioDim.MANO,
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "nav": None,
            "high": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "nav": None,
            "high": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bc_z": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": "image2",
            "left_wrist": "hand_image",
            "right_wrist": None,
            "nav": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
            "wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "robo_net": {
        "image_obs_keys": {"primary": "image", "secondary": "image1", "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": None,
            "secondary": None,
            "high": None,
            "nav": None,
            "left_wrist": "hand_image",
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"mano": None, "bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "mano": ProprioDim.MANO,
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "uiuc_d3field": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utaustin_mutex": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image",
            # "right_wrist": "wrist_image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "finger_vision_1",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_play_fusion": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_stretch": {
        "image_obs_keys": {
            "primary": "image",
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "omnimimic_gnm_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": None,
            "nav": "image",
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_NAV,
        "action_encoding": ActionEncoding.NAV_2D,
        "override_traj_transform_kwargs": {
            "goal_relabeling_kwargs": {"max_goal_distance": 15},
            "task_augment_kwargs": {"keep_image_prob": 1.0},
        },
    },
    "aloha_dagger_dataset": {
        "image_obs_keys": {
            "primary": "cam_high",
            "secondary": "cam_low",
            "wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_mobile_dataset": {
        "image_obs_keys": {
            "primary": "cam_high",
            "secondary": None,
            "wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL_NAV,
    },
    "fmb_dataset": {
        "image_obs_keys": {
            "primary": "image_side_1",
            "secondary": "image_side_2",
            "wrist": "image_wrist_1",
        },
        "depth_obs_keys": {
            "primary": "image_side_1_depth",
            "secondary": "image_side_2_depth",
            "wrist": "image_wrist_1_depth",
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dobbe": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "roboset": {
        "image_obs_keys": {
            "primary": "image_left",
            "secondary": "image_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "rh20t": {
        "image_obs_keys": {
            "primary": "image_front",
            "secondary": "image_side_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "mujoco_manip": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "go1": {
        "image_obs_keys": {
            "primary": None,
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": "proprio"},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.QUADRUPED,
        "action_encoding": ActionEncoding.QUADRUPED,
        "override_traj_transform_kwargs": {
            "task_augment_kwargs": {"keep_image_prob": 0.0}
        },
    },
    "a1": {
        "image_obs_keys": {
            "primary": None,
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": "proprio"},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.QUADRUPED,
        "action_encoding": ActionEncoding.QUADRUPED,
        "override_traj_transform_kwargs": {
            "task_augment_kwargs": {"keep_image_prob": 0.0}
        },
    },
    "go1_real_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": None,
            "nav": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": "proprio"},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.QUADRUPED,
        "action_encoding": ActionEncoding.QUADRUPED,
        "override_traj_transform_kwargs": {
            "task_augment_kwargs": {"keep_image_prob": 0.0},
            "window_size": 1,
        },
    },
    "aloha_pen_uncap_diverse_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_new_sushi_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_dough_cut_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_lucy_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_drawer_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_pick_place_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_static_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_sushi_cut_full_dataset": {
        "image_obs_keys": {
            "primary": None,
            "high": "cam_high",
            "nav": None,
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image_left",
            # "right_wrist": "wrist_image_left",
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "filter_functions": [
            ModuleSpec.create("crossformer.data.utils.data_utils:filter_success_droid")
        ],
    },
    "droid_wipe": {
        "image_obs_keys": {
            "primary": "exterior_image_2_left",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image_left",
            # "right_wrist": "wrist_image_left",
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "droid_flip_pot_upright": {
        "image_obs_keys": {
            "primary": "exterior_image_2_right",
            "high": None,
            "nav": None,
            # "left_wrist": "wrist_image_left",
            # "right_wrist": "wrist_image_left",
            "left_wrist": None,
            "right_wrist": None,
        },
        "depth_obs_keys": DIMOBS(),
        "proprio_obs_keys": {"bimanual": None, "quadruped": None},
        "proprio_obs_dims": {
            "bimanual": ProprioDim.BIMANUAL,
            "quadruped": ProprioDim.QUADRUPED,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
}
