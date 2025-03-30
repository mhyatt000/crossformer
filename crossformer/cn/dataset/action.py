"""action space"""

from rich.pretty import pprint

from crossformer.data.utils.data_utils import NormalizationType
from dataclasses import dataclass, fields


from dataclasses import dataclass, asdict

from crossformer.utils.spec import ModuleSpec

# from crossformer.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from typing import *

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import tyro

from crossformer.cn.base import CN, default


logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")


class ActionSpace(Enum):

    NONE = 0
    QUADRUPED = 12
    NAV = 2  # 2D

    # single action space
    POS_EULER = 7  # EEF XYZ + roll-pitch-yaw + gripper open/close
    POS_QUAT = 8  # EEF XYZ + quaternion + gripper open/close
    JOINT = 8  # joint angles + gripper open/close

    # bimanual action space
    BI_POS_EULER = 14  # 2x(6+1)
    BI_POS_QUAT = 16  # 2x(7+1)
    BI_JOINT = 16  # 2x(7+1)

    # human action
    # TODO make better names
    # DMANO_18 = 18  # 3 palm & 15 finger params
    DMANO_7 = 7  # xyz,rot,grippiness
    DMANO_35 = 35  # xyz,rot, 6*3 major knuckles and thumb , 11*1 other knuckles
    DMANO_51 = 51  # 3 palm & 48 pose params
    DMANO_52 = 52  #
    MANO = 63  # 21 joints x 3


class ActionRep(Enum):
    """Action representation: relative or absolute"""

    RELATIVE = "rel"
    ABSOLUTE = "abs"


@dataclass
class OBS(CN):
    name: str = ""

    def __iter__(self):
        """allows conversion of OBS keys into a list or set"""
        # Iterate over the dataclass fields (which excludes ClassVars)
        return (field.name for field in fields(self) if field.name != "name")

    def items(self):
        # asdict() returns a dict of instance fields, and .items() returns its key-value pairs
        return [(k, v) for k, v in asdict(self).items() if k in self]
        return {k: v for k, v in asdict(self).items() if k in self}


@dataclass
class IMOBS(OBS):
    primary: Optional[str] = None
    high: Optional[str] = None
    side: Optional[str] = None
    nav: Optional[str] = None
    left_wrist: Optional[str] = None
    right_wrist: Optional[str] = None


@dataclass
class DIMOBS(OBS):
    primary: Optional[str] = None
    secondary: Optional[str] = None
    wrist: Optional[str] = None


@dataclass
class POBS(OBS):
    single: Optional[str] = None
    mano: Optional[str] = None
    bimanual: Optional[str] = None
    quadruped: Optional[str] = None


class Head(Enum):
    """Output head / Action Space of the model"""

    BIMANUAL = "bimanual"
    QUADRUPED = "quadruped"
    NAV = "nav"
    SINGLE = "single"
    MANO = "mano"

    MULTI = "multi"  # reserved for MultiDataSource


HEAD2SPACE = {
    Head.BIMANUAL: ActionSpace.BI_POS_EULER,
    Head.QUADRUPED: ActionSpace.QUADRUPED,
    Head.NAV: ActionSpace.NAV,
    Head.SINGLE: ActionSpace.JOINT,
    Head.MANO: ActionSpace.MANO,
}


@dataclass
class DataSpec(CN):
    """defines 1. keymapping to standard form 2. types of data 3. transforms"""

    REGISTRY: ClassVar[Dict[str, "DataSpec"]] = {}

    image_obs_keys: IMOBS = IMOBS().field()
    depth_obs_keys: DIMOBS = DIMOBS().field()
    proprio_obs_keys: POBS = POBS().field()
    state_obs_keys: List[str] = default([])

    proprio_encoding: ActionSpace = ActionSpace.NONE
    action_encoding: ActionSpace = ActionSpace.NONE

    def __post_init__(self):
        self.REGISTRY[self.name] = self

    @property
    def action_space(self):
        return self.action_encoding


@dataclass
class XGYM(DataSpec):

    image_obs_keys: IMOBS = IMOBS(
        primary="worm", high="overhead", side="side", left_wrist="wrist"
    ).field()
    depth_obs_keys: DIMOBS = DIMOBS().field()
    proprio_obs_keys: POBS = POBS(single="proprio").field()
    # state_obs_keys: []

    proprio_encoding: ActionSpace = ActionSpace.POS_EULER
    # TODO map head to possible action spaces
    action_encoding: ActionSpace = ActionSpace.JOINT

    action_rep: ActionRep = ActionRep.RELATIVE


class BasicDataSpec(DataSpec):
    """includes one image quat proprio and euler action"""

    image_obs_keys = IMOBS(primary="image").field()
    proprio_encoding = ActionSpace.POS_QUAT
    action_encoding = ActionSpace.POS_EULER


xgym_specs = {
    k: XGYM(name=k)
    for k in [
        "xgym_lift_single",
        "xgym_duck_single",
        "xgym_stack_single",
        "xgym_play_single",
        "xgym_single",
    ]
}
xgym_specs = xgym_specs | {
    k: XGYM(name=k)
    for k in [
        "xgym_lift_mano",
        "xgym_stack_mano",
        "xgym_stack_mano",
        "xgym_duck_mano",
        "xgym_duck_mano",
    ]
}

DATA_SPECS = xgym_specs


fractal = BasicDataSpec(
    name="fractal20220817_data",
)
kuka = BasicDataSpec(name="kuka")
# NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
# can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
bridge = BasicDataSpec(name="bridge_dataset", image_obs_keys=IMOBS(primary="image_0"))

taco_play = BasicDataSpec(
    name="taco_play",
    image_obs_keys=IMOBS(primary="rgb_static"),
    depth_obs_keys=DIMOBS(primary="depth_static", wrist="depth_gripper"),
)
taco_extra = BasicDataSpec(
    name="taco_extra", image_obs_keys=IMOBS(primary="rgb_static")
)
jaco_play = BasicDataSpec(name="jaco_play")

bcable = BasicDataSpec(
    name="berkeley_cable_routing",
    proprio_encoding=ActionSpace.JOINT,
    action_encoding=ActionSpace.POS_EULER,
)

roboturk = BasicDataSpec(
    name="roboturk",
    image_obs_keys=IMOBS(primary="front_rgb"),
    proprio_encoding=ActionSpace.NONE,
    action_encoding=ActionSpace.POS_EULER,
)

nyu_door = BasicDataSpec(
    name="nyu_door_opening_surprising_effectiveness",
    image_obs_keys=IMOBS(left_wrist="image", right_wrist="image"),
    proprio_encoding=ActionSpace.NONE,
    action_encoding=ActionSpace.POS_EULER,
)

viola = BasicDataSpec(
    name="viola",
    image_obs_keys=IMOBS(primary="agentview_rgb"),
    proprio_encoding=ActionSpace.JOINT,
    action_encoding=ActionSpace.POS_EULER,
)

DATA_SPECS = DATA_SPECS | {
    x.name: x
    for x in [
        fractal,
        kuka,
        bridge,
        taco_play,
        taco_extra,
        jaco_play,
        bcable,
        roboturk,
        nyu_door,
        viola,
    ]
}

print(DataSpec.REGISTRY.keys())


from crossformer.data.oxe.oxe_dataset_configs import ProprioDim
proprio_obs_dims = {
    "mano": ProprioDim.MANO,
    "single": ProprioDim.POS_EULER,
    "bimanual": ProprioDim.BIMANUAL,
    "quadruped": ProprioDim.QUADRUPED,
}


@dataclass
class DataPrep(CN):
    """bundles DataSpec with loading rules"""

    name: str = ""  # tfds name
    weight: float = 0.0  # global sampling weight

    # oxw_kwargs: Dict[str, Any]

    # the folder that contains tfds
    loc: Path | str = Path().home().expanduser() / "tensorflow_datasets"

    # which of the provided camera views to load
    load_camera_views: Sequence[str] = ()

    load_depth: bool = False  # y/n load depth images
    load_proprio: bool = False  # y/n load proprioception
    load_language: bool = True  # y/n load language instructions

    skip_norm_keys: List[str] = default([])
    force_recompute_dataset_statistics: bool = False
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL

    def __post_init__(self):

        self.loc = str(self.loc)
        assert (
            self.load_camera_views
        ), f"Must specify at least one camera view to load for {self.name}"

        if missing_keys := (
            set(self.load_camera_views) - set(self.spec.image_obs_keys)
        ):
            raise ValueError(
                f"Cannot load {self.name} with views {missing_keys} since they are not available inside {self.spec}"
            )

        try:
            _ = self.norm_mask
        except KeyError:
            raise ValueError(
                f"Cannot load {sefl.name} with unsupported action encoding {self.spec.action_encoding}"
            )

    @property
    def image_obs_keys(self):
        return {
            k: v
            for k, v in self.spec.image_obs_keys.items()
            if k in self.load_camera_views
        }

    @property
    def depth_obs_keys(self):
        d = {
            k: v
            for k, v in self.spec.depth_obs_keys.items()
            if k in self.load_camera_views
        }
        return d if self.load_depth else None

    @property
    def proprio_obs_keys(self):
        return {"primary": "proprio"} if self.load_proprio else None

    @property
    def language_key(self):
        return "language_instruction" if self.load_language else None

    @property
    def spec(self) -> DataSpec:
        return DataSpec.REGISTRY[self.name]

    @property
    def norm_mask(self):
        """
        if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
            # with EEF_POS actions, the last action dimension is gripper
            dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]

        elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS:
            # with JOINT_POS actions, last dimension is gripper
            dataset_kwargs["action_normalization_mask"] = [True] * 7 + [False]

        elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL:
            # with JOINT_POS_BIMANUAL actions, 7th and 14th dimension are gripper
            dataset_kwargs["action_normalization_mask"] = (
                [True] * 6 + [False] + [True] * 6 + [False]
            )

        elif dataset_kwargs["action_encoding"] is ActionEncoding.NAV_2D:
            # with NAV_2D actions, all dimensions are deltas
            dataset_kwargs["action_normalization_mask"] = [True] * 2

        elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL_NAV:
            # with JOINT_POS_BIMANUAL_NAV actions, 7th and 14th dimension are gripper
            dataset_kwargs["action_normalization_mask"] = (
                [True] * 6 + [False] + [True] * 6 + [False] + [True] * 2
            )

        elif dataset_kwargs["action_encoding"] is ActionEncoding.QUADRUPED:
            dataset_kwargs["action_normalization_mask"] = [True] * 12

        # MANO
        elif dataset_kwargs["action_encoding"] is ActionEncoding.MANO:
            # dataset_kwargs["action_normalization_mask"] = [True] * (ActionDim.MANO-9) + [False] * 9
            dataset_kwargs["action_normalization_mask"] = [True] * ActionDim.DMANO_7
        """

        space2norm = {
            # ActionSpace.POS_EULER: 7,
            # ActionSpace.POS_QUAT: 8,
            ActionSpace.JOINT: [True] * 7 + [False],
            # ActionSpace.BI_POS_EULER: 14,
            # ActionSpace.BI_POS_QUAT: 16,
            ActionSpace.BI_JOINT: [True] * 6 + [False] + [True] * 6 + [False],
            ActionSpace.QUADRUPED: [True] * 12,
            # ActionSpace.DMANO_35: 35,
            # ActionSpace.DMANO_51: 51,
            # ActionSpace.DMANO_52: 52,
            ActionSpace.MANO: [True] * ActionSpace.DMANO_7.value,
        }
        out = space2norm[self.spec.action_encoding]
        assert isinstance(out, list), "action space not found"
        return out

    @property
    def standardize_fn(self, oxe_fns: Dict[str, Callable]):
        """takes OXE_STANDARDIZATION_TRANSFORMS"""
        # return ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])
        return ModuleSpec.create(oxe_fns[self.name])

    def create(self,  oxe_fns):
        d = self.asdict()

        # computed not set
        d2 = {
            "action_normalization_mask": self.norm_mask,
            "data_dir": self.loc,
            #
            "language_key": self.language_key,
            "image_obs_keys": self.image_obs_keys,
            # "depth_obs_keys": self.depth_obs_keys,
            'state_obs_keys': self.spec.state_obs_keys,
        }

        if self.load_proprio:
            d2['proprio_obs_keys'] =self.proprio_obs_keys 

        h2s = {k.value: v for k, v in HEAD2SPACE.items()}
        last = {
            "standardize_fn": ModuleSpec.create(oxe_fns[self.name]),
            "proprio_obs_dims": proprio_obs_dims,
        }

        d = d | d2 | last

        pops = [
            "weight",
            "loc",
            "load_camera_views",
            "load_depth",
            "load_proprio",
            "load_language",
            # "force_recompute_dataset_statistics",
        ]

        for p in pops:
            d.pop(p)

        return d


"""
# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
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
"""
