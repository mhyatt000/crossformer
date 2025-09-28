from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

# from crossformer.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from typing import Callable, ClassVar, Sequence

import tyro

from crossformer.cn.base import CN, default
from crossformer.cn.dataset.mix import DataSource
from crossformer.cn.dataset.types import ActionRep, ActionSpace, HEAD2SPACE
from crossformer.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS, ProprioDim
from crossformer.data.oxe.oxe_dataset_mixes import HEAD_TO_DATASET
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)
from crossformer.data.utils.data_utils import NormalizationType
from crossformer.utils.spec import ModuleSpec


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
    primary: str | None = None
    high: str | None = None
    side: str | None = None
    nav: str | None = None
    left_wrist: str | None = None
    right_wrist: str | None = None


@dataclass
class DIMOBS(OBS):
    primary: str | None = None
    secondary: str | None = None
    wrist: str | None = None


@dataclass
class POBS(OBS):
    """proprio obs"""

    single: str | None = None
    mano: str | None = None
    bimanual: str | None = None
    quadruped: str | None = None


@dataclass
class DataSpec(CN):
    """defines 1. keymapping to standard form 2. types of data 3. transforms"""

    REGISTRY: ClassVar[dict[str, DataSpec]] = {}

    image_obs_keys: IMOBS = IMOBS().field()
    depth_obs_keys: DIMOBS = DIMOBS().field()
    proprio_obs_keys: POBS = POBS().field()
    state_obs_keys: list[str] = default([])

    proprio_encoding: ActionSpace = ActionSpace.NONE
    action_encoding: ActionSpace = ActionSpace.NONE

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init:
            parent_post_init()
        self.REGISTRY[self.name] = self
        assert self.name in OXE_DATASET_CONFIGS, f"{self.name} missing OXE config"
        members = {d for sub in HEAD_TO_DATASET.values() for d in sub}
        assert self.name in members, f"{self.name} missing from HEAD_TO_DATASET"
        assert self.name in OXE_STANDARDIZATION_TRANSFORMS, (
            f"{self.name} missing OXE standardization"
        )

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

    proprio_encoding: ActionSpace = ActionSpace.JOINT
    # TODO map head to possible action spaces
    action_encoding: ActionSpace = ActionSpace.JOINT

    action_rep: ActionRep = ActionRep.ABSOLUTE

    freq_data: int = 50  # data collection frequency hz
    freq_train: int = 50  # training frequency hz

    def __post_init__(self):
        super().__post_init__()

        head = DataSource.REGISTRY[self.name].head
        space = HEAD2SPACE.get(head, None)
        msg = f"action_encoding {self.action_encoding} does not match head {head} for {self.name}"
        assert self.action_encoding == space, msg


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
        "xgym_sweep_single",
        # "xgym_single",
    ]
}

DATA_SPECS = xgym_specs

proprio_obs_dims = {
    "mano": ProprioDim.MANO,
    "single": ProprioDim.JOINT,
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

    skip_norm_keys: list[str] = default([])
    force_recompute_dataset_statistics: bool = tyro.MISSING
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL

    def __post_init__(self):
        self.loc = str(self.loc)
        assert self.load_camera_views, (
            f"Must specify at least one camera view to load for {self.name}"
        )

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
                f"Cannot load {self.name} with unsupported action encoding {self.spec.action_encoding}"
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
        return {"single_arm": "proprio"} if self.load_proprio else None

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
            ActionSpace.POS_EULER: [True] * 6 + [False],
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
    def standardize_fn(self, oxe_fns: dict[str, Callable]):
        """takes OXE_STANDARDIZATION_TRANSFORMS"""
        # return ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])
        return ModuleSpec.create(oxe_fns[self.name])

    def create(self, oxe_fns):
        d = self.asdict()

        # computed not set
        d2 = {
            "action_normalization_mask": self.norm_mask,
            "data_dir": self.loc,
            #
            "language_key": self.language_key,
            "image_obs_keys": self.image_obs_keys,
            # "depth_obs_keys": self.depth_obs_keys,
            "state_obs_keys": self.spec.state_obs_keys,
        }

        if self.load_proprio:
            d2["proprio_obs_keys"] = self.proprio_obs_keys

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
