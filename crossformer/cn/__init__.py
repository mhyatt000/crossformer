"""config nodes"""

from typing import Sequence
from dataclasses import dataclass, field, Field
import enum
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from rich import print as pprint
from six import u
import tyro
from crossformer.cn.dataset import (
    DataSource,
    Head,
    MultiDataSource,
)
from crossformer.cn.dataset.action import DataSpec, DataPrep, HEAD2SPACE
from crossformer.cn.dataset import Dataset, transform
from crossformer.cn.eval import Eval
from crossformer.cn.optim import Optimizer
from crossformer.cn.rollout import Rollout
from crossformer.cn.wab import Wandb
from crossformer.cn import wab
import flax
from crossformer.cn.base import CN, default
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.model.components.action_heads import (
    ActionHead,
    DiffusionActionHead,
    L1ActionHead,
)
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec


logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")


class ModuleE(Enum):
    L1 = L1ActionHead
    DIFFUSION = DiffusionActionHead


@dataclass
class HeadFactory(CN):

    horizon: int = 4
    module: ModuleE = ModuleE.L1
    steps: int = 0  # diffusion steps

    def __post_init__(self):

        if self.module == ModuleE.DIFFUSION:
            assert self.steps, "Diffusion steps must be 1+"
        if self.module == ModuleE.L1:
            assert not self.steps, "Diffusion steps must be 0"

        h = Head[self.name.upper()]
        self.dim = HEAD2SPACE[h]

    def kwargs(self):
        d = dict(
            pool_strategy="use_map",
            clip_pred=False,
            loss_weight=1.0,
            constrain_loss_dims=True,
            readout_key=f"readout_{self.name}",
            num_preds= 0, # force to use dim * horizon
        )
        if self.module == ModuleE.DIFFUSION:
            d["diffusion_steps"] = self.steps

        return d

    def create(self):
        return ModuleSpec.create(
            self.module.value,
            action_horizon=self.horizon,
            action_dim=self.dim.value,
            **self.kwargs(),
        )


@dataclass
class ModelFactory(CN):

    im: Sequence[str] = default(["primary", "left_wrist"])
    proprio: Sequence[str] = default(["single"])

    # which heads to create
    heads: Sequence[str] = default(["single", "bimanual", "mano"])

    single: HeadFactory = HeadFactory(name="single").field()
    bimanual: HeadFactory = HeadFactory(name="bimanual").field()
    mano: HeadFactory = HeadFactory(name="mano").field()
    debug: bool = False  # y/n load model?

    def create(self) -> Dict[str, Any]:
        """create the model config"""

        tok = {k: self.make_obs_im(k) for k in self.im} | {
            k: self.make_obs_proprio(k) for k in self.proprio
        }

        heads = {
            v.name: v
            for v in [self.single, self.bimanual, self.mano]
            if v.name in self.heads
        }
        assert len(heads) > 0, "No heads selected"
        model = dict(
            observation_tokenizers=tok,
            heads={k: v.create() for k, v in heads.items()},
            readouts={k: v.horizon for k, v in heads.items()},
        )
        return {"model": model}

    def spec(self) -> Dict[str, Any]:
        "simplified keys that can be used to delete old model config"
        model = self.create()["model"]
        model = {
            "observation_tokenizers": {
                k: v["module"] for k, v in model["observation_tokenizers"].items()
            },
            "heads": {k: v["module"] for k, v in model["heads"].items()},
            "readouts": {k: v for k, v in model["readouts"].items()},
        }
        return {"model": model}

    def flatten(self) -> List[str]:
        """expand the delete config into a list of keys"""
        flattened = flax.traverse_util.flatten_dict(self.spec(), keep_empty_nodes=True)
        flattened = list(flattened.keys())
        return flattened

    def delete(self, flat):
        """delete keys from flax config tree based on flat spec"""
        """
        Delete keys from the model config based on the flattened list
        :param flat: list of keys to delete
        """

        def inside(a: List[str], b: List[str]):
            """see if a is inside b"""
            if len(a) > len(b):
                return False
            for _a, _b in zip(a, b[: len(a)]):
                if _a != _b:
                    return False
            return True

        mykeys = self.flatten() # keep any keys already shared
        # delete model: observation_tokenizers, heads, readouts
        deletespec = set([m[:2] for m in mykeys])

        for c in list(flat.keys()):
            if any([inside(m, c) for m in mykeys]):
                continue
            if any([inside(d, c) for d in deletespec]):
                print(f"del: {'.'.join(c)}")
                del flat[c]

        return flat

    def make_obs_proprio(self, key: str):
        """create observation tokenizer for proprio"""
        return ModuleSpec.create(
            LowdimObsTokenizer, obs_keys=[f"proprio_{key}"], dropout_rate=0.2
        )

    def make_obs_im(self, key: str):
        """create observation tokenizer for image"""
        return ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=[f"image_{key}"],
            task_stack_keys=[f"image_{key}"],
            task_film_keys=["language_instruction"],
            encoder=ModuleSpec.create(ResNet26FILM),
        )

    def max_horizon(self) -> int:
        h = 0
        for head in [self.single, self.bimanual, self.mano]:
            if head.name in self.heads:
                h = max(h, head.horizon)
        return h

    def max_action_dim(self) -> int:
        d = 0
        for head in [self.single, self.bimanual, self.mano]:
            if head.name in self.heads:
                d = max(d, head.dim.value)
        return d


@dataclass()
class Train(CN):
    """Base Config"""

    debug: bool = False  # mostly turns off wandb

    steps: int = int(5e6)  # n training steps
    grad_acc = None

    modality: transform.Modality = transform.Modality.MULTI  # mode of observation
    window_size: int = 1
    head_name: Optional[str] = None  # TODO why is this here ... see logger warning

    pretrained_path: Union[Path, str] = "hf://rail-berkeley/crossformer"
    pretrained_step: Optional[int] = None  # elapsed steps (if resume/restart)

    wandb: Wandb = Wandb().field()
    data: Dataset = Dataset().field()
    model: ModelFactory = ModelFactory().field()  # model specs
    optimizer: Optimizer = Optimizer().field()
    eval: Eval = Eval().field()
    rollout: Rollout = Rollout().field()

    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 2000
    save_dir: Union[str, Path] = os.environ.get("BAFL_SAVE", Path().home())
    seed: int = 42

    def __post_init__(self):

        if self.data.transform.traj.action_horizon != self.model.max_horizon():
            logger.warning(
                "WARNING: action horizon mismatch."
                f"data.transform.traj ({self.data.transform.traj.action_horizon}) "
                f"model.max_horizon ({self.model.max_horizon()})."
            )
            self.data.transform.traj.action_horizon = self.model.max_horizon()

        if self.data.transform.traj.max_action_dim != self.model.max_action_dim():
            logger.warning(
                "WARNING: max action dim mismatch."
                f"data.transform.traj ({self.data.transform.traj.max_action_dim}) "
                f"model.max_action_dim ({self.model.max_action_dim()})."
            )
            self.data.transform.traj.max_action_dim = self.model.max_action_dim()

        if self.optimizer.lr.decay_steps is None:
            logger.warning(f'WARNING: decay_steps is None, setting it to {self.steps}')
            self.optimizer.lr.decay_steps = self.steps

        logger.warn("TODO: fix grad_acc and steps")
        logger.warn("TODO: fix dataset")

        logger.info("SUGGESTION: propogate from the main cfg down to children")
        logger.warn("TODO: assert all keys that appear twice are the same")
        logger.warn("TODO: is update_config for model arch only?")

    def transform_schema(self) -> Dict[str, Any]:
        """
        Transform the schema of the config
        for use in crossformer repo
        """
        return {}


MODELS = []

# transform pipelines
TFORMS = []

DATAS = []

# experiment manager
MGRS = []


CONFIGS = []
CONFIGS += [Train(name="default")]

"""
# Single Arm Only
CONFIGS += [
    Train(name=x.name.replace("xgym", "bela"), data=x.name)
    for x in DataSource.REGISTRY.values()
    if x.head == Head.SINGLE
]

# # Human (Mano) Only
CONFIGS += [
    Train(name=x.name.replace("xgym", "bela"), data=x.name)
    for x in DataSource.REGISTRY.values()
    if x.head == Head.MANO
]
"""

CONFIGS = {x.name: x for x in CONFIGS}

# Train(
# name="bela_mano",
# # dataset=Dataset(),
# ),
# # Cross Embodiment
# Train(
# name="bela_cross",
# # dataset=Dataset(),
# ),


def cli() -> Train:
    """tyro.cli wrapper for subcommands (predefined experiment configs)"""
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in CONFIGS.items()})


# for tyro strong typing
ET = Enum("ExperimentTypes", {k: k for k, v in CONFIGS.items()})


@dataclass()
class Experiment(Train):

    # whether to use CLI or defined experiment
    exp: Optional[ET] = None

    def __post_init__(self):
        if self.exp:
            d = CONFIGS[self.exp.value].asdict()
            raise NotImplementedError("TODO: reverse merge with self")
            # if you reverse the merge then you can still apply CLI args to the chosen Experiment
            # need to be careful about the defaults somehow though

            self.update(d)


TYP = Experiment


class Sweep(Train):

    # sweep parameters
    # -- expand into list of training runs
    # -- TODO how to compose them in a modular way?
    sweep: Optional[Dict[str, List[Any]]] = None


def main(cfg: TYP) -> None:  # experiment or sweep
    pprint(cfg)
    print()


if __name__ == "__main__":
    main(tyro.cli(TYP))
    # main(cli())

    # cmd = tyro.cli(None | Experiment)
    # print(cmd)
