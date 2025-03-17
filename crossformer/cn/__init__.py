"""config nodes"""

from dataclasses import dataclass, field, Field
import enum
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from rich import print as pprint
import tyro

from crossformer.cn.dataset import DataSource, Head, MultiDataSource, SOURCES

""" from crossformer.cn import dataset, model """

from crossformer.cn.dataset import Dataset, transform
from crossformer.cn.eval import Eval
from crossformer.cn.optim import Optimizer
from crossformer.cn.rollout import Rollout
from crossformer.cn.wab import Wandb

logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")


import flax

from crossformer.cn.base import CN, default


@dataclass
class Delete(CN):
    data: Dict[str,Any] = tyro.MISSING

    def __post_init__(self):
        logger.warn("TODO: was this for model only or all?")
        logger.warn("TODO: prob better way to impose structure")

    def expand(self) -> List[str]:
        """expand the delete config into a list of keys"""
        flattened = flax.traverse_util.flatten_dict(self.data, keep_empty_nodes=True)
        flattened = list(flattened.keys())
        return flattened


@dataclass
class BasicDelete(Delete):
    """cant be a tyro CN until we know what the keys are"""

    data: Dict[str, Any] = default(
        {
            "model": {
                "readouts": {
                    "bimanual": 4,
                    "quadruped": None,
                    "nav": None,
                    "single_arm": 4,
                },
                "observation_tokenizers": {
                    "bimanual": None,
                    "quadruped": None,
                    # "high": None,
                    "nav": None,
                },
                "heads": {
                    "single_arm": "diffusion",
                    "quadruped": None,
                    "nav": None,
                },
            },
        }
    )


from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.model.components.action_heads import DiffusionActionHead, L1ActionHead
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec

UPDATE_CONFIG = dict(
    model=dict(
        observation_tokenizers=dict(
            side=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_side"],
                task_stack_keys=["image_side"],
                task_film_keys=["language_instruction"],
                encoder=ModuleSpec.create(ResNet26FILM),
            ),
            single=ModuleSpec.create(
                LowdimObsTokenizer,
                obs_keys=["proprio_single"],
                dropout_rate=0.2,
            ),
        ),
        heads=dict(
            single_arm=ModuleSpec.create(
                DiffusionActionHead,
                action_horizon=4,
                action_dim=ActionDim.SINGLE,
                num_preds=ActionDim.SINGLE,
                pool_strategy="mean",  # isnt there another/better strategy
                readout_key="readout_single_arm",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
                diffusion_steps=20,
            ),
            bimanual=ModuleSpec.create(
                L1ActionHead,
                action_horizon=4,
                action_dim=ActionDim.BIMANUAL,
                # num_preds=ActionDim.BIMANUAL,
                pool_strategy="pass",
                readout_key="readout_bimanual",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            mano=ModuleSpec.create(
                DiffusionActionHead,
                action_horizon=4,
                action_dim=ActionDim.DMANO_7,
                pool_strategy="mean",
                readout_key="readout_mano",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
                diffusion_steps=5,
            ),
        ),
        readouts=dict(single_arm=4, mano=4, bimanual=4),
    )
)


class FreezeMode(Enum):
    """training mode"""

    FULL = "full"
    HEAD = "head_only"
    LORA = "lora"
    FROZEN = "frozen"


@dataclass()
class Loader(CN):
    pass


@dataclass()
class Train(CN):
    """Base Config"""

    max_steps: int = int(5e5)  # 500k

    grad_acc = None
    # max_steps = max_steps * (grad_acc or 1)

    window_size: int = 1

    # max action horizon
    # -- TODO make this dynamic to dataset ... with factory
    action_horizon = 50

    # Spec for dataset, loader and its transform pipeline
    # -- ( "multi", window_size, action_horizon=action_horizon, mix="xstack")
    # -- TODO change name to pipe
    dataset: Dataset = Dataset().field()
    # data source
    data: DataSource = default(SOURCES["xgym"])

    pretrained_path: Union[Path, str] = "hf://rail-berkeley/crossformer"
    pretrained_step: Optional[int] = None  # elapsed steps (if resume/restart)
    resume_path: Optional[str] = None

    wandb: Wandb = Wandb().field()

    # uncomment this line to add new observation tokenizer and action head
    # -- TODO update: model.Model = default( model.Model)
    update: Dict[str, Any] = default(UPDATE_CONFIG)

    skip_norm_keys: List[str] = default(["proprio_bimanual, proprio_mano"])

    # delete: Optional[Delete] = BasicDelete.field()

    loader: Loader = Loader().field()

    shuffle_buffer_size: int = 10000

    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 2000
    save_dir: Union[str, Path] = os.environ.get("BAFL_SAVE", Path().home())
    seed: int = 42

    frame_transform_threads: Optional[int] = None
    prefetch_num_batches: int = 64

    modality: transform.Modality = transform.Modality.MULTI  # mode of observation
    finetuning_mode: FreezeMode = FreezeMode.FULL  # mode of training

    head_name: Optional[str] = None  # TODO why is this here ... see logger warning

    optimizer: Optimizer = Optimizer().field()
    eval: Eval = Eval().field()
    rollout: Rollout = Rollout().field()

    debug: bool = False

    def expand_enum(self):
        logger.warn("TODO: make recursive")
        for k, v in self.__dict__.items():
            if isinstance(v, enum.Enum):
                setattr(self, k, v.value)

    def __post_init__(self):

        logger.warn("TODO: fix grad_acc and max_steps")
        logger.warn("TODO: fix dataset")

        logger.warn("TODO: expand_enum")

        logger.warn("self.optimizer.lr.decay_steps = self.max_steps")
        # assert ( self.optimizer.lr.decay_steps == self.max_steps), "Decay steps must match max steps"

        logger.warn("TODO: wandb disabled if debug")

        logger.info("SUGGESTION: propogate from the main cfg down to children")
        logger.warn("TODO: assert all keys that appear twice are the same")
        logger.warn("TODO: is update_config for model arch only?")
        logger.warn(f"TODO: is head_name for {FreezeMode.HEAD}?")

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
# Single Arm Only
CONFIGS += [
    Train(name=x.name.replace("xgym", "bela"), data=x.name)
    for x in SOURCES.values()
    if x.head == Head.SINGLE
]

# # Human (Mano) Only
CONFIGS += [
    Train(name=x.name.replace("xgym", "bela"), data=x.name)
    for x in SOURCES.values()
    if x.head == Head.MANO
]
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
