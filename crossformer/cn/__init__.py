import enum
from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf
from rich import print as pprint

from crossformer.cn import dataset, model
from crossformer.cn.dataset import Dataset, transform
from crossformer.cn.eval import Eval
from crossformer.cn.optimizer import Optimizer
from crossformer.cn.optimizer.lr import LearningRate
from crossformer.cn.rollout import Rollout
from crossformer.cn.util import asdataclass, CN, CS, default, store, tryex
from crossformer.cn.wab import Wandb
from crossformer.log import logger

logger.info("Importing crossformer.cn")


class Delete(CN):
    model: Any = default(
        dict(
            readouts=dict(
                bimanual=4,
                quadruped=None,
                nav=None,
                single_arm=4,
            ),
            observation_tokenizers=dict(
                bimanual=None,
                quadruped=None,
                nav=None,
            ),
            heads=dict(
                single_arm="diffusion",
                quadruped=None,
                nav=None,
            ),
        )
    )

    def __post_init__(self):
        logger.warn("TODO: was this for model only or all?")
        logger.warn("TODO: prob better way to impose structure")


class Mode(Enum):
    """training mode"""

    FULL = "full"
    HEAD = "head_only"
    LORA = "lora"
    FROZEN = "frozen"


# you cant use module.Object
# because the compiler thinks you are referencing the attribute


class Run(CN):
    """Base Config"""

    max_steps: int = 500_000
    grad_acc = None
    # max_steps = max_steps * (grad_acc or 1)

    window_size: int = 1

    action_horizon = 4  # max action horizon
    dataset: Dataset = Dataset()  # mode , window, horizon, mix
    # ( "multi", window_size, action_horizon=action_horizon, mix="xstack")

    pretrained_path = "hf://rail-berkeley/crossformer"
    pretrained_step: Optional[int] = None
    resume_path: Optional[str] = None

    wandb: Wandb = Wandb()
    wandb_resume_id: Optional[str] = None

    update: model.Model = default(
        model.Model
    )  # uncomment this line to add new observation tokenizer and action head

    skip_norm_keys: List[str] = ["proprio_bimanual, proprio_mano"]
    config_delete_keys: Optional[Delete] = None

    batch_size: Optional[int] = None
    shuffle_buffer_size: int = 10000
    # num_steps = max_steps
    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 2000
    save_dir: Union[str, Path] = os.environ.get("BAFL_SAVE", Path().expanduser())
    seed: int = 42

    frame_transform_threads: Optional[int] = None
    prefetch_num_batches: int = 64
    modality: transform.Modality = transform.Modality.MULTI
    finetuning_mode: Mode = Mode.FULL

    head_name: Optional[str] = None  # TODO why is this here ... see logger warning

    optimizer: Optimizer = Optimizer()

    eval: Eval = Eval()
    rollout: Rollout = Rollout()

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

        self.batch_size = self.dataset.batch_size

        self.optimizer.lr.decay_steps = self.max_steps
        assert (
            self.optimizer.lr.decay_steps == self.max_steps
        ), "Decay steps must match max steps"

        logger.warn("TODO: wandb disabled if debug")

        logger.info("SUGGESTION: propogate from the main cfg down to children")
        logger.warn("TODO: assert all keys that appear twice are the same")
        logger.warn("TODO: is update_config for model arch only?")
        logger.warn(f"TODO: is head_name for {Mode.HEAD}?")

    def transform_schema(self) -> Dict[str, Any]:
        """
        Transform the schema of the config
        for use in crossformer repo
        """
        return {}


TYP = Run


# help with sweepers
# https://stackoverflow.com/questions/70619014/specify-hydra-multirun-sweeps-in-a-config-file
# https://github.com/facebookresearch/hydra/issues/1376#issuecomment-1011704938


@tryex
@hydra.main(version_base="1.3.2", config_path=".", config_name=str(TYP()))
@asdataclass(TYP)
def main(cfg: Union[DictConfig, TYP]):  # ExperimentSweep. or will that be yaml

    print("done!")

    logger.warning(
        "TODO: add post_init support to attempt type casting... ie: 1e6 float to 1000000 int"
    )

    logger.warn("TODO figure out the hydra options")
    logger.warn("TODO add todo to the logger")
    logger.warn("TODO set wandb dir to $TMP or /tmp")

    # _cfg = get from absl config file
    # assert cfg == _cfg, "Configurations do not match"


if __name__ == "__main__":
    main()


"""

### Using Hydra Tips:

`python main.py job/wandb=dont algo=sac train.use_train=True`

* a/b selects config group b from parent group a
* a.b sets attribute b of group a

`python main.py -m +exp=may29_sac`

* `-m` flag is needed for multirun experiments

"""


if False:
    # how to structure so we can override the defaults of a
    # or is this better done with the post init?
    @dataclass
    class A:
        x: int = 1
        y: int = 2

    @dataclass
    class B:
        a: A = default(A)
        a.x: int = 10

    b = B()
    pprint(b)
