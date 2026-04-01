from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Literal

from rich import print
import tyro

from crossformer.cn.base import CN
from crossformer.cn.dataset import Dataset, DataSource, Head, MultiDataSource, transform
from crossformer.cn.eval import Eval
from crossformer.cn.model_factory import ModelFactory
from crossformer.cn.optim import Optimizer
from crossformer.cn.rollout import Rollout
from crossformer.cn.wab import Wandb
from crossformer.data.oxe.oxe_dataset_configs import ActionDim

log = logging.getLogger(__name__)


@dataclass()
class Train(CN):
    """Base Config"""

    debug: bool = False

    steps: int = int(5e5)
    grad_acc = None

    modality: transform.Modality = transform.Modality.MULTI
    window_size: int = 1
    head_name: str | None = None

    pretrained_path: Path | str = "hf://rail-berkeley/crossformer"
    pretrained_step: int | None = None

    wandb: Wandb = Wandb().field()
    data: Dataset = Dataset().field()
    model: ModelFactory = ModelFactory().field()
    optimizer: Optimizer = Optimizer().field()
    eval: Eval = Eval().field()
    rollout: Rollout = Rollout().field()

    verbosity: int = 1
    log_level: Literal["debug", "info", "warning", "error"] = "warning"
    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 2000
    save_dir: str | Path = os.environ.get("BAFL_SAVE", Path().home())
    seed: int = 42

    def vprint(self, *args, level: int = 1, **kwargs):
        if isinstance(args[0], str):
            a = f"[bold green]{args[0]}[/bold green]"
            args: tuple = (a, *args[1:])
        if self.verbosity >= level:
            print(*args, **kwargs)

    def set_log_level(self):
        logging.basicConfig(level=self.log_level.upper(), force=True)
        if self.verbosity == 0:
            logging.getLogger("jax").setLevel(logging.ERROR)
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            logging.getLogger("absl").setLevel(logging.ERROR)
        log.info(f"Logging level set to {self.log_level.upper()}")

    def __post_init__(self):
        self.set_log_level()
        if self.data.transform.traj.action_horizon != self.model.max_horizon():
            log.debug(
                "action horizon mismatch: "
                f"data.transform.traj ({self.data.transform.traj.action_horizon}) "
                f"vs model.max_horizon ({self.model.max_horizon()}), auto-correcting"
            )
            self.data.transform.traj.action_horizon = self.model.max_horizon()

        if self.data.transform.traj.max_action_dim != self.model.max_action_dim():
            log.debug(
                "max action dim mismatch: "
                f"data.transform.traj ({self.data.transform.traj.max_action_dim}) "
                f"vs model.max_action_dim ({self.model.max_action_dim()}), auto-correcting"
            )
            self.data.transform.traj.max_action_dim = ActionDim(self.model.max_action_dim())

        if self.optimizer.lr.decay_steps is None:
            log.debug(f"decay_steps is None, setting to {self.steps}")
            self.optimizer.lr.decay_steps = self.steps

    def transform_schema(self) -> dict[str, Any]:
        return {}


MODELS = []
TFORMS = []
DATAS = []
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


def cli() -> Train:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in CONFIGS.items()})


ET = Enum("ExperimentTypes", {k: k for k, v in CONFIGS.items()})


@dataclass()
class Experiment(Train):
    exp: ET | None = None

    def __post_init__(self):
        if self.exp:
            d = CONFIGS[self.exp.value].asdict()
            raise NotImplementedError("TODO: reverse merge with self")
            self.update(d)


TYP = Experiment


class Sweep(Train):
    sweep: dict[str, list[Any]] | None = None


def main(cfg: TYP) -> None:
    print(cfg)
    print()


if __name__ == "__main__":
    main(tyro.cli(TYP))
