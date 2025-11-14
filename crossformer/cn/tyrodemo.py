from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import tyro

from . import CN

"""
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

"""

# import openpi.training.config as _config


@dataclass
class SubConfig:
    # tmp value
    tmp: str = "default"


@dataclass
class Config(CN):
    name: str
    x: int = 1
    y: str = "default"
    verbose: bool = False

    # Optional subconfig
    # defines additional attributes
    sub: SubConfig = dataclasses.field(default_factory=lambda: SubConfig())


def main(config: Config) -> None:
    if config.verbose:
        print("Verbose mode enabled!")
    print(f"x: {config.x}, y: {config.y}")


_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    # TrainConfig( name="pi0_aloha", model=pi0.Pi0Config(), data=LeRobotAlohaDataConfig( assets=AssetsConfig(asset_id="trossen"),),),
    Config(name="pi0_aloha", x=1, y="default", verbose=True),
    Config(name="default", x=1, y="default", verbose=True),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> Config:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


if __name__ == "__main__":
    # Parses command-line arguments into a Config instance
    # config = tyro.cli(Config)
    main(tyro.cli(Config))
