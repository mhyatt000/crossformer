from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.log import logger
from crossformer.utils.spec import ModuleSpec

from .common import Module
from .tokenizers import Image, LowDim, Side, Single, Tokenizer


class Head(Module):
    # hydra doesnt like non-standard type
    # TODO fix by casting obs as string for code linting
    module: str = "crossformer.model.components.heads:L1ActionHead"
    action_horizon: int = 4

    pool_strategy: str = "mean"
    clip_pred: bool = False  # add description here... see class
    # readout_key: str = MISSING # "readout"
    loss_weight: float = 1.0
    constrain_loss_dims: bool = True

    def __post_init__(self):
        self.readout_key: str = f"readout_{type(self).__name__.lower()}"


class DiffusionHead(Head):
    module: str = "crossformer.model.components.heads:DiffusionActionHead"
    diffusion_steps: int = 20


class SingleArm(Head):
    action_dim: ActionDim = ActionDim.SINGLE
    num_preds: ActionDim = ActionDim.SINGLE


class Bimanual(Head):
    action_horizon: int = 10
    action_dim: ActionDim = ActionDim.BIMANUAL
    num_preds: ActionDim = ActionDim.BIMANUAL


class SingleArmDiffusion(SingleArm, DiffusionHead):
    pass


class BimanualDiffusion(Bimanual, DiffusionHead):
    pass


class Mano(DiffusionHead):
    """Mano Head ... uses diffusion by default"""

    action_dim: ActionDim = ActionDim.DMANO_7
    num_preds: ActionDim = ActionDim.DMANO_7


class Model(CN):
    tokenizers: dict[str, Tokenizer] = default(
        {
            "side": Side,
            "single": Single,
        }
    )
    heads: dict[str, Head] = default(
        {
            "single_arm": SingleArm,
            "bimanual": Bimanual,
            "mano": Mano,
        }
    )
    readouts: dict[str, int] = default({"single_arm": 4, "mano": 4, "bimanual": 10})

    def __post_init__(self):
        # check that all heads have a readout and readout = action_horizon
        for k in self.heads:
            assert k in self.readouts, f"Head {k} missing readout"
            read, horizon = self.readouts[k], self.heads[k].action_horizon
            assert read == horizon, f"Readout {k} doesn't match head {read} != {horizon}"

        logger.warn("TODO: Model needs fix to ignore list and dict")
        logger.warn("TODO: MANO does not have a tokenizer")


class Update(Model):
    pass
