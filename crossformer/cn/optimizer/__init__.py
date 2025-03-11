from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.log import logger

from .lr import LearningRate


class Optimizer(CN):
    lr: LearningRate = LearningRate()
    weight_decay: float = 0.01
    clip_gradient: float = 1.0
    frozen_keys: Optional[List[str]] = None
    grad_accumulatiion_steps: int = 1  # if using, adjust max_steps accordingly

    def __post_init__(self):
        logger.warn("TODO post init optimizer for frozen keys")
        return
        if mode == "full":
            frozen_keys = None
        elif mode == "head_only":
            frozen_keys = ("crossformer_transformer.*",)
        else:
            raise ValueError("Invalid mode")
