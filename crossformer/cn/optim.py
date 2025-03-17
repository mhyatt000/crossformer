from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.base import CN

logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")


@dataclass()
class LearningRate(CN):
    """Learning Rate and Decay Scheduler"""

    name: str = "linear"
    init_value = 0.0
    peak_value = 3e-4
    warmup_steps = 2000
    decay_steps: Optional[int] = None  # max_steps
    end_value = 0.0

    def __post_init__(self):
        assert self.name in ["linear", "cosine", "exponential"]
        logger.warn(f" decay is none: { self.decay_steps is None}")
        logger.warn("TODO make enum")
        logger.warn("TODO make scheduler")
        # self.scheduler = self.name


@dataclass()
class Optimizer(CN):
    """Optimizer Config. Which weights to learn, and how"""

    lr: LearningRate = LearningRate().field()
    weight_decay: float = 0.01
    clip_gradient: float = 1.0
    frozen_keys: Optional[List[str]] = None
    grad_accumulation_steps: int = 1  # if using, adjust max_steps accordingly

    def __post_init__(self):
        logger.warn("TODO post init optimizer for frozen keys")
        return
        if mode == "full":
            frozen_keys = None
        elif mode == "head_only":
            frozen_keys = ("crossformer_transformer.*",)
        else:
            raise ValueError("Invalid mode")
