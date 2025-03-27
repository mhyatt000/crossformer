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
    first: float = 0.0  # initial value
    peak: float = 3e-4  # peak value
    warmup_steps: int = 2000  # num steps to reach peak
    decay_steps: Optional[int] = None  # max_steps
    last: float = 0.0  # final value

    def __post_init__(self):
        assert self.name in ["linear", "cosine", "exponential"]
        logger.warn(f" decay is none: { self.decay_steps is None}")
        logger.warn("TODO make scheduler")


class LRRegistry(Enum):
    linear = LearningRate()
    cosine = LearningRate(name="cosine")
    exponential = LearningRate(name="exponential")

class FreezeMode(Enum):
    """training mode"""

    FULL = "full"
    HEAD = "head_only"
    LORA = "lora"
    FROZEN = "frozen"


@dataclass()
class Optimizer(CN):
    """Optimizer Config. Which weights to learn, and how"""

    lr: LRRegistry = LRRegistry.cosine
    weight_decay: float = 0.01
    clip_gradient: float = 1.0
    frozen_keys: Optional[List[str]] = None
    grad_accumulation_steps: int = 1  # if using, adjust max_steps accordingly
    mode: FreezeMode = FreezeMode.FULL

    def __post_init__(self):
        logger.warn("TODO post init optimizer for frozen keys")
        self.lr = self.lr
