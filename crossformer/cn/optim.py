from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

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
    decay_steps: int | None = None  # max_steps
    last: float = 1e-7  # final value

    def __post_init__(self):
        assert self.name in ["linear", "cosine", "exponential", "rsqrt"]
        logger.warn(f" decay is none: {self.decay_steps is None}")
        logger.warn("TODO make scheduler")

    def create(self):
        assert self.decay_steps is not None, "decay_steps must be set to train steps"
        d = self.asdict()
        d["init_value"] = d.pop("first")
        d["peak_value"] = d.pop("peak")
        d["end_value"] = d.pop("last")
        return d


@dataclass
class Cosine(LearningRate):
    name: str = "cosine"
    decay_steps: int | None = None  # max_steps

    def create(self):
        # assert decay here
        d = super().create()
        return d


@dataclass
class RSQRT(LearningRate):
    name: str = "rsqrt"
    timescale: float = 1e4


linear = LearningRate()
cosine = LearningRate(name="cosine")
exponential = LearningRate(name="exponential")
rsqrt = RSQRT()


class FreezeMode(Enum):
    """training mode"""

    FULL = "full"
    HEAD = "head_only"
    LORA = "lora"
    FROZEN = "frozen"


@dataclass()
class Optimizer(CN):
    """Optimizer Config. Which weights to learn, and how"""

    lr: LearningRate = LearningRate(name="cosine").field()
    weight_decay: float = 0.01
    clip_gradient: float = 1.0
    frozen_keys: list[str] | None = None
    grad_accumulation_steps: int = 1  # if using, adjust max_steps accordingly
    mode: FreezeMode = FreezeMode.FULL

    def __post_init__(self):
        logger.warn("TODO post init optimizer for frozen keys")

    def create(self):
        lr = self.lr.create()
        d = self.asdict()
        d.pop("lr")
        d.pop("name")
        d["learning_rate"] = lr
        d.pop("mode")
        return d
