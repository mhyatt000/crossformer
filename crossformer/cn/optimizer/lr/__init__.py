from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.log import logger

from omegaconf import MISSING

class LearningRate(CN):
    """Learning Rate and Decay Scheduler"""

    name: str = "linear"
    init_value = 0.0
    peak_value = 3e-4
    warmup_steps = 2000
    decay_steps: int  = MISSING # max_steps
    end_value = 0.0

    def __post_init__(self):
        assert self.name in ["linear", "cosine", "exponential"]
        assert self.decay_steps is not None
        logger.warn("TODO make enum")
        self.scheudler = self.name
