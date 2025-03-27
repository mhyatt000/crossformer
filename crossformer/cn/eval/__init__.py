from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from tyro import MISSING

from crossformer.cn.base import CN, default

logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")

@dataclass()
class Eval(CN):
    shuffle_buffer: int = 1000
    nbatch: int = 1

    def create(self, datasets: List[str]):
        # eval_datasets: List[str] = default(["xgym_stack_single", "xgym_stack_mano"])
        return self.asdict() | dict(datasets=datasets)
