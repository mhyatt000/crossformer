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
    val_shuffle_buffer_size: int = 1000
    num_val_batches: int = 1

    # eval_datasets: List[str] = default(["xgym_stack_single", "xgym_stack_mano"])

    def create(self, datasets: List[str]):
        return self.asdict() | dict(datasets=datasets)
