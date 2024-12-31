from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.log import logger


class Eval(CN):
    val_shuffle_buffer_size: int = 1000
    num_val_batches: int = 1

    eval_datasets: List[str] = default(["xgym_stack_single", "xgym_stack_mano"])
