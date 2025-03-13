from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.log import logger
from crossformer.utils.spec import ModuleSpec


class Module(CN):
    module: str = MISSING

    def __post_init__(self):
        if str(self) in ["module", "tokenizer"]:
            logger.warn("dont use on its own")
            logger.warn("TODO can we make this an ABC and CN?")
        else:
            assert self.module is not None

    def to_module(self):
        raise NotImplementedError
        return self.ModuleSpec(self)
