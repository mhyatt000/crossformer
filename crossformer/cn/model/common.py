from __future__ import annotations

from omegaconf import MISSING

from crossformer.cn.util import CN
from crossformer.log import logger


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
