from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.log import logger
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec

from ..common import Module


class Tokenizer(Module):
    """Observation Tokenizer"""


class LowDim(Tokenizer):
    module: str = "crossformer.model.components.tokenizers:LowdimObsTokenizer"
    obs_keys: list[str] = MISSING
    dropout_rate: float = MISSING


class Single(LowDim):
    obs_keys: list[str] = default(["proprio_single"])
    dropout_rate: float = 0.2
    readout_key: str = "readout_single"


class Encoder(Enum):
    R18 = "R18"
    R34 = "R34"
    R50 = "R50"
    DINO = "DINO"
    FC = "FC"


class Image(Tokenizer):
    module: str = "crossformer.model.components.tokenizers:ImageTokenizer"
    obs_stack_keys: list[str] = MISSING
    task_stack_keys: list[str] = MISSING
    task_film_keys: list[str] = MISSING
    encoder: str = ModuleSpec.to_string(ModuleSpec.create(ResNet26FILM))
    # Encoder.R18  # architecture ie: R18


class Side(Image):
    obs_stack_keys: list[str] = defalt(["image_side"])
    task_stack_keys: list[str] = defalt(["image_side"])
    task_film_keys: list[str] = defalt(["language_instruction"])
