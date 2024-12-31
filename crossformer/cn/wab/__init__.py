from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.log import logger


class WandbMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


class Wandb(CN):
    project: str = 'bafl'
    group: str = 'ssl-luc'
    entity: Optional[str] = None # should take default entity

    resume_id: Optional[str] = None
    mode: WandbMode = WandbMode.ONLINE
