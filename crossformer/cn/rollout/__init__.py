from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.log import logger


class Rollout(CN):
    num_envs: int = 4
    use_rollout: bool = False
