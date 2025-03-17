from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.base import CN

@dataclass()
class Rollout(CN):
    num_envs: int = 4
    use_rollout: bool = False
