from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.cn.util import CNMeta
from crossformer.log import logger

from rich.repr import auto

import reprlib
def short_repr(cls):
    cls.__repr__ = lambda self: f"{self.__class__.__name__}()"
    return cls

class Augment(CN):
    random_resized_crop: Dict[str, Any] = default(
        dict(scale=[0.8, 1.0], ratio=[0.9, 1.1])
    )
    random_brightness: List[float] = default([0.1])
    random_contrast: List[float] = default([0.9, 1.1])
    random_saturation: List[float] = default([0.9, 1.1])
    random_hue: List[float] = default([0.05])
    augment_order: List[str] = default(
        [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ]
    )


@short_repr
class AlohaAug(Augment):
    random_resized_crop: Dict[str, Any] = default(
        dict(scale=[0.9, 1.0], ratio=[0.75, 4.0 / 3.0])
    )


@short_repr
class BridgeAug(Augment):
    random_resized_crop: Dict[str, Any] = dict(scale=[0.8, 1.0], ratio=[0.9, 1.1])


class Primary(BridgeAug):
    pass
