from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.log import logger

from .augment import AlohaAug, Augment, BridgeAug, Primary


class FrameTransform(CN):

    # workspace (3rd person) camera is at 224x224
    resize_size: Dict[str, List] = {"primary": [224, 224]}
    image_augment_kwargs: Dict[str, Augment] = {"primary": Primary}

    # for the most CPU-intensive ops
    # threads: int  # if dloading too slow, set to 32 for (decoding, resizing, augmenting)
    num_parallel_calls: int = 200

    def __post_init__(self):
        logger.warn("TODO: set self.threads")



class PerViewFrameTransform(FrameTransform):
    resize_size: Dict[str, List] = default(
        {
            "primary": [224, 224],
            "side": [224, 224],
            "high": [224, 224],
            "nav": [224, 224],
            "left_wrist": [224, 224],
            "right_wrist": [224, 224],
        }
    )

    image_augment_kwargs: Dict[str, Augment] = default(
        {
            "primary": BridgeAug,
            "side": BridgeAug,
            "high": AlohaAug,
            "nav": BridgeAug,
            "left_wrist": AlohaAug,
            "right_wrist": AlohaAug,
        }
    )
