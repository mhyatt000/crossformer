from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.base import CN, default

logger = logging.getLogger(__name__)


#
# AUGMENTS
#


@dataclass()
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


@dataclass()
class AlohaAug(Augment):
    random_resized_crop: Dict[str, Any] = default(
        dict(scale=[0.9, 1.0], ratio=[0.75, 4.0 / 3.0])
    )


@dataclass()
class BridgeAug(Augment):
    random_resized_crop: Dict[str, Any] = default(dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]))


@dataclass()
class Primary(BridgeAug):
    pass


#
# TRANSFORMS
#


@dataclass()
class FrameTransform(CN):

    # workspace (3rd person) camera is at 224x224
    resize_size: Dict[str, List] = default({"primary": [224, 224]})
    image_augment_kwargs: Dict[str, Augment] = default({"primary": Primary})

    # for the most CPU-intensive ops
    # threads: int  # if dloading too slow, set to 32 for (decoding, resizing, augmenting)
    num_parallel_calls: int = 200

    def __post_init__(self):
        logger.warn("TODO: set self.threads")


@dataclass()
class PerViewFrameTransform(FrameTransform):
    resize_size: Dict[str, List[int]] = default(
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
