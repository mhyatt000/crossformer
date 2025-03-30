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
    random_resized_crop: Dict[str, Any] = default(
        dict(scale=[0.8, 1.0], ratio=[0.9, 1.1])
    )


#
# TRANSFORMS
#


from typing import Sequence


@dataclass()
class FrameTransform(CN):

    # workspace (3rd person) camera is at 224x224
    resize_size: Sequence = default([224, 224])
    image_augment_kwargs: Augment = AlohaAug().field()

    # for the most CPU-intensive ops
    # threads: int  # if dloading too slow, set to 32 for (decoding, resizing, augmenting)
    num_parallel_calls: int = 200

    def __post_init__(self):
        logger.warn("TODO: set self.threads")

    def create(self, load_camera_views: Sequence[str]):
        """Create a per-view frame transform for the specified camera views."""

        aug = self.image_augment_kwargs.asdict()
        aug.pop('name')
        d = {
            "resize_size": {k: self.resize_size for k in load_camera_views},
            "num_parallel_calls": self.num_parallel_calls,
            "image_augment_kwargs": {
                k: self.image_augment_kwargs.asdict() for k in load_camera_views
            },
        }
        return d
