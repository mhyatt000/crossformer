from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.log import logger

from .frame import FrameTransform, PerViewFrameTransform
from .traj import TrajectoryTransform


class Modality(Enum):
    IMG = "image_conditioned"
    TEXT = "text_conditioned"
    LANG = "text_conditioned"
    MULTI = "multimodal"


class KeepProb(Enum):
    """Keep Probability for Augmentation"""

    LOW = 0.1
    MEDIUM = 0.5
    HIGH = 0.9


class Transform(CN):
    traj: TrajectoryTransform = default(TrajectoryTransform)
    frame: FrameTransform = default(PerViewFrameTransform)

    task_cond: Modality = Modality.MULTI  # alias for modality
    keep_image_prob: float = 0.5

    def __post_init__(self):

        keep: Dict[Modality, float] = {
            Modality.IMG: 1.0,
            Modality.LANG: 0.0,
            Modality.MULTI: 0.5,
        }
        if self.keep_image_prob != keep[self.task_cond]:
            self.keep_image_prob = keep[self.task_cond]
            logger.info(f"Post init override", keep_image_prob=self.keep_image_prob)

        logger.warn("TODO: separate configs from tensorflow dependency")
