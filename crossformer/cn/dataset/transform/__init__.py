from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from crossformer.cn.base import CN, default
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET

from .frame import FrameTransform
from .traj import TrajectoryTransform

logger = logging.getLogger(__name__)


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


@dataclass()
class Transform(CN):
    REGISTRY: ClassVar[dict[str, Transform]] = {}

    # TODO rename to seq
    traj: TrajectoryTransform = TrajectoryTransform(name="").field()

    frame: FrameTransform = FrameTransform(name="").field()

    skip_norm_keys: list[str] = default(["proprio_bimanual", "proprio_mano"])

    task_cond: Modality = Modality.MULTI  # alias for modality
    keep_image_prob: float = 0.5

    def __post_init__(self):
        self.REGISTRY[self.name] = self

        keep: dict[Modality, float] = {
            Modality.IMG: 1.0,
            Modality.LANG: 0.0,
            Modality.MULTI: 0.5,
        }
        if self.keep_image_prob != keep[self.task_cond]:
            self.keep_image_prob = keep[self.task_cond]
            logger.info("Post init override", keep_image_prob=self.keep_image_prob)

        logger.warn("TODO: separate configs from tensorflow dependency")
