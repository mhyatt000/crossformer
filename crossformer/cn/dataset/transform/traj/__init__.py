from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.log import logger


class GoalRelabel(Enum):
    """Goal Relabeling Strategy"""

    NONE = "none"
    FINAL = "final"
    RANDOM = "random"
    FUTURE = "future"
    PAST = "past"
    UNIFORM = "uniform"


class TrajectoryTransform(CN):
    window_size: int = 1
    action_horizon: int = 10  # max horizon

    max_action_dim: ActionDim = ActionDim.BIMANUAL
    max_proprio_dim: ActionDim = ActionDim.BIMANUAL
    # head_to_dataset: Dict = default(HEAD_TO_DATASET)
    goal_relabling_strategy: GoalRelabel = GoalRelabel.UNIFORM  # TODO define

    task_augment_strategy: str = "delete_task_conditioning"
    task_augment_kwargs: Dict[str, Any] = default(
        {
            "keep_image_prob": 0.5,
            # If the default data loading speed is too slow, try these:
            # num_parallel_calls=16,  # for less CPU-intensive ops
        }
    )

    subsample_length: int = 100  # subsample length for episode -> trajectory

    def __post_init__(self):

        logger.warn("TODO: ensure max dims are truly the max")
        logger.warn("TODO: define goal relabeling strategies")
