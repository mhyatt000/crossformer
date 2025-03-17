from dataclasses import dataclass
import tyro
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.base import CN, default
from crossformer.cn.dataset.transform import Transform
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES

logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")


@dataclass()
class Dataset(CN):

    # dataset mix. a mix is a registered combination of datasets
    mix: Any = ""
    # location of tensorflow_datasets
    loc: Path = os.environ.get(
        "BAFL_DATA", Path("tensorflow_datasets").expanduser()
    )

    load_camera_views: List[str] = default(["primary", "side", "high", "left_wrist"])
    load_proprio: bool = True
    load_depth: bool = False

    transform: Transform = Transform().field()

    # loader kwargs
    batch_size: int = 256

    shuffle_buffer_size: int = 50_000
    balance_weights: bool = False
    traj_transform_threads: int = 48
    traj_read_threads: int = 48

    def __post__init__(self):

        # ensure that each dataset has a head
        is_valid = [
            any([d in dsets for head, dsets in HEAD_TO_DATASET.items()])
            for d, weight in OXE_NAMED_MIXES[self.mix]
        ]
        assert all(is_valid), f"Dataset in mix: {self.mix} doesn't have assigned head."




from .action import Head

@dataclass()
class DataSource(CN):
    name: str = tyro.MISSING
    head: Head = tyro.MISSING


@dataclass()
class TFDS(DataSource):
    pass


@dataclass()
class LeRobot(DataSource):
    pass


@dataclass()
class MultiDataSource(DataSource):
    """Data Mix Configuration"""

    data: List[DataSource] = tyro.MISSING
    weights: List[float] = tyro.MISSING
    head: str = Head.MULTI

    def __post_init__(self):
        msg = "Datasets and weights must be same length."
        assert len(self.data) == len(self.weights), msg


XGYM = [
    TFDS(name="xgym_duck_single", head=Head.SINGLE),
    TFDS(name="xgym_lift_single", head=Head.SINGLE),
    TFDS(name="xgym_stack_single", head=Head.SINGLE),
]
SOURCES = [] + XGYM

# multi source
MDS = [MultiDataSource(name="xgym", data=XGYM, weights=[1.0] * len(XGYM))]
SOURCES += MDS
SOURCES = {s.name: s for s in SOURCES}
