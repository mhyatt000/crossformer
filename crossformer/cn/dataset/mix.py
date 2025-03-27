from dataclasses import dataclass
import tyro
from enum import Enum
import os
from pathlib import Path
from typing import *

from crossformer.cn.base import CN, default
from crossformer.cn.dataset.transform import Transform
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES

from crossformer.cn.dataset.action import Head


@dataclass
class DataSource(CN):
    REGISTRY: ClassVar[Dict[str, "DataSource"]] = {}

    name: str = tyro.MISSING
    head: Head = tyro.MISSING

    def __post_init__(self):
        self.REGISTRY[self.name] = self

    def flatten(self):
        return [(self.name, 1.0)]


@dataclass
class TFDS(DataSource):
    pass


@dataclass
class LeRobot(DataSource):
    pass


@dataclass
class MultiDataSource(DataSource):
    """Data Mix Configuration"""

    data: Sequence[DataSource] = tyro.MISSING
    weights: List[float] = tyro.MISSING
    head: str = Head.MULTI

    def __post_init__(self):
        msg = "Datasets and weights must be same length."
        assert len(self.data) == len(self.weights), msg

    def flatten(self) -> List[Tuple[str, float]]:
        """for each d in dataset, flatten recursively  and multiply the contents by its weight"""
        out = []
        for d, w in zip(self.data, self.weights):
            for name, weight in d.flatten():
                out.append((name, weight * w))
        return out


XGYM = [
    TFDS(name="xgym_duck_single", head=Head.SINGLE),
    TFDS(name="xgym_lift_single", head=Head.SINGLE),
    TFDS(name="xgym_stack_single", head=Head.SINGLE),
]

# multi source
MultiDataSource(
    name="xgym",
    data=XGYM,
    weights=[1.0] * len(XGYM),
)
