from dataclasses import dataclass
from typing import ClassVar, Sequence

import tyro

from crossformer.cn.base import CN
from crossformer.cn.dataset.types import Head
from crossformer.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
from crossformer.data.oxe.oxe_dataset_mixes import HEAD_TO_DATASET
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)


@dataclass
class DataSource(CN):
    REGISTRY: ClassVar[dict[str, "DataSource"]] = {}

    name: str = tyro.MISSING
    head: Head = tyro.MISSING

    def __post_init__(self):
        self.REGISTRY[self.name] = self
        members = {
            dataset
            for datasets in HEAD_TO_DATASET.values()
            for dataset in datasets
        }
        assert self.name in members, f"{self.name} missing from HEAD_TO_DATASET"
        assert self.name in OXE_DATASET_CONFIGS, f"{self.name} missing OXE config"
        assert (
            self.name in OXE_STANDARDIZATION_TRANSFORMS
        ), f"{self.name} missing OXE standardization"

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
    weights: list[float] = tyro.MISSING
    head: str = Head.MULTI

    def __post_init__(self):
        msg = "Datasets and weights must be same length."
        assert len(self.data) == len(self.weights), msg

    def flatten(self) -> list[tuple[str, float]]:
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
    TFDS(name="xgym_sweep_single", head=Head.SINGLE),
]

# multi source
MultiDataSource(
    name="xgym",
    data=XGYM,
    weights=[1.0] * len(XGYM),
)
