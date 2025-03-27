from dataclasses import dataclass
import jax
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


from crossformer.cn.dataset.mix import DataSource, MultiDataSource


_ = Transform(name="default")


@dataclass
class Reader(CN):
    """Data Reader Configuration"""

    # location of tensorflow_datasets
    loc: Path = os.environ.get("BAFL_DATA", Path("tensorflow_datasets").expanduser())

    load_camera_views: List[str] = default(["primary", "side", "high", "left_wrist"])
    load_proprio: bool = True
    load_depth: bool = False


@dataclass
class Loader(CN):
    """Data Loader"""

    global_batch_size: int = 256
    local_batch_size: int | None = None

    shuffle_buffer: int = 50_000
    balance_weights: bool = False

    threads_traj_transform: int = 48
    threads_traj_read: int = 48
    threads_frame_transform: Optional[int] = None

    @property
    def batch_size(self):
        """retruns local batch size (per gpu)"""
        return self.global_batch_size // jax.process_count()


DataSourceE = Enum("DataSourceE", {k: v for k, v in DataSource.REGISTRY.items()})
TransformE = Enum("TransformE", {k: v for k, v in Transform.REGISTRY.items()})


@dataclass()
class Dataset(CN):
    """new dataset object consists of:
    - mix: a registered combination of datasets and sampling weights
    - transform: combination of data transforms
    - reader: data location and preloading selection rules
    - loader: batch size and parallelism settings
    """

    mix: DataSourceE = tyro.MISSING # a mix is a registered combination of datasets
    transform: TransformE = tyro.MISSING
    reader: Reader = Reader().field()
    loader: Loader = Loader().field()

    def __post__init__(self):

        # assert self.mix in DataSourceE, f"Mix {self.mix} not registered."

        # ensure that each dataset has a head
        is_valid = [
            any([d in dsets for head, dsets in HEAD_TO_DATASET.items()])
            for d, weight in OXE_NAMED_MIXES[self.mix]
        ]
        assert all(is_valid), f"Dataset in mix: {self.mix} doesn't have assigned head."

    def get_mix(self):
        return MultiDataSource.REGISTRY[self.mix]
