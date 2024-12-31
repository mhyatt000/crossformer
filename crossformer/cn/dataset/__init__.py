from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.dataset.transform import Transform
from crossformer.cn.util import asdataclass, CN, CS, default, store
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES


class Dataset(CN):

    # OXE kwargs
    mix: Any = ""
    data_dir: Path = os.environ.get(
        "BAFL_DATA", Path("tensorflow_datasets").expanduser()
    )
    load_camera_views: Tuple[str] = ("primary", "side", "high", "left_wrist")
    load_proprio: bool = True
    load_depth: bool = False

    transform: Transform = default(Transform)
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
