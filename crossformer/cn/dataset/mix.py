from __future__ import annotations

from dataclasses import dataclass, field

# from crossformer.data.oxe.oxe_standardization_transforms import  OXE_STANDARDIZATION_TRANSFORMS
import logging
from pathlib import Path
from typing import ClassVar, Sequence

import tyro

from crossformer.cn.base import CN
from crossformer.cn.dataset.types import Head
from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.oxe.oxe_dataset_mixes import DATASET_TO_HEAD
from crossformer.embody import Embodiment, HUMAN_SINGLE, SINGLE, SINGLE_GRIP_CAL
from crossformer.utils.spec import ModuleSpec

log = logging.getLogger(__name__)


@dataclass
class DataSource(CN):
    REGISTRY: ClassVar[dict[str, DataSource]] = {}
    EMBODIMENT_TO_DS: ClassVar[dict[str, list[str]]] = {}

    name: str = tyro.MISSING
    head: Head = tyro.MISSING  # which head to associate with this dataset. soon to be deprecated
    embodiment: Embodiment = tyro.MISSING  # embodiment has 1+ bodyparts. this will eventually replace self.head

    def _register(self):
        self.REGISTRY[self.name] = self
        if isinstance(self.embodiment, Embodiment):
            self.EMBODIMENT_TO_DS.setdefault(self.embodiment.name, []).append(self.name)

    def __post_init__(self):
        self._register()

    def flatten(self):
        return [(self.name, 1.0)]


@dataclass
class TFDS(DataSource):
    pass


@dataclass
class Arec(DataSource):
    version: str | None = None
    # upgrade: bool = False
    branch: str = "main"

    chunk: int = 20
    goal: bool = False
    restructure: ModuleSpec | None = None

    builder: ArrayRecordBuilder = field(init=False)
    _cache: Path = field(init=False, default=Path("~/.cache/arrayrecords").expanduser().resolve())

    def __post_init__(self):
        super().__post_init__()
        if self.version is None:
            self.version = self.infer_version()
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            branch=self.branch,
        )

    @property
    def source(self):
        meta = self.builder.meta
        writers = meta.get("writers", {})
        if writers:
            self.builder.writers = self.builder._normalize_writers(writers)
            self.builder.default_writer = "data" if "data" in self.builder.writers else next(iter(self.builder.writers))
        if {"image", "proprio"}.issubset(self.builder.writers):
            try:
                from crossformer.data.grain.datasets import MultiArrayRecordSource

                return MultiArrayRecordSource(
                    self.builder.get_source("image"),
                    self.builder.get_source("proprio"),
                    chunk=self.chunk,
                    goal=self.goal,
                )
            except Exception:
                log.exception("failed to open multisource arec for %s; falling back", self.name)
        return self.builder.source

    @property
    def root(self) -> Path:
        return self.builder.root

    @staticmethod
    def from_name(name: str) -> Arec:
        config = DataSource.REGISTRY.get(name)
        if config is None:
            raise ValueError(f"No OXE dataset config found for name: {name}")
        version = getattr(config, "version", None)
        branch = getattr(config, "branch", "main")
        chunk = getattr(config, "chunk", 50)
        head = DATASET_TO_HEAD.get(name)
        embodiment = getattr(config, "embodiment")
        restructure = getattr(config, "restructure", None)
        return Arec(
            name=name,
            version=version,
            head=head,
            embodiment=embodiment,
            branch=branch,
            chunk=chunk,
            restructure=restructure,
        )

    def create(self):
        pass

    def infer_version(self) -> str:
        log.info(f"No version specified for dataset {self.name}.")
        log.info(f"Inferring latest version for dataset {self.name}")
        path = self._cache / self.name  # no root yet
        versions = [v.name for v in path.iterdir() if v.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No versions found for dataset {self.name} in {path}")
        return sorted(versions)[-1]

    def get_version(self):
        # TODO infer version at post init
        return self.version if self.version else self.infer_version()

    @property
    def loc(self):
        return Path(self.name) / self.get_version() / self.branch

    def get_shards(self):
        shards = sorted(self.root.glob("*.arrayrecord"))
        if not shards:
            raise FileNotFoundError(f"No ArrayRecord shards found in {self.root}")
        return shards


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
        self._register()  # no need to assert multidataset
        msg = "Datasets and weights must be same length."
        assert len(self.data) == len(self.weights), msg

    def flatten(self) -> list[tuple[str, float]]:
        """for each d in dataset, flatten recursively  and multiply the contents by its weight"""
        out = []
        for d, w in zip(self.data, self.weights):
            for name, weight in d.flatten():
                out.append((name, weight * w))
        return out


##### ##### ##### #####
# defined sources
##### ##### ##### #####

_ = (TFDS(name="xgym_duck_single", head=Head.SINGLE, embodiment=SINGLE),)

XGYM = [
    Arec(name="xgym_lift_single", head=Head.SINGLE, embodiment=SINGLE, version="0.0.1", branch="main"),
]

# Registered for action.py's xgym_specs lookups; kept out of XGYM because their cache dirs
# are empty on this machine and XGYM_WEIGHTS would eagerly call .source on every entry.
_ = (
    Arec(name="xgym_stack_single", head=Head.SINGLE, embodiment=SINGLE, version="0.5.5", branch="main"),
    Arec(name="xgym_sweep_single", head=Head.SINGLE, embodiment=SINGLE, version="0.5.6", branch="main"),
    Arec(name="sweep_mano", head=Head.MANO, embodiment=HUMAN_SINGLE, version="0.0.2", branch="to_step"),
)
XGYM_WEIGHTS = [len(x.source) for x in XGYM]  # size weighted rn, not uniform
XGYM_WEIGHTS = [w / sum(XGYM_WEIGHTS) for w in XGYM_WEIGHTS]

NEW = [
    Arec(
        name="xarm_dream_100k",
        head=Head.SINGLE,
        embodiment=SINGLE_GRIP_CAL,
        version="0.0.2",
        branch="main",
        chunk=1,
        restructure=ModuleSpec.create("crossformer.data.grain.restructure:restructure_xarm_dream"),
    ),
]

HUMAN = [
    Arec(
        name="lift1_mano",
        head=Head.MANO,
        embodiment=HUMAN_SINGLE,
        version="0.0.2",  # multisource rebuild: image + proprio writers (all cams)
        branch="main",
        chunk=50,  # horizon must match the robot
        restructure=ModuleSpec.create("crossformer.data.grain.restructure:restructure_mano_window"),
    ),
]

# robot + human co-train (PoC). Both leaf arecs are registered above; make()
# builds each source independently — xgym_lift_single takes the multisource
# branch, lift1_mano the restructure branch — within the same mix.
MultiDataSource(
    name="lift_robot_human",
    data=[XGYM[0], HUMAN[0]],  # xgym_lift_single, lift1_mano
    weights=[0.8, 0.2],        # robot-heavy: human set is small + noisier
)

# 3 dated lift arecs replace the untraceable xgym_lift_single in training.
# Same schema/fingerprint as xgym_lift_single, but provenance is in the name.
LIFT = [
    Arec(name="xgym_lift_2026-05-07_1119_single", head=Head.SINGLE, embodiment=SINGLE, version="0.0.1", branch="main", chunk=50),
    Arec(name="xgym_lift_2026-05-07_1313_single", head=Head.SINGLE, embodiment=SINGLE, version="0.0.1", branch="main", chunk=50),
    Arec(name="xgym_lift_2026-05-25_1213_single", head=Head.SINGLE, embodiment=SINGLE, version="0.0.1", branch="main", chunk=50),
]
LIFT_WEIGHTS = [len(a.source) for a in LIFT]  # size-weighted within robot
LIFT_WEIGHTS = [w / sum(LIFT_WEIGHTS) for w in LIFT_WEIGHTS]

# inner: 3 robot sets proportional to record count
lift_robot = MultiDataSource(name="lift_robot3", data=LIFT, weights=LIFT_WEIGHTS)

# outer: 50/50 robot vs human
MultiDataSource(
    name="lift3_human",
    data=[lift_robot, HUMAN[0]],  # robot group + lift1_mano
    weights=[0.5, 0.5],
)

# multi source
MultiDataSource(
    name="xgym",
    data=XGYM,
    weights=XGYM_WEIGHTS,
)

# disabled along with the XGYM entries above — registry lookups would KeyError
# sweep = [DataSource.REGISTRY["xgym_sweep_single"], DataSource.REGISTRY["sweep_mano"]]
# MultiDataSource(
#     name="xgym_sweep",
#     data=sweep,
#     weights=[1.0] * len(sweep),
# )
