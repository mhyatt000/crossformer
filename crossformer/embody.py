"""Embodiment-centric data specification.

Core abstractions:
  BodyPart   — named group of DOFs (shared across embodiments)
  Embodiment — ordered tuple of body parts defining a robot's action space
  Dataset    — recipe for loading one dataset (has exactly one embodiment)
  DataMix    — weighted composite of datasets (multiple embodiments)

Body parts are the reuse primitive.  Two embodiments that share an arm
share the *same* BodyPart instance, so the DOF embeddings are identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto, Enum
from typing import ClassVar, Iterator, Protocol, runtime_checkable, Sequence

# ---------------------------------------------------------------------------
# DOF vocabulary
# ---------------------------------------------------------------------------

MASK_ID = 0

DOF: dict[str, int] = {
    "MASK": MASK_ID,
    "j0": 1,
    "j1": 2,
    "j2": 3,
    "j3": 4,
    "j4": 5,
    "j5": 6,
    "j6": 7,
    "gripper": 8,
    "ee_x": 9,
    "ee_y": 10,
    "ee_z": 11,
    "ee_rx": 12,
    "ee_ry": 13,
    "ee_rz": 14,
    **{f"hand_j{i}": 15 + i for i in range(16)},
    "base_vx": 31,
    "base_vy": 32,
    "base_wz": 33,
    **{f"mano_{i}": 34 + i for i in range(7)},
    **{f"k3d_{i}": 41 + i for i in range(84)},
}

VOCAB_SIZE = 128


def ids(*names: str) -> tuple[int, ...]:
    """Look up DOF names → integer IDs."""
    return tuple(DOF[n] for n in names)


def slot_positions(n: int) -> tuple[float, ...]:
    """Slot positions: (0., 1., ..., n-1.)."""
    return tuple(float(i) for i in range(n))


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class HasDOFs(Protocol):
    """Anything that can report its DOF IDs."""

    @property
    def dof_ids(self) -> tuple[int, ...]: ...

    @property
    def action_dim(self) -> int: ...


@runtime_checkable
class Loadable(Protocol):
    """A data source that knows how to describe itself for loading."""

    name: str

    @property
    def embodiment(self) -> Embodiment: ...


# ---------------------------------------------------------------------------
# Source type
# ---------------------------------------------------------------------------


class SourceType(Enum):
    TFDS = auto()
    AREC = auto()
    LEROBOT = auto()


# ---------------------------------------------------------------------------
# Observation keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageObs:
    primary: str | None = None
    high: str | None = None
    side: str | None = None
    nav: str | None = None
    left_wrist: str | None = None
    right_wrist: str | None = None

    def active(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class ProprioObs:
    key: str = "proprio"
    dim: int = 8


# ---------------------------------------------------------------------------
# BodyPart
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyPart:
    """Named group of DOFs — the reuse primitive.

    >>> arm = BodyPart("arm_7dof", ("j0","j1","j2","j3","j4","j5","j6"))
    >>> arm.action_dim
    7
    """

    name: str
    dof_names: tuple[str, ...]

    @property
    def dof_ids(self) -> tuple[int, ...]:
        return ids(*self.dof_names)

    @property
    def action_dim(self) -> int:
        return len(self.dof_names)

    def __repr__(self) -> str:
        return f"BodyPart({self.name!r}, dim={self.action_dim})"


# ---------------------------------------------------------------------------
# Shared body-part catalog
# ---------------------------------------------------------------------------

# Arm + effector (shared DOF IDs — slot embedding distinguishes instances)
ARM_7DOF = BodyPart("arm_7dof", ("j0", "j1", "j2", "j3", "j4", "j5", "j6"))
GRIPPER = BodyPart("gripper", ("gripper",))
CART_POSE = BodyPart("cart_pose", ("ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"))
CART_POS = BodyPart("cart_pos", ("ee_x", "ee_y", "ee_z"))

# Hands
HAND_11 = BodyPart("hand_11", tuple(f"hand_j{i}" for i in range(11)))
HAND_16 = BodyPart("hand_16", tuple(f"hand_j{i}" for i in range(16)))
MANO_7 = BodyPart("mano_7", tuple(f"mano_{i}" for i in range(7)))

# Mobile base
BASE_2D = BodyPart("base_2d", ("base_vx", "base_vy", "base_wz"))


# ---------------------------------------------------------------------------
# Embodiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Embodiment:
    """Ordered composition of body parts.

    >>> single = Embodiment("single", (ARM_7DOF, GRIPPER))
    >>> single.action_dim
    8
    >>> single.dof_ids
    (1, 2, 3, 4, 5, 6, 7, 8)
    """

    REGISTRY: ClassVar[dict[str, Embodiment]] = {}

    name: str
    parts: tuple[BodyPart, ...]

    def __post_init__(self) -> None:
        type(self).REGISTRY[self.name] = self

    @property
    def dof_ids(self) -> tuple[int, ...]:
        out: tuple[int, ...] = ()
        for p in self.parts:
            out = out + p.dof_ids
        return out

    @property
    def action_dim(self) -> int:
        return sum(p.action_dim for p in self.parts)

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parts)

    @property
    def slot_positions(self) -> tuple[float, ...]:
        """Ordinal position for each DOF in the action vector: (0., 1., ..., n-1.)."""
        return slot_positions(self.action_dim)

    def has_part(self, part: BodyPart) -> bool:
        return part in self.parts

    def shared_parts(self, other: Embodiment) -> tuple[BodyPart, ...]:
        return tuple(p for p in self.parts if other.has_part(p))

    def __iter__(self) -> Iterator[BodyPart]:
        return iter(self.parts)

    def __repr__(self) -> str:
        parts = "+".join(p.name for p in self.parts)
        return f"Embodiment({self.name!r}, {parts}, dim={self.action_dim})"


# shared registry across all instances
Embodiment.REGISTRY = {}

# ---------------------------------------------------------------------------
# Embodiment catalog
# ---------------------------------------------------------------------------

SINGLE = Embodiment("single", (ARM_7DOF, GRIPPER))
BIMANUAL = Embodiment("bimanual", (ARM_7DOF, GRIPPER, ARM_7DOF, GRIPPER))
CART_GRIPPER = Embodiment("cart_gripper", (CART_POSE, GRIPPER))
HUMAN_SINGLE = Embodiment("human_single", (CART_POSE, MANO_7))
NAV = Embodiment("nav", (BASE_2D,))
XARM_RUKA = Embodiment("xarm_ruka", (ARM_7DOF, HAND_11))
POSE_RUKA = Embodiment("pose_ruka", (CART_POSE, HAND_11))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """Recipe for loading a single dataset.

    Each dataset maps to exactly one embodiment.

    >>> ds = Dataset("xgym_lift_single", SINGLE, SourceType.TFDS,
    ...     images=ImageObs(primary="worm", side="side", left_wrist="wrist"))
    >>> ds.action_dim
    8
    """

    REGISTRY: ClassVar[dict[str, Dataset]] = {}

    name: str
    embodiment: Embodiment
    source: SourceType = SourceType.TFDS

    images: ImageObs = field(default_factory=ImageObs)
    proprio: ProprioObs | None = None
    state_keys: tuple[str, ...] = ()

    weight: float = 1.0
    version: str | None = None
    branch: str = "main"

    def __post_init__(self) -> None:
        self.REGISTRY[self.name] = self

    @property
    def action_dim(self) -> int:
        return self.embodiment.action_dim

    @property
    def dof_ids(self) -> tuple[int, ...]:
        return self.embodiment.dof_ids

    def active_images(self) -> dict[str, str]:
        return self.images.active()

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {self.embodiment.name}, {self.source.name})"


# ---------------------------------------------------------------------------
# DataMix
# ---------------------------------------------------------------------------


@dataclass
class DataMix:
    """Weighted composite of datasets.

    >>> mix = DataMix("xgym", [ds1, ds2], [0.5, 0.5])
    >>> mix.embodiments  # unique embodiments
    """

    name: str
    datasets: list[Dataset]
    weights: list[float]

    def __post_init__(self) -> None:
        assert len(self.datasets) == len(self.weights), "datasets and weights must match"

    @property
    def embodiments(self) -> list[Embodiment]:
        seen: dict[str, Embodiment] = {}
        for ds in self.datasets:
            seen.setdefault(ds.embodiment.name, ds.embodiment)
        return list(seen.values())

    @property
    def all_parts(self) -> set[BodyPart]:
        return {p for ds in self.datasets for p in ds.embodiment.parts}

    @property
    def all_dof_ids(self) -> set[int]:
        return {d for ds in self.datasets for d in ds.dof_ids}

    @property
    def max_action_dim(self) -> int:
        return max(ds.action_dim for ds in self.datasets)

    def weighted(self) -> list[tuple[str, float]]:
        return [(ds.name, w) for ds, w in zip(self.datasets, self.weights)]

    def by_embodiment(self) -> dict[str, list[Dataset]]:
        out: dict[str, list[Dataset]] = {}
        for ds in self.datasets:
            out.setdefault(ds.embodiment.name, []).append(ds)
        return out

    def __iter__(self) -> Iterator[tuple[Dataset, float]]:
        return zip(self.datasets, self.weights)

    def __len__(self) -> int:
        return len(self.datasets)

    def __repr__(self) -> str:
        embods = ", ".join(e.name for e in self.embodiments)
        return f"DataMix({self.name!r}, n={len(self)}, embodiments=[{embods}])"

    @classmethod
    def from_pairs(cls, name: str, pairs: Sequence[tuple[str, float]]) -> DataMix:
        """Build from (dataset_name, weight) pairs using the Dataset registry."""
        datasets = [Dataset.REGISTRY[n] for n, _ in pairs]
        weights = [w for _, w in pairs]
        return cls(name, datasets, weights)


# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------

_XGYM_IMG = ImageObs(primary="worm", side="side", left_wrist="wrist")
_XGYM_PROPRIO = ProprioObs(key="proprio", dim=8)

xgym_lift = Dataset("xgym_lift_single", SINGLE, SourceType.TFDS, images=_XGYM_IMG, proprio=_XGYM_PROPRIO)
xgym_duck = Dataset("xgym_duck_single", SINGLE, SourceType.TFDS, images=_XGYM_IMG, proprio=_XGYM_PROPRIO)
xgym_stack = Dataset("xgym_stack_single", SINGLE, SourceType.TFDS, images=_XGYM_IMG, proprio=_XGYM_PROPRIO)
xgym_sweep = Dataset(
    "xgym_sweep_single",
    SINGLE,
    SourceType.AREC,
    images=_XGYM_IMG,
    proprio=_XGYM_PROPRIO,
    version="0.5.0",
    branch="to_step",
)
my_dataset = Dataset("my_dataset", SINGLE, SourceType.AREC, images=_XGYM_IMG, proprio=_XGYM_PROPRIO, version="0.5.3")

_MANO_IMG = ImageObs(primary="image")
_MANO_PROPRIO = ProprioObs(key="proprio", dim=8)
_MANO_STATE = ("cam_intr", "mano_pose", "mano_shape", "joints_3d", "joints_vis")

sweep_mano = Dataset(
    "sweep_mano",
    HUMAN_SINGLE,
    SourceType.AREC,
    images=_MANO_IMG,
    proprio=_MANO_PROPRIO,
    state_keys=_MANO_STATE,
    version="0.0.2",
    branch="to_step",
)

# ---------------------------------------------------------------------------
# Mix catalog
# ---------------------------------------------------------------------------

XGYM_SINGLE_MIX = DataMix("xgym_single", [xgym_lift, xgym_duck, xgym_stack], [1.0, 1.0, 1.0])

XGYM_ALL_MIX = DataMix("xgym_all", [xgym_lift, xgym_duck, xgym_stack, xgym_sweep], [1.0, 1.0, 1.0, 1.0])

XGYM_SWEEP_MIX = DataMix("xgym_sweep", [xgym_sweep, sweep_mano], [1.0, 1.0])
