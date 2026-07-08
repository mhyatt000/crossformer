"""Protocol for observation, action, and proprioception semantics.

A camera is either mobile or stable. Stable cameras observe from a fixed frame.
Mobile cameras observe from a frame that may move with the agent, robot, or
scene.

An action is either innate or spatial. Innate actions are interpreted by their
native actuator semantics and do not declare an external spatial frame. Spatial
actions declare whether their coordinates are with respect to self, global, or
camera, and whether their spatial extent is 2D or 3D.

An action is either absolute or relative. Absolute values describe the target
state in the declared semantics. Relative values describe a delta to apply from
the current state in the declared semantics.

The same rules apply to proprioception: proprio values may be innate or spatial,
spatial values declare self/global/camera and 2D/3D, and values are absolute or
relative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CameraMobility(StrEnum):
    """Whether the camera frame is stable or mobile."""

    STABLE = "stable"
    MOBILE = "mobile"


class Meaning(StrEnum):
    """Whether a value is innate actuator state or spatial state."""

    INNATE = "innate"
    SPATIAL = "spatial"


class SpatialFrame(StrEnum):
    """Frame of reference for spatial values."""

    SELF = "self"
    GLOBAL = "global"
    CAMERA = "camera"


class SpatialDim(StrEnum):
    """Dimensionality of a spatial value."""

    D2 = "2d"
    D3 = "3d"


class ValueMode(StrEnum):
    """Whether a value is a target state or a delta."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass(frozen=True)
class Camera:
    """Observation camera contract."""

    mobility: CameraMobility


@dataclass(frozen=True)
class StableCamera(Camera):
    """Camera fixed in its observation frame."""

    mobility: CameraMobility = CameraMobility.STABLE


@dataclass(frozen=True)
class MobileCamera(Camera):
    """Camera whose observation frame may move."""

    mobility: CameraMobility = CameraMobility.MOBILE


@dataclass(frozen=True)
class ActingSignal:
    """Shared contract for action-like and proprio-like values."""

    mode: ValueMode
    meaning: Meaning


@dataclass(frozen=True)
class InnateSignal(ActingSignal):
    """Value interpreted by native actuator or sensor semantics."""

    meaning: Meaning = field(default=Meaning.INNATE, init=False)


@dataclass(frozen=True)
class SpatialSignal(ActingSignal):
    """Value interpreted in an explicit spatial frame."""

    frame: SpatialFrame
    dim: SpatialDim
    meaning: Meaning = field(default=Meaning.SPATIAL, init=False)


@dataclass(frozen=True)
class Action(ActingSignal):
    """Action command contract."""


@dataclass(frozen=True)
class InnateAction(Action):
    """Action command with native actuator semantics."""

    meaning: Meaning = field(default=Meaning.INNATE, init=False)


@dataclass(frozen=True)
class SpatialAction(Action):
    """Action command with explicit spatial semantics."""

    frame: SpatialFrame
    dim: SpatialDim
    meaning: Meaning = field(default=Meaning.SPATIAL, init=False)


@dataclass(frozen=True)
class Proprio(ActingSignal):
    """Proprioceptive value contract."""


@dataclass(frozen=True)
class InnateProprio(Proprio):
    """Proprioceptive value with native sensor semantics."""

    meaning: Meaning = field(default=Meaning.INNATE, init=False)


@dataclass(frozen=True)
class SpatialProprio(Proprio):
    """Proprioceptive value with explicit spatial semantics."""

    frame: SpatialFrame
    dim: SpatialDim
    meaning: Meaning = field(default=Meaning.SPATIAL, init=False)
