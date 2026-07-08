"""Typed contracts for the crossformer data and acting pipeline.

- ``acting``: observation/action/proprio semantic contracts (frames, meanings).
- ``batch``:  post-collate training-batch shapes (TypedDicts + jaxtyping).
- ``oracle``: config-derived expected batch spec + observed-batch checker.
"""

from __future__ import annotations

from crossformer.contract.acting import (
    ActingSignal,
    Action,
    Camera,
    CameraMobility,
    InnateAction,
    InnateProprio,
    InnateSignal,
    Meaning,
    MobileCamera,
    Proprio,
    SpatialAction,
    SpatialDim,
    SpatialFrame,
    SpatialProprio,
    SpatialSignal,
    StableCamera,
    ValueMode,
)
from crossformer.contract.batch import (
    ActBlock,
    ActionHeadMasks,
    AXES,
    Batch,
    BatchInfo,
    BodypartMasks,
    CameraState,
    CameraStateMask,
    EmbodimentMasks,
    Extrinsics,
    Intrinsics,
    Masks,
    Observation,
    PadMaskDict,
    RawActions,
    RegInfo,
    SampleId,
    StateMask,
)
from crossformer.contract.oracle import (
    check_batch,
    ContractReport,
    Dims,
    expected_batch_spec,
)

__all__ = [
    # batch
    "AXES",
    "ActBlock",
    "ActingSignal",
    # acting
    "Action",
    "ActionHeadMasks",
    "Batch",
    "BatchInfo",
    "BodypartMasks",
    "Camera",
    "CameraMobility",
    "CameraState",
    "CameraStateMask",
    # oracle
    "ContractReport",
    "Dims",
    "EmbodimentMasks",
    "Extrinsics",
    "InnateAction",
    "InnateProprio",
    "InnateSignal",
    "Intrinsics",
    "Masks",
    "Meaning",
    "MobileCamera",
    "Observation",
    "PadMaskDict",
    "Proprio",
    "RawActions",
    "RegInfo",
    "SampleId",
    "SpatialAction",
    "SpatialDim",
    "SpatialFrame",
    "SpatialProprio",
    "SpatialSignal",
    "StableCamera",
    "StateMask",
    "ValueMode",
    "check_batch",
    "expected_batch_spec",
]
