"""Typed contract for the post-collate training batch.

A batch is a nested dict of arrays produced at the grain loader boundary.
At runtime these stay plain Python dicts — dicts are already JAX pytrees, so
the model consumes them directly and there is nothing to construct. The
TypedDicts below are annotation-only: they document nesting, leaf shapes, and
dtypes without changing what flows through the pipeline (which is why they are
TypedDicts and not dataclasses).

Axis-name convention
--------------------
This is the canonical axis vocabulary the codebase migrates toward. Nothing in
existing code is renamed here; this module only declares the convention and
annotates leaves with it via jaxtyping. The same names appear in ``AXES`` below
for grepping/importing.

===== ==========================================================================
Axis  Meaning
===== ==========================================================================
B     batch
TW    observation window / history time (a.k.a. window_size)
TH    action horizon / chunk time (never collapsed into a single ``T`` with TW)
A     padded unified action dim (``DataMix.max_action_dim``, e.g. 140)
V     camera views (``crossformer.embody.MAX_VIEWS`` = 3)
IH    image height
IW    image width
IC    image channels (3)
K     keypoints (14 for the robot kinematic chain; 21 for hands)
D     embedding / feature dim
S     fixed-length encoded-string bytes (32)
L     language embedding dim (512)
===== ==========================================================================

Mask semantics
--------------
Every boolean mask uses the same polarity: **True = valid / real,
False = padded / masked-out**. This holds for ``mask.act`` (True = supervised
DOF slot), ``mask.view`` (True = real camera slot), ``mask.horizon`` (True =
real chunk step), ``action_head_masks`` (True = route to this head), and every
``pad_mask_dict`` entry.

Variable key sets
-----------------
Per-run key sets under ``action``, the ``observation.proprio_*`` keys,
``pad_mask_dict``, ``mask.bodypart.*``, ``mask.embodiment.*``,
``action_head_masks.*``, and ``state`` are the *union over the mix's
embodiments*. Which keys actually appear is derivable from
``crossformer.embody.DataMix`` (its datasets' embodiments and body parts).
Those TypedDicts are therefore ``total=False`` (or use ``NotRequired``): any
given batch carries only the subset its datasets produce. Fixed-contract keys
(the ``act`` block, ``observation.image``, ``observation.timestep*``,
``dataset_name``, ``mask.act``, ``mask.view``, ``mask.horizon``) are required.

The example shapes in the field comments come from an observed run with
B=4, TW=1, TH=50, A=140, V=3, K=14, IH=IW=64; treat B/TW/TH/A as symbolic.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from jaxtyping import Bool, Float32, Int32, UInt8
import numpy as np

# ---------------------------------------------------------------------------
# Axis registry (greppable / importable form of the docstring table)
# ---------------------------------------------------------------------------

AXES: dict[str, str] = {
    "B": "batch",
    "TW": "observation window / history time",
    "TH": "action horizon / chunk time",
    "A": "padded unified action dim (DataMix.max_action_dim)",
    "V": "camera views (MAX_VIEWS = 3)",
    "IH": "image height",
    "IW": "image width",
    "IC": "image channels (3)",
    "K": "keypoints (14 robot kinematic chain; 21 hands)",
    "D": "embedding / feature dim",
    "S": "fixed-length encoded-string bytes (32)",
    "L": "language embedding dim (512)",
}

# ---------------------------------------------------------------------------
# act block — fixed contract (always present)
# ---------------------------------------------------------------------------


class ActBlock(TypedDict):
    """Padded unified action target and its per-slot routing metadata.

    Built by ``crossformer.data.grain.embody.embody_transform``. ``base`` is the
    padded action, ``id``/``view`` route each slot to a DOF and camera frame,
    and ``embody`` records the source embodiment name.
    """

    base: Float32[np.ndarray, "B TH A"]  # padded action target
    embody: UInt8[np.ndarray, "B S"]  # encoded embodiment name (S bytes)
    id: Int32[np.ndarray, "B A"]  # DOF vocab ids, 0 = MASK
    view: Int32[np.ndarray, "B A"]  # 0 = view-independent, 1..V per camera


# ---------------------------------------------------------------------------
# action — per-part raw actions (mix-dependent key set)
# ---------------------------------------------------------------------------


class RawActions(TypedDict, total=False):
    """Per-body-part raw action arrays, keyed by action-dict name.

    Keys are the union of ``PART_TO_ACTION_KEY`` values over the mix's
    embodiments; any given batch has only the subset its datasets emit.
    """

    gripper: Float32[np.ndarray, "B TH 1"]
    joints: Float32[np.ndarray, "B TH 7"]
    position: Float32[np.ndarray, "B TH 3"]  # ee xyz
    orientation: Float32[np.ndarray, "B TH 3"]  # ee rxyz
    pose: Float32[np.ndarray, "B TH 6"]  # ee xyz + rxyz
    base: Float32[np.ndarray, "B TH 3"]  # mobile base vx/vy/wz
    hand: Float32[np.ndarray, "B TH 16"]  # up to 16 hand joints
    mano: Float32[np.ndarray, "B TH 7"]
    k3ds: Float32[np.ndarray, "B TH 63"]  # 21 hand keypoints x xyz
    kp2d: Float32[np.ndarray, "B TH 30"]  # 10 landmarks x (u, v, vis)
    cam_intr: Float32[np.ndarray, "B TH 4"]
    cam_extr: Float32[np.ndarray, "B TH 9"]
    kp3dc_robot: Float32[np.ndarray, "B TH V K 3"]  # per-view camera-frame kp
    kp3dw_robot: Float32[np.ndarray, "B TH K 3"]  # world-frame kp


# ---------------------------------------------------------------------------
# action_head_masks / per-head routing (mix-dependent key set)
# ---------------------------------------------------------------------------


class ActionHeadMasks(TypedDict, total=False):
    """Per-head routing flags: True = this sample is supervised by that head.

    Keys are the action-head names present in the mix.
    """

    mano: Bool[np.ndarray, "B 1"]
    single_arm: Bool[np.ndarray, "B 1"]


# ---------------------------------------------------------------------------
# info block
# ---------------------------------------------------------------------------


# functional syntax: "global" is a Python keyword, so it can't be a class field
SampleId = TypedDict(
    "SampleId",
    {
        "episode": Int32[np.ndarray, "B 1"],
        "global": Int32[np.ndarray, "B 1"],
        "step": Int32[np.ndarray, "B 1"],
    },
)


class RegInfo(TypedDict, total=False):
    iou: Float32[np.ndarray, "B 1"]


class BatchInfo(TypedDict, total=False):
    """Bookkeeping carried alongside the batch (not fed to the model)."""

    dataset_name: UInt8[np.ndarray, "B S"]  # encoded dataset name
    id: SampleId
    len: Int32[np.ndarray, "B 1"]
    reg: RegInfo


# ---------------------------------------------------------------------------
# mask block
# ---------------------------------------------------------------------------


class BodypartMasks(TypedDict, total=False):
    """Presence of each body part in the sample (mix-dependent keys)."""

    gripper: Bool[np.ndarray, "B 1"]
    joints: Bool[np.ndarray, "B 1"]
    position: Bool[np.ndarray, "B 1"]
    orientation: Bool[np.ndarray, "B 1"]
    kp3dc_robot: Bool[np.ndarray, "B 1"]
    kp3dw_robot: Bool[np.ndarray, "B 1"]


class EmbodimentMasks(TypedDict, total=False):
    """Presence of each embodiment in the sample (mix-dependent keys)."""

    single: Bool[np.ndarray, "B 1"]


class CameraStateMask(TypedDict, total=False):
    w2c: Bool[np.ndarray, "B TH V"]  # per-view extrinsics validity


class StateMask(TypedDict, total=False):
    extr: CameraStateMask


class Masks(TypedDict, total=False):
    """Validity / routing masks. True = valid/real everywhere.

    ``act``, ``view``, and ``horizon`` are the fixed contract; the rest vary
    with the mix. (A ``total=False`` TypedDict cannot mark a subset required, so
    the always-present keys are documented here and asserted by producers.)
    """

    act: Bool[np.ndarray, "B A"]  # required — True = supervised DOF slot
    view: Bool[np.ndarray, "B V"]  # required — True = real camera slot
    horizon: Bool[np.ndarray, "B TW TH"]  # required — True = real chunk step
    action_head_masks: ActionHeadMasks
    bodypart: BodypartMasks
    embodiment: EmbodimentMasks
    proprio_kp3dc_robot: Bool[np.ndarray, "B TH V K"]
    proprio_kp3dw_robot: Bool[np.ndarray, "B TH K"]
    state: StateMask
    timestep_pad_mask: Bool[np.ndarray, "B TW"]


# ---------------------------------------------------------------------------
# observation block
# ---------------------------------------------------------------------------


class PadMaskDict(TypedDict, total=False):
    """
    TODO Deprecate in favor of Masks
    Per-modality presence masks over the observation window (TW).

    Keys are the union over the mix's proprio modalities plus ``image`` /
    ``timestep``; True = the modality is real at that window step.
    """

    image: Bool[np.ndarray, "B TW"]
    timestep: Bool[np.ndarray, "B TW"]
    proprio_gripper: Bool[np.ndarray, "B TW"]
    proprio_joints: Bool[np.ndarray, "B TW"]
    proprio_orientation: Bool[np.ndarray, "B TW"]
    proprio_position: Bool[np.ndarray, "B TW"]
    proprio_kp3dc_robot: Bool[np.ndarray, "B TW"]
    proprio_kp3dw_robot: Bool[np.ndarray, "B TW"]


class Observation(TypedDict):
    """Model inputs: stacked views + per-modality proprio over the window.

    ``image`` and ``timestep*`` are the fixed contract; ``proprio_*`` keys are
    ``NotRequired`` because the set is the union over the mix's modalities.
    """

    image: UInt8[np.ndarray, "B TW V IH IW IC"]  # stacked camera views
    timestep: Int32[np.ndarray, "B TW"]
    timestep_pad_mask: Bool[np.ndarray, "B TW"]
    pad_mask_dict: NotRequired[PadMaskDict]
    proprio_gripper: NotRequired[Float32[np.ndarray, "B TW 1"]]
    proprio_joints: NotRequired[Float32[np.ndarray, "B TW 7"]]
    proprio_orientation: NotRequired[Float32[np.ndarray, "B TW 3"]]
    proprio_position: NotRequired[Float32[np.ndarray, "B TW 3"]]
    proprio_kp3dc_robot: NotRequired[Float32[np.ndarray, "B TW V K 3"]]
    proprio_kp3dw_robot: NotRequired[Float32[np.ndarray, "B TW K 3"]]


# ---------------------------------------------------------------------------
# state block (camera intrinsics / extrinsics, mix-dependent)
# ---------------------------------------------------------------------------


class Extrinsics(TypedDict, total=False):
    w2c: Float32[np.ndarray, "B TH V 4 4"]  # world-to-camera per view


class Intrinsics(TypedDict, total=False):
    K: Float32[np.ndarray, "B TH V 3 3"]  # camera matrix per view


class CameraState(TypedDict, total=False):
    """Camera calibration carried for spatial supervision (mix-dependent)."""

    extr: Extrinsics
    intr: Intrinsics


# ---------------------------------------------------------------------------
# top-level batch
# ---------------------------------------------------------------------------


class Batch(TypedDict):
    """Post-collate training batch.

    Required keys are the fixed contract; ``NotRequired`` keys vary with the
    data mix. ``task`` is often an empty dict and only present for some mixes.
    """

    act: ActBlock
    observation: Observation
    mask: Masks
    action: NotRequired[RawActions]

    info: NotRequired[BatchInfo]
    state: NotRequired[CameraState]

    dataset_name: UInt8[np.ndarray, "B S"]  # encoded dataset name
    language_instruction: Float32[np.ndarray, "B L"]  # language embedding
    action_head_masks: NotRequired[ActionHeadMasks]
    task: NotRequired[dict[str, np.ndarray]]  # often empty


__all__ = [
    "AXES",
    "ActBlock",
    "ActionHeadMasks",
    "Batch",
    "BatchInfo",
    "BodypartMasks",
    "CameraState",
    "CameraStateMask",
    "EmbodimentMasks",
    "Extrinsics",
    "Intrinsics",
    "Masks",
    "Observation",
    "PadMaskDict",
    "RawActions",
    "RegInfo",
    "SampleId",
    "StateMask",
]
