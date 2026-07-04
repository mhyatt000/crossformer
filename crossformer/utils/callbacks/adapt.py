"""Map bundled slot actions onto canonical DOF orders for viz/rast callbacks."""

from __future__ import annotations

from typing import Any

import numpy as np

from crossformer.embody import DOF
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer

JOINT_NAMES = tuple(f"j{i}" for i in range(7))
JOINT_IDS = tuple(DOF[name] for name in JOINT_NAMES)
JOINT_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(JOINT_IDS)}
POS_NAMES = ("ee_x", "ee_y", "ee_z")
POS_IDS = tuple(DOF[name] for name in POS_NAMES)
POS_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(POS_IDS)}
RAST_NAMES = (*JOINT_NAMES, "gripper")
RAST_IDS = tuple(DOF[name] for name in RAST_NAMES)
RAST_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(RAST_IDS)}


def _sample_mask(mask: Any, batch_size: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.bool_)
    if arr.ndim == 0:
        return np.full(batch_size, bool(arr), dtype=np.bool_)
    if arr.shape[0] != batch_size:
        raise ValueError(f"Expected mask batch {batch_size}, got {arr.shape}")
    return arr.reshape(batch_size, -1).any(axis=-1)


def _embodiment_mask(batch, name: str) -> np.ndarray | None:
    emb = batch.get("mask", {}).get("embodiment", {})
    if name not in emb:
        return None
    batch_size = np.asarray(batch["act"]["id"]).shape[0]
    return _sample_mask(emb[name], batch_size)


def _canonical_arrays(act, flow):
    base = np.asarray(act["base"], dtype=np.float32)
    dof_ids = np.asarray(act["id"])
    flow = np.asarray(flow, dtype=np.float32)
    if base.ndim == 3:
        base = base[:, None, :, :]
    if base.ndim != 4:
        raise ValueError(f"Expected act.base ndim 3 or 4, got {base.shape}")
    if flow.ndim != 5:
        raise ValueError(f"Expected flow ndim 5, got {flow.shape}")
    if dof_ids.ndim != 2:
        raise ValueError(f"Expected act.id ndim 2, got {dof_ids.shape}")
    if base.shape[0] != dof_ids.shape[0] or flow.shape[1] != dof_ids.shape[0]:
        raise ValueError(f"Batch mismatch: base={base.shape} flow={flow.shape} dof_ids={dof_ids.shape}")
    return base, flow, dof_ids


def _copy_canonical_slots(base, flow, dof_ids, dof_id_to_idx, sample_mask):
    keep = []
    out_dim = len(dof_id_to_idx)
    base_joint = np.zeros((*base.shape[:-1], out_dim), dtype=np.float32)
    flow_joint = np.zeros((*flow.shape[:-1], out_dim), dtype=np.float32)
    for b, row in enumerate(dof_ids):
        if sample_mask is not None and not bool(sample_mask[b]):
            continue
        has_target = False
        for src, dof_id in enumerate(row):
            dst = dof_id_to_idx.get(int(dof_id))
            if dst is None:
                continue
            has_target = True
            base_joint[b, ..., dst] = base[b, ..., src]
            flow_joint[:, b, ..., dst] = flow[:, b, ..., src]
        if has_target:
            keep.append(b)
    return base_joint, flow_joint, keep


def adapt_canonical_batch(act, flow, dof_id_to_idx, sample_mask: Any | None = None):
    """Map bundled slot actions to a canonical DOF order."""
    base, flow, dof_ids = _canonical_arrays(act, flow)
    if sample_mask is not None:
        sample_mask = _sample_mask(sample_mask, dof_ids.shape[0])

    base_joint, flow_joint, keep = _copy_canonical_slots(base, flow, dof_ids, dof_id_to_idx, sample_mask)
    if not keep:
        return None, None
    keep = np.asarray(keep, dtype=np.int32)
    return {"act": {"base": base_joint[keep]}, "predict": flow_joint[:, keep]}, keep


def adapt_viz_batch(batch, flow):
    """Split viz inputs into robot joints and human xyz tracks."""
    robot_mask = _embodiment_mask(batch, "single")
    human_mask = _embodiment_mask(batch, "human_single")
    if robot_mask is None and human_mask is not None:
        robot_mask = ~human_mask

    robot, robot_keep = adapt_canonical_batch(batch["act"], flow, JOINT_ID_TO_IDX, sample_mask=robot_mask)
    robot_xyz, robot_xyz_keep = adapt_canonical_batch(batch["act"], flow, POS_ID_TO_IDX, sample_mask=robot_mask)
    human, human_keep = adapt_canonical_batch(batch["act"], flow, POS_ID_TO_IDX, sample_mask=human_mask)

    if robot is None and robot_xyz is None and human is None:
        return None, None

    out = {}
    if robot is not None:
        out |= robot
    if robot_xyz is not None:
        out["robot_xyz"] = {"base": robot_xyz["act"]["base"], "predict": robot_xyz["predict"]}
    if human is not None:
        out["human_xyz"] = {"base": human["act"]["base"], "predict": human["predict"]}
    keep = {"robot": robot_keep, "robot_xyz": robot_xyz_keep, "human": human_keep}
    return out, keep


def adapt_rast_batch(act, flow):
    """Map bundled actions to canonical j0..j6+gripper order for RastCallback."""
    return adapt_canonical_batch(act, flow, RAST_ID_TO_IDX)


def denorm_canonical(arr: np.ndarray, denorm: ActionBatchDenormalizer, ds_name: str, dof_ids: np.ndarray) -> np.ndarray:
    """Denormalize canonical actions with explicit DOF ids."""
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.stack([denorm.denormalize_slot(row, dof_ids, ds_name) for row in flat], axis=0)
    return out.reshape(arr.shape)
