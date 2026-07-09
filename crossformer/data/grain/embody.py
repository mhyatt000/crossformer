"""Embodiment-aware action construction for grain pipeline.

Per-sample transform that builds a padded action block (`act.base`),
DOF identity vector (`act.id`), and boolean mask (`mask.act`) by
randomly sampling which body parts to include or mask, in random order.

Wired into the grain pipeline via GrainDataFactory in loader.py.
"""

from __future__ import annotations

import numpy as np

from crossformer.embody import (
    BodyPart,
    Embodiment,
    MASK_ID,
)

# ---------------------------------------------------------------------------
# DOF ID sentinels (in act.id)
# ---------------------------------------------------------------------------

MASK_DOF = MASK_ID  # body part excluded — zeros in act.base, masked in loss

# ---------------------------------------------------------------------------
# Sampling modes per body part
# ---------------------------------------------------------------------------

INCLUDE = 0  # real DOF vocab IDs
MASK = 1  # excluded


def sample_modes(
    n_parts: int,
    rng: np.random.Generator,
    mask_prob: float = 0.1,
) -> list[int]:
    """Sample INCLUDE/MASK per body part, at least one included."""
    modes = rng.choice([INCLUDE, MASK], size=n_parts, p=[1 - mask_prob, mask_prob]).tolist()
    if all(m == MASK for m in modes):
        modes[rng.integers(n_parts)] = INCLUDE
    return modes


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_action_block(
    parts: list[BodyPart],
    actions: list[np.ndarray],
    modes: list[int],
    order: list[int],
    max_a: int,
    valid_masks: list[np.ndarray | None] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build act.base, act.id, mask.act from body parts in random order.

    Args:
        parts: body parts for this embodiment.
        actions: per-part arrays, each (H, D_i) float.
        modes: per-part INCLUDE/MASK from sample_modes.
        order: permutation indices for body part ordering.
        max_a: max action dim. pad width
        valid_masks: optional per-part per-DOF validity, each (D_i,) bool or None.
            ANDed into mask.act for INCLUDEd parts. None entry = all valid.

    Returns:
        dict with act.base (H, max_a) float32,
                  act.id  (max_a,)   int32,
                  act.view (max_a,)  int32 (0 = NO_VIEW, 1..MAX_VIEWS per view),
                  mask.act (max_a,)  bool.
    """
    assert len(parts) == len(actions) == len(modes) == len(order)
    vms: list[np.ndarray | None] = list(valid_masks) if valid_masks is not None else [None] * len(parts)
    assert len(vms) == len(parts)
    H = actions[0].shape[0]

    act_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []
    view_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []

    for idx in order:
        part, act, mode = parts[idx], actions[idx], modes[idx]
        D = part.action_dim
        assert act.shape == (H, D), f"{part.name}: expected ({H},{D}), got {act.shape}"

        if mode == MASK:
            act_chunks.append(np.zeros((H, D), dtype=np.float32))
            id_chunks.append(np.full(D, MASK_DOF, dtype=np.int32))
            view_chunks.append(np.zeros(D, dtype=np.int32))
            valid_chunks.append(np.zeros(D, dtype=bool))
        else:  # INCLUDE
            # camera-frame keypoints are NaN for uncalibrated views (see
            # project_world_to_cam); zero-fill so masked slots can't poison the
            # loss (NaN * 0 = NaN) — mask.act carries their validity.
            act_chunks.append(np.nan_to_num(act.astype(np.float32)))
            id_chunks.append(np.array(part.dof_ids, dtype=np.int32))
            view_chunks.append(np.full(D, part.view, dtype=np.int32))
            vm = vms[idx]
            if vm is None:
                valid_chunks.append(np.ones(D, dtype=bool))
            else:
                vm = np.asarray(vm, dtype=bool)
                assert vm.shape == (D,), f"{part.name}: valid_mask {vm.shape} != ({D},)"
                valid_chunks.append(vm)

    used = sum(p.action_dim for p in parts)
    pad_d = max_a - used

    act_base = np.concatenate(act_chunks, axis=-1)  # (H, used)
    act_id = np.concatenate(id_chunks)  # (used,)
    act_view = np.concatenate(view_chunks)  # (used,)
    valid = np.concatenate(valid_chunks)  # (used,)

    if pad_d > 0:
        act_base = np.pad(act_base, ((0, 0), (0, pad_d)))
        act_id = np.pad(act_id, (0, pad_d), constant_values=MASK_DOF)
        act_view = np.pad(act_view, (0, pad_d))
        valid = np.pad(valid, (0, pad_d))

    mask_act = (act_id != MASK_DOF) & valid

    return {
        "act": {
            "base": act_base,
            "id": act_id,
            "view": act_view,
        },
        "mask": {"act": mask_act},
    }


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


# body part name → action dict key in the decoded data
PART_TO_ACTION_KEY: dict[str, str] = {
    "arm_7dof": "joints",
    "gripper": "gripper",
    "cart_pos": "position",
    "cart_ori": "orientation",
    "cart_pose": "pose",
    "kp3d_21": "k3ds",
    "mano_7": "mano",
    "mano_48": "mano",
    "hand_11": "hand",
    "hand_16": "hand",
    "base_2d": "base",
    "kp2d_arm10dof": "kp2d",
    "cam_intr": "cam_intr",
    "cam_extr": "cam_extr",
    "kp3dc": "kp3dc_robot",
    "kp3dc_hand": "kp3dc_hand",
    "kp2d_hand": "kp2d_hand",
}


def extract_part_actions(
    action_dict: dict[str, np.ndarray],
    embodiment: Embodiment,
    key_map: dict[str, str] | None = None,
) -> list[np.ndarray | None]:
    """Pull per-body-part action arrays from a flat action dict.

    Args:
        action_dict: e.g. {"joints": (H,7), "gripper": (H,1), ...}
        embodiment: the dataset's embodiment.
        key_map: override for PART_TO_ACTION_KEY.

    Returns:
        list of (H, D_i) arrays, one per body part in embodiment.expanded.
        None where the action key is absent (part must be force-masked).
        Per-view parts slice their view from the (H, V, ...) array; trailing
        dims are flattened (e.g. kp3dc (H, 14, 3) -> (H, 42)).
    """
    km = key_map or PART_TO_ACTION_KEY
    out: list[np.ndarray | None] = []
    for part in embodiment.expanded:
        key = km.get(part.name)
        if key is None:
            raise KeyError(f"no action key mapping for body part {part.name!r}")
        if key not in action_dict:
            out.append(None)
            continue
        arr = np.asarray(action_dict[key])
        if part.view > 0:
            arr = arr[:, part.view - 1]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        out.append(arr)
    return out


def build_action_norm_mask(
    action_dict: dict[str, np.ndarray],
    embodiment: Embodiment,
    key_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Build per-action-key normalization masks from the embodiment."""
    km = key_map or PART_TO_ACTION_KEY
    parts_by_key: dict[str, list[BodyPart]] = {}
    for part in embodiment.parts:
        if part.per_view:
            # raw multiview arrays keep their (H, V, ...) shape here; the
            # default all-True mask below broadcasts, a per-part mask can't
            continue
        key = km.get(part.name)
        if key is None:
            raise KeyError(f"no action key mapping for body part {part.name!r}")
        parts_by_key.setdefault(key, []).append(part)

    masks: dict[str, np.ndarray] = {}
    for key, value in action_dict.items():
        parts = parts_by_key.get(key)
        if not parts:
            masks[key] = np.ones(value.shape[-1], dtype=bool)
            continue
        mask = np.concatenate([np.asarray(part.action_norm_mask, dtype=bool) for part in parts])
        if mask.shape[-1] != value.shape[-1]:
            raise ValueError(f"{key!r}: norm mask len {mask.shape[-1]} != action dim {value.shape[-1]}")
        masks[key] = mask
    return masks


def build_embodiment_action(
    action_dict: dict[str, np.ndarray],
    embodiment: Embodiment,
    max_a: int,
    rng: np.random.Generator,
    mask_prob: float = 0.10,
    shuffle_slot: bool = True,
    key_map: dict[str, str] | None = None,
    valid_mask_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """End-to-end: extract actions, sample modes, shuffle order, build block.

    Args:
        action_dict: decoded sample's action sub-dict,
            e.g. {"joints": (H,7), "gripper": (H,1), "position": (H,3), ...}
        embodiment: this dataset's embodiment.
        max_a: global pad width.
        rng: numpy generator for mode sampling and shuffling.
        mask_prob: probability of masking each body part (default 0.25).
        shuffle_slot: whether to randomly permute body-part slot order.
        key_map: optional override for body-part-name → action-dict-key.
        valid_mask_dict: optional per-action-key per-DOF validity masks, shape
            (D_part,) bool. Keys align with action_dict keys. Missing keys =
            fully valid.

    Returns:
        {"act": {"base": (H, max_a), "id": (max_a,), "view": (max_a,)}, "mask": {"act": (max_a,)}}.
    """
    parts = list(embodiment.expanded)
    extracted = extract_part_actions(action_dict, embodiment, key_map)
    modes = sample_modes(len(parts), rng, mask_prob)

    # parts whose action key is absent in this dataset: zero-fill + force MASK
    present = [i for i, a in enumerate(extracted) if a is not None]
    assert present, f"no action keys found for any part of {embodiment.name!r}. got={list(action_dict)}"
    H = extracted[present[0]].shape[0]  # type: ignore[union-attr]
    actions: list[np.ndarray] = []
    for i, a in enumerate(extracted):
        if a is None:
            actions.append(np.zeros((H, parts[i].action_dim), dtype=np.float32))
            modes[i] = MASK
        else:
            actions.append(a)
    if all(m == MASK for m in modes):  # keep the sample_modes guarantee
        modes[present[rng.integers(len(present))]] = INCLUDE

    order = rng.permutation(len(parts)).tolist() if shuffle_slot else list(range(len(parts)))
    valid_masks: list[np.ndarray | None] | None = None
    if valid_mask_dict is not None:
        km = key_map or PART_TO_ACTION_KEY
        valid_masks = []
        for p in parts:
            vm = valid_mask_dict.get(km.get(p.name, ""), None)
            if vm is not None and p.view > 0:
                vm = np.asarray(vm)[p.view - 1]  # (V, D) -> this view's (D,)
            valid_masks.append(vm)
    return build_action_block(parts, actions, modes, order, max_a, valid_masks)


# ---------------------------------------------------------------------------
# Grain-compatible per-sample transform
# ---------------------------------------------------------------------------


def _encode_name(name: str, length: int = 32) -> np.ndarray:
    """Encode a string as a fixed-length uint8 array, zero-padded."""
    raw = np.frombuffer(name.encode("utf-8")[:length], dtype=np.uint8)
    out = np.zeros(length, dtype=np.uint8)
    out[: len(raw)] = raw
    return out


def note_bodypart(sample: dict, *, embodiment: Embodiment) -> dict:
    """Annotate sample mask with body-part and embodiment presence."""
    mask = sample.setdefault("mask", {})
    mask["bodypart"] = {k: np.array(1, dtype=np.bool_).reshape(-1) for k in sample["action"]}
    mask["embodiment"] = {embodiment.name: np.array(1, dtype=np.bool_).reshape(-1)}
    return sample


def decode_embody_name(arr: np.ndarray) -> str:
    """Decode a uint8 array (from act.embody) back to a string."""
    return arr.astype(np.uint8).tobytes().rstrip(b"\x00").decode("utf-8")


def kp3dc_valid(sample: dict, key: str = "kp3dc_robot") -> np.ndarray | None:
    """Per-view kp3dc validity (V, 3K) from per-keypoint + extrinsics masks.

    Conservative over the horizon: a keypoint/view is valid only if valid at
    every real (non-padded) chunk step. Padded steps (mask.horizon False) are
    ignored. Handles both per-keypoint (H, V, K) masks (kp3dc_robot) and
    per-view (H, V) masks (kp3dc_hand), broadcasting the latter over the
    keypoint axis of the action array.
    """
    m = sample.get("mask", {})
    kp = m.get("proprio", {}).get(key)  # (H, V, K) or (H, V) bool
    if kp is None:
        return None
    kp = np.asarray(kp, dtype=bool)
    if kp.ndim == 2:  # per-view validity only -> broadcast over keypoints
        n_kp = np.asarray(sample["action"][key]).shape[-2]
        kp = np.repeat(kp[..., None], n_kp, axis=-1)  # (H, V, K)
    w2c = m.get("state", {}).get("extr", {}).get("w2c")  # (H, V) bool
    w2c = np.ones(kp.shape[:2], dtype=bool) if w2c is None else np.asarray(w2c, dtype=bool)
    horizon = np.asarray(m.get("horizon", np.ones(kp.shape[0])), dtype=bool)  # (H,)

    kp = np.where(horizon[:, None, None], kp, True).all(axis=0)  # (V, K)
    w2c = np.where(horizon[:, None], w2c, True).all(axis=0)  # (V,)
    valid = kp & w2c[:, None]  # (V, K)
    return np.repeat(valid, 3, axis=-1)  # (V, 3K) — xyz per keypoint


def add_state_from_action_block(sample: dict, block: dict[str, dict[str, np.ndarray]]) -> dict:
    """Add state slots matching the action slot order.

    Shapes before batching:
        act.base:        (H, A)
        act.id/view:     (A,)
        mask.act:        (A,)
        state.base:      (A,) from act.base[0]
        state.id/view:   (A,)
        mask.state.base: (A,)
    """
    sample.setdefault("state", {}).update(
        {
            "base": np.asarray(block["act"]["base"][0], dtype=np.float32),
            "id": np.asarray(block["act"]["id"], dtype=np.int32),
            "view": np.asarray(block["act"]["view"], dtype=np.int32),
        }
    )
    sample.setdefault("mask", {}).setdefault("state", {})["base"] = np.asarray(block["mask"]["act"], dtype=bool)
    return sample


def embody_transform(
    sample: dict,
    *,
    embodiment: Embodiment,
    max_a: int,
    mask_prob: float = 0.10,
    shuffle_slot: bool = True,
) -> dict:
    """Grain .map() transform: adds act.base, act.id, act.view, act.embody, mask.act."""
    rng = np.random.default_rng()
    sample = note_bodypart(sample, embodiment=embodiment)
    # Consume per-part validity masks from restructure, if present. Pop because
    # mask.act is re-written below as the (max_a,) slot mask of the built block.
    valid_mask_dict = sample.get("mask", {}).pop("act", None)
    for key in ("kp3dc_robot", "kp3dc_hand"):
        kpv = kp3dc_valid(sample, key)
        if kpv is not None:
            valid_mask_dict = dict(valid_mask_dict or {})
            valid_mask_dict[key] = kpv  # (V, D); sliced per view part
    block = build_embodiment_action(
        sample["action"],
        embodiment,
        max_a,
        rng,
        mask_prob,
        shuffle_slot=shuffle_slot,
        valid_mask_dict=valid_mask_dict,
    )
    sample = add_state_from_action_block(sample, block)
    block["act"]["embody"] = _encode_name(embodiment.name)
    sample["act"] = block["act"]
    sample.setdefault("mask", {})["act"] = block["mask"]["act"]
    return sample
