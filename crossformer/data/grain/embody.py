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
) -> dict[str, np.ndarray]:
    """Build act.base, act.id, mask.act from body parts in random order.

    Args:
        parts: body parts for this embodiment.
        actions: per-part arrays, each (H, D_i) float.
        modes: per-part INCLUDE/MASK from sample_modes.
        order: permutation indices for body part ordering.
        max_a: max action dim. pad width

    Returns:
        dict with act.base (H, max_a) float32,
                  act.id  (max_a,)   int32,
                  mask.act (max_a,)  bool.
    """
    assert len(parts) == len(actions) == len(modes) == len(order)
    H = actions[0].shape[0]

    act_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []

    for idx in order:
        part, act, mode = parts[idx], actions[idx], modes[idx]
        D = part.action_dim
        assert act.shape == (H, D), f"{part.name}: expected ({H},{D}), got {act.shape}"

        if mode == MASK:
            act_chunks.append(np.zeros((H, D), dtype=np.float32))
            id_chunks.append(np.full(D, MASK_DOF, dtype=np.int32))
        else:  # INCLUDE
            act_chunks.append(act.astype(np.float32))
            id_chunks.append(np.array(part.dof_ids, dtype=np.int32))

    used = sum(p.action_dim for p in parts)
    pad_d = max_a - used

    act_base = np.concatenate(act_chunks, axis=-1)  # (H, used)
    act_id = np.concatenate(id_chunks)  # (used,)

    if pad_d > 0:
        act_base = np.pad(act_base, ((0, 0), (0, pad_d)))
        act_id = np.pad(act_id, (0, pad_d), constant_values=MASK_DOF)

    mask_act = act_id != MASK_DOF

    return {
        "act.base": act_base,
        "act.id": act_id,
        "mask.act": mask_act,
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
}


def extract_part_actions(
    action_dict: dict[str, np.ndarray],
    embodiment: Embodiment,
    key_map: dict[str, str] | None = None,
) -> list[np.ndarray]:
    """Pull per-body-part action arrays from a flat action dict.

    Args:
        action_dict: e.g. {"joints": (H,7), "gripper": (H,1), ...}
        embodiment: the dataset's embodiment.
        key_map: override for PART_TO_ACTION_KEY.

    Returns:
        list of (H, D_i) arrays, one per body part in embodiment.parts.
    """
    km = key_map or PART_TO_ACTION_KEY
    out = []
    for part in embodiment.parts:
        key = km.get(part.name)
        if key is None:
            raise KeyError(f"no action key mapping for body part {part.name!r}")
        if key not in action_dict:
            raise KeyError(f"action key {key!r} (for {part.name!r}) not in data. got={list(action_dict)}")
        out.append(action_dict[key])
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
    key_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """End-to-end: extract actions, sample modes, shuffle order, build block.

    Args:
        action_dict: decoded sample's action sub-dict,
            e.g. {"joints": (H,7), "gripper": (H,1), "position": (H,3), ...}
        embodiment: this dataset's embodiment.
        max_a: global pad width.
        rng: numpy generator for mode sampling and shuffling.
        mask_prob: probability of masking each body part (default 0.25).
        key_map: optional override for body-part-name → action-dict-key.

    Returns:
        {"act.base": (H, max_a), "act.id": (max_a,), "mask.act": (max_a,)}.
    """
    parts = list(embodiment.parts)
    actions = extract_part_actions(action_dict, embodiment, key_map)
    modes = sample_modes(len(parts), rng, mask_prob)
    order = rng.permutation(len(parts)).tolist()
    return build_action_block(parts, actions, modes, order, max_a)


# ---------------------------------------------------------------------------
# Grain-compatible per-sample transform
# ---------------------------------------------------------------------------


def _encode_name(name: str, length: int = 32) -> np.ndarray:
    """Encode a string as a fixed-length uint8 array, zero-padded."""
    raw = np.frombuffer(name.encode("utf-8")[:length], dtype=np.uint8)
    out = np.zeros(length, dtype=np.uint8)
    out[: len(raw)] = raw
    return out


def decode_embody_name(arr: np.ndarray) -> str:
    """Decode a uint8 array (from act.embody) back to a string."""
    return arr.astype(np.uint8).tobytes().rstrip(b"\x00").decode("utf-8")


def embody_transform(
    sample: dict,
    *,
    embodiment: Embodiment,
    max_a: int,
    mask_prob: float = 0.10,
) -> dict:
    """Grain .map() transform: adds act.base, act.id, act.embody, mask.act."""
    rng = np.random.default_rng()
    block = build_embodiment_action(
        sample["action"],
        embodiment,
        max_a,
        rng,
        mask_prob,
    )
    block["act.embody"] = _encode_name(embodiment.name)
    sample.update(block)
    return sample
