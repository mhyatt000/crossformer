"""Embodiment-aware action construction for grain pipeline.

Per-sample transform that builds a padded action block (`act.base`),
DOF identity vector (`act.id`), and boolean mask (`mask.act`) by
randomly sampling which body parts to include, mask, or anonymize.

Not wired into the grain pipeline yet — call `build_embodiment_action`
directly for testing.
"""

from __future__ import annotations

import numpy as np

from crossformer.embody import BodyPart, Embodiment

# ---------------------------------------------------------------------------
# DOF ID sentinels (in act.id)
# ---------------------------------------------------------------------------

MASK_DOF = -1  # body part excluded — zeros in act.base, masked in loss
ANY_DOF = 0  # body part included but identity-erased

# ---------------------------------------------------------------------------
# Sampling modes per body part
# ---------------------------------------------------------------------------

INCLUDE = 0  # real DOF vocab IDs
MASK = 1  # excluded
ANY = 2  # included, identity-erased


def sample_modes(
    n_parts: int,
    rng: np.random.Generator,
    probs: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> list[int]:
    """Sample INCLUDE/MASK/ANY per body part, at least one non-masked."""
    modes = rng.choice([INCLUDE, MASK, ANY], size=n_parts, p=probs).tolist()
    if all(m == MASK for m in modes):
        modes[rng.integers(n_parts)] = INCLUDE
    return modes


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def compute_max_action_dim(embodiments: list[Embodiment]) -> int:
    """Max total DOFs across a set of embodiments."""
    return max(e.action_dim for e in embodiments)


def build_action_block(
    parts: list[BodyPart],
    actions: list[np.ndarray],
    modes: list[int],
    max_a: int,
) -> dict[str, np.ndarray]:
    """Build act.base, act.id, mask.act from body parts and sampled modes.

    Args:
        parts: body parts for this embodiment.
        actions: per-part arrays, each (H, D_i) float.
        modes: per-part INCLUDE/MASK/ANY from sample_modes.
        max_a: pad width (from compute_max_action_dim).

    Returns:
        dict with act.base (H, max_a) float32,
                  act.id  (max_a,)   int32,
                  mask.act (max_a,)  bool.
    """
    assert len(parts) == len(actions) == len(modes)
    H = actions[0].shape[0]

    act_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []

    for part, act, mode in zip(parts, actions, modes):
        D = part.action_dim
        assert act.shape == (H, D), f"{part.name}: expected ({H},{D}), got {act.shape}"

        if mode == MASK:
            act_chunks.append(np.zeros((H, D), dtype=np.float32))
            id_chunks.append(np.full(D, MASK_DOF, dtype=np.int32))
        elif mode == ANY:
            act_chunks.append(act.astype(np.float32))
            id_chunks.append(np.full(D, ANY_DOF, dtype=np.int32))
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

    mask_act = act_id >= 0

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


def build_embodiment_action(
    action_dict: dict[str, np.ndarray],
    embodiment: Embodiment,
    max_a: int,
    rng: np.random.Generator,
    probs: tuple[float, float, float] = (0.50, 0.25, 0.25),
    key_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """End-to-end: extract actions, sample modes, build padded block.

    Args:
        action_dict: decoded sample's action sub-dict,
            e.g. {"joints": (H,7), "gripper": (H,1), "position": (H,3), ...}
        embodiment: this dataset's embodiment.
        max_a: global pad width.
        rng: numpy generator for mode sampling.
        probs: (include, mask, any) probabilities.
        key_map: optional override for body-part-name → action-dict-key.

    Returns:
        {"act.base": (H, max_a), "act.id": (max_a,), "mask.act": (max_a,)}.
    """
    parts = list(embodiment.parts)
    actions = extract_part_actions(action_dict, embodiment, key_map)
    modes = sample_modes(len(parts), rng, probs)
    return build_action_block(parts, actions, modes, max_a)
