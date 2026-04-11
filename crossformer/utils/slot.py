"""Slot utilities — convert flat action/id blocks into per-bodypart dicts."""

from __future__ import annotations

import sys

from jax import Array
import jax.numpy as jnp

from crossformer import embody
from crossformer.embody import BodyPart, Embodiment


def _all_bodyparts() -> tuple[BodyPart, ...]:
    mod = sys.modules[embody.__name__]
    seen: dict[str, BodyPart] = {}
    for v in vars(mod).values():
        if isinstance(v, BodyPart):
            seen.setdefault(v.name, v)
    return tuple(seen.values())


# Sort longest first so greedy matching prefers larger parts (e.g. cart_pose over cart_pos).
_BODYPARTS: tuple[BodyPart, ...] = tuple(sorted(_all_bodyparts(), key=lambda p: -p.action_dim))


def split_by_bodypart(
    action: Array,
    ids: Array,
    embodiments: tuple[Embodiment, ...],
) -> dict[str, Array]:
    """Split (..., A) action into {part_name: slice} given possible embodiments.

    Body parts may be shuffled to any slot offset and some parts may be MASK'd
    out. The candidate vocabulary is the union of parts across `embodiments`.
    For each part we build a one-hot gather `(n, A)` from `ids == dof_ids`,
    then einsum it against `action` to pull each DOF from whichever slot it
    landed in. Missing / MASK'd parts produce all-zero gathers, hence zero
    output. No offset scan.
    """
    A = ids.shape[-1]
    max_dim = max(e.action_dim for e in embodiments)
    if action.shape[-1] != A:
        raise ValueError(f"action {action.shape} and ids {ids.shape} last dim mismatch")
    if max_dim != A:
        raise ValueError(f"slot width {A} != max embodiment dim {max_dim}")

    parts: dict[str, BodyPart] = {}
    for e in embodiments:
        for p in e.parts:
            parts.setdefault(p.name, p)

    # Right-align ids with action's leading dims by inserting singleton axes.
    extra = action.ndim - ids.ndim
    if extra < 0:
        raise ValueError(f"action.ndim {action.ndim} < ids.ndim {ids.ndim}")
    ids_b = ids.reshape(*ids.shape[:-1], *(1,) * extra, ids.shape[-1])

    out: dict[str, Array] = {}
    for p in parts.values():
        expected = jnp.asarray(p.dof_ids, dtype=ids.dtype)[:, None]  # (n, 1)
        gather = (ids_b[..., None, :] == expected).astype(action.dtype)  # (..., n, A)
        out[p.name] = (action[..., None, :] * gather).sum(-1)
    return out
