from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.data.grain.embody import build_action_block, INCLUDE, MASK
from crossformer.embody import HUMAN_SINGLE, SINGLE
from crossformer.utils.slot import split_by_bodypart

MAX_A = 14  # >= max(SINGLE.action_dim, HUMAN_SINGLE.action_dim)


def _make_part_actions(emb, h: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.standard_normal((h, p.action_dim)).astype(np.float32) for p in emb.parts]


def _block(emb, h, rng, modes=None, order=None):
    parts = list(emb.parts)
    actions = _make_part_actions(emb, h, rng)
    if modes is None:
        modes = [INCLUDE] * len(parts)
    if order is None:
        order = list(range(len(parts)))
    blk = build_action_block(parts, actions, modes, order, MAX_A)
    return blk, parts, actions, modes, order


def _expected(parts, actions, modes) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for p, a, m in zip(parts, actions, modes):
        out[p.name] = a if m == INCLUDE else np.zeros_like(a)
    return out


@pytest.mark.parametrize("embodiments", [(SINGLE,), (HUMAN_SINGLE,), (SINGLE, HUMAN_SINGLE)])
def test_canonical_order_all_present(embodiments):
    rng = np.random.default_rng(0)
    max_a = max(e.action_dim for e in embodiments)
    for emb in embodiments:
        parts = list(emb.parts)
        actions = _make_part_actions(emb, 4, rng)
        modes = [INCLUDE] * len(parts)
        blk = build_action_block(parts, actions, modes, list(range(len(parts))), max_a)
        out = split_by_bodypart(jnp.asarray(blk["act.base"]), jnp.asarray(blk["act.id"]), embodiments)
        exp = _expected(parts, actions, modes)
        for name, val in exp.items():
            np.testing.assert_allclose(np.asarray(out[name]), val, atol=1e-6)
        # parts not in this sample's embodiment must be zero
        present = {p.name for p in parts}
        for name, arr in out.items():
            if name not in present:
                assert np.all(np.asarray(arr) == 0)


@pytest.mark.parametrize("embodiments", [(SINGLE,), (SINGLE, HUMAN_SINGLE)])
def test_shuffled_order(embodiments):
    rng = np.random.default_rng(1)
    order = [3, 0, 2, 1]  # arbitrary permutation of SINGLE's 4 parts
    blk, parts, actions, _modes, _ = _block(SINGLE, h=5, rng=rng, order=order)
    out = split_by_bodypart(jnp.asarray(blk["act.base"]), jnp.asarray(blk["act.id"]), embodiments)
    for p, a in zip(parts, actions):
        np.testing.assert_allclose(np.asarray(out[p.name]), a, atol=1e-6)


@pytest.mark.parametrize("embodiments", [(SINGLE,), (SINGLE, HUMAN_SINGLE)])
def test_some_parts_masked(embodiments):
    rng = np.random.default_rng(2)
    modes = [INCLUDE, MASK, INCLUDE, MASK]  # mask gripper + cart_ori
    blk, parts, actions, _, _ = _block(SINGLE, h=3, rng=rng, modes=modes)
    out = split_by_bodypart(jnp.asarray(blk["act.base"]), jnp.asarray(blk["act.id"]), embodiments)
    for p, a, m in zip(parts, actions, modes):
        ref = a if m == INCLUDE else np.zeros_like(a)
        np.testing.assert_allclose(np.asarray(out[p.name]), ref, atol=1e-6)


def test_human_single_only_returns_cart_pos():
    rng = np.random.default_rng(3)
    parts = list(HUMAN_SINGLE.parts)
    actions = _make_part_actions(HUMAN_SINGLE, 2, rng)
    blk = build_action_block(parts, actions, [INCLUDE], [0], MAX_A)
    out = split_by_bodypart(
        jnp.asarray(blk["act.base"]),
        jnp.asarray(blk["act.id"]),
        (SINGLE, HUMAN_SINGLE),
    )
    np.testing.assert_allclose(np.asarray(out["cart_pos"]), actions[0], atol=1e-6)
    # SINGLE-only parts should be all zero for a human sample
    for name in ("arm_7dof", "gripper", "cart_ori"):
        assert np.all(np.asarray(out[name]) == 0)


def test_batched_leading_dims():
    rng = np.random.default_rng(4)
    blocks = [_block(SINGLE, h=3, rng=rng, order=list(rng.permutation(4)))[0] for _ in range(5)]
    act = jnp.stack([jnp.asarray(b["act.base"]) for b in blocks])  # (B, H, A)
    ids = jnp.asarray(blocks[0]["act.id"])  # same id layout broadcast? No — varies per sample.
    # For batching, ids must match per-sample. Stack them.
    ids = jnp.stack([jnp.asarray(b["act.id"]) for b in blocks])  # (B, A)
    out = split_by_bodypart(act, ids, (SINGLE,))
    assert out["arm_7dof"].shape == (5, 3, 7)
    assert out["gripper"].shape == (5, 3, 1)
    assert out["cart_pos"].shape == (5, 3, 3)
    assert out["cart_ori"].shape == (5, 3, 3)


@pytest.mark.parametrize(
    "act_lead,ids_lead",
    [
        ((2, 3, 4), (2, 3, 4)),  # BWHA, BWHA
        ((2, 3, 4), (2,)),  # BWHA, BA (per-sample ids)
        ((4,), (4,)),  # HA, HA
        ((2, 3, 4), ()),  # BWHA, A
        ((5, 2, 3, 4), (5, 2)),  # TBWHA, TBA
        ((5, 2, 3, 4), ()),  # TBWHA, A
    ],
)
def test_leading_dim_combinations(act_lead, ids_lead):
    rng = np.random.default_rng(7)
    parts = list(SINGLE.parts)
    A = SINGLE.action_dim
    order = [2, 0, 3, 1]

    # Single canonical sample: H=1, shape (1, A) and (A,)
    actions = _make_part_actions(SINGLE, 1, rng)
    blk = build_action_block(parts, actions, [INCLUDE] * len(parts), order, A)
    base_act = blk["act.base"]  # (1, A)
    base_ids = blk["act.id"]  # (A,)

    # Broadcast to requested leading shapes
    act = np.broadcast_to(base_act.reshape((1,) * (len(act_lead) - 1) + (1, A)), (*act_lead, A)).copy()
    ids = np.broadcast_to(base_ids.reshape((1,) * len(ids_lead) + (A,)), (*ids_lead, A)).copy()

    out = split_by_bodypart(jnp.asarray(act), jnp.asarray(ids), (SINGLE,))

    canon = split_by_bodypart(jnp.asarray(base_act), jnp.asarray(base_ids), (SINGLE,))
    for p in parts:
        expected = np.broadcast_to(np.asarray(canon[p.name]), (*act_lead, p.action_dim))
        np.testing.assert_allclose(np.asarray(out[p.name]), expected, atol=1e-6)


def test_jit_with_static_embodiments():
    import jax

    rng = np.random.default_rng(5)
    blk, parts, actions, _, _ = _block(SINGLE, h=4, rng=rng, order=[2, 0, 3, 1])
    fn = jax.jit(split_by_bodypart, static_argnames=("embodiments",))
    out = fn(jnp.asarray(blk["act.base"]), jnp.asarray(blk["act.id"]), (SINGLE, HUMAN_SINGLE))
    for p, a in zip(parts, actions):
        np.testing.assert_allclose(np.asarray(out[p.name]), a, atol=1e-6)
