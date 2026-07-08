from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.data.grain.embody import build_action_block, INCLUDE, MASK
from crossformer.embody import HUMAN_SINGLE, SINGLE
from crossformer.utils.slot import split_by_bodypart

MAX_A = max(SINGLE.action_dim, HUMAN_SINGLE.action_dim)
N_PARTS = len(SINGLE.expanded)  # 4 innate + MAX_VIEWS kp3dc copies


def _part_key(p) -> str:
    return p.name if p.view == 0 else f"{p.name}_v{p.view}"


def _make_part_actions(emb, h: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.standard_normal((h, p.action_dim)).astype(np.float32) for p in emb.expanded]


def _block(emb, h, rng, modes=None, order=None, max_a=MAX_A):
    parts = list(emb.expanded)
    actions = _make_part_actions(emb, h, rng)
    if modes is None:
        modes = [INCLUDE] * len(parts)
    if order is None:
        order = list(range(len(parts)))
    blk = build_action_block(parts, actions, modes, order, max_a)
    return blk, parts, actions, modes, order


def _split(blk, embodiments):
    return split_by_bodypart(
        jnp.asarray(blk["act"]["base"]),
        jnp.asarray(blk["act"]["id"]),
        embodiments,
        views=jnp.asarray(blk["act"]["view"]),
    )


def _expected(parts, actions, modes) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for p, a, m in zip(parts, actions, modes):
        out[_part_key(p)] = a if m == INCLUDE else np.zeros_like(a)
    return out


@pytest.mark.parametrize("embodiments", [(SINGLE,), (HUMAN_SINGLE,), (SINGLE, HUMAN_SINGLE)])
def test_canonical_order_all_present(embodiments):
    rng = np.random.default_rng(0)
    max_a = max(e.action_dim for e in embodiments)
    for emb in embodiments:
        parts = list(emb.expanded)
        actions = _make_part_actions(emb, 4, rng)
        modes = [INCLUDE] * len(parts)
        blk = build_action_block(parts, actions, modes, list(range(len(parts))), max_a)
        out = split_by_bodypart(
            jnp.asarray(blk["act"]["base"]),
            jnp.asarray(blk["act"]["id"]),
            embodiments,
            views=jnp.asarray(blk["act"]["view"]),
        )
        exp = _expected(parts, actions, modes)
        for name, val in exp.items():
            np.testing.assert_allclose(np.asarray(out[name]), val, atol=1e-6)
        # parts not in this sample's embodiment must be zero
        present = {_part_key(p) for p in parts}
        for name, arr in out.items():
            if name not in present:
                assert np.all(np.asarray(arr) == 0)


@pytest.mark.parametrize("embodiments", [(SINGLE,), (SINGLE, HUMAN_SINGLE)])
def test_shuffled_order(embodiments):
    rng = np.random.default_rng(1)
    order = list(rng.permutation(N_PARTS))
    max_a = max(e.action_dim for e in embodiments)
    blk, parts, actions, _modes, _ = _block(SINGLE, h=5, rng=rng, order=order, max_a=max_a)
    out = _split(blk, embodiments)
    for p, a in zip(parts, actions):
        np.testing.assert_allclose(np.asarray(out[_part_key(p)]), a, atol=1e-6)


@pytest.mark.parametrize("embodiments", [(SINGLE,), (SINGLE, HUMAN_SINGLE)])
def test_some_parts_masked(embodiments):
    rng = np.random.default_rng(2)
    # mask gripper + cart_ori + the second kp3dc view
    masked = {"gripper", "cart_ori", "kp3dc_v2"}
    modes = [MASK if _part_key(p) in masked else INCLUDE for p in SINGLE.expanded]
    max_a = max(e.action_dim for e in embodiments)
    blk, parts, actions, _, _ = _block(SINGLE, h=3, rng=rng, modes=modes, max_a=max_a)
    out = _split(blk, embodiments)
    for p, a, m in zip(parts, actions, modes):
        ref = a if m == INCLUDE else np.zeros_like(a)
        np.testing.assert_allclose(np.asarray(out[_part_key(p)]), ref, atol=1e-6)


def test_human_single_parts():
    rng = np.random.default_rng(3)
    parts = list(HUMAN_SINGLE.expanded)  # cart_pos + MAX_VIEWS kp3dc_hand copies
    actions = _make_part_actions(HUMAN_SINGLE, 2, rng)
    modes = [INCLUDE] * len(parts)
    blk = build_action_block(parts, actions, modes, list(range(len(parts))), MAX_A)
    out = split_by_bodypart(
        jnp.asarray(blk["act"]["base"]),
        jnp.asarray(blk["act"]["id"]),
        (SINGLE, HUMAN_SINGLE),
        views=jnp.asarray(blk["act"]["view"]),
    )
    for p, a in zip(parts, actions):
        np.testing.assert_allclose(np.asarray(out[_part_key(p)]), a, atol=1e-6)
    # SINGLE-only parts should be all zero for a human sample
    for name in ("arm_7dof", "gripper", "cart_ori", "kp3dc_v1", "kp3dc_v2", "kp3dc_v3"):
        assert np.all(np.asarray(out[name]) == 0)


def test_batched_leading_dims():
    rng = np.random.default_rng(4)
    blocks = [_block(SINGLE, h=3, rng=rng, order=list(rng.permutation(N_PARTS)))[0] for _ in range(5)]
    act = jnp.stack([jnp.asarray(b["act"]["base"]) for b in blocks])  # (B, H, A)
    ids = jnp.stack([jnp.asarray(b["act"]["id"]) for b in blocks])  # (B, A)
    views = jnp.stack([jnp.asarray(b["act"]["view"]) for b in blocks])  # (B, A)
    out = split_by_bodypart(act, ids, (SINGLE, HUMAN_SINGLE), views=views)
    assert out["arm_7dof"].shape == (5, 3, 7)
    assert out["gripper"].shape == (5, 3, 1)
    assert out["cart_pos"].shape == (5, 3, 3)
    assert out["cart_ori"].shape == (5, 3, 3)
    assert out["kp3dc_v1"].shape == (5, 3, 42)


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
    parts = list(SINGLE.expanded)
    A = SINGLE.action_dim
    order = list(np.random.default_rng(8).permutation(N_PARTS))

    # Single canonical sample: H=1, shape (1, A) and (A,)
    actions = _make_part_actions(SINGLE, 1, rng)
    blk = build_action_block(parts, actions, [INCLUDE] * len(parts), order, A)
    base_act = blk["act"]["base"]  # (1, A)
    base_ids = blk["act"]["id"]  # (A,)
    base_views = blk["act"]["view"]  # (A,)

    # Broadcast to requested leading shapes
    act = np.broadcast_to(base_act.reshape((1,) * (len(act_lead) - 1) + (1, A)), (*act_lead, A)).copy()
    ids = np.broadcast_to(base_ids.reshape((1,) * len(ids_lead) + (A,)), (*ids_lead, A)).copy()
    views = np.broadcast_to(base_views.reshape((1,) * len(ids_lead) + (A,)), (*ids_lead, A)).copy()

    out = split_by_bodypart(jnp.asarray(act), jnp.asarray(ids), (SINGLE,), views=jnp.asarray(views))

    canon = split_by_bodypart(jnp.asarray(base_act), jnp.asarray(base_ids), (SINGLE,), views=jnp.asarray(base_views))
    for p in parts:
        expected = np.broadcast_to(np.asarray(canon[_part_key(p)]), (*act_lead, p.action_dim))
        np.testing.assert_allclose(np.asarray(out[_part_key(p)]), expected, atol=1e-6)


def test_jit_with_static_embodiments():
    import jax

    rng = np.random.default_rng(5)
    blk, parts, actions, _, _ = _block(SINGLE, h=4, rng=rng, order=list(rng.permutation(N_PARTS)))
    fn = jax.jit(split_by_bodypart, static_argnames=("embodiments",))
    out = fn(
        jnp.asarray(blk["act"]["base"]),
        jnp.asarray(blk["act"]["id"]),
        (SINGLE, HUMAN_SINGLE),
        views=jnp.asarray(blk["act"]["view"]),
    )
    for p, a in zip(parts, actions):
        np.testing.assert_allclose(np.asarray(out[_part_key(p)]), a, atol=1e-6)
