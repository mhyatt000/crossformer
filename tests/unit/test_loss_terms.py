from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.embody import ARM_7DOF, DOF, GRIPPER, KP3DC, SINGLE
from crossformer.model.components.heads.loss_terms import (
    denormalize_by_dof,
    dof_stats_table,
    fk_consistency_loss,
    FKConsistencyTerm,
    load_loss_terms,
    LossTerm,
    smoothness_loss,
    SmoothnessTerm,
    x1_estimate,
)

pytestmark = pytest.mark.nn

_LOSS_YAML = Path(__file__).parents[2] / "config" / "loss.yaml"


def _fake_fk(joints: jax.Array) -> jax.Array:
    """Deterministic fake FK: 14 keypoints from a linear map of joints."""
    w = jnp.arange(7 * 14 * 3, dtype=jnp.float32).reshape(7, 14 * 3) / 100.0
    return (joints @ w).reshape(*joints.shape[:-1], 14, 3)


# ---------------------------------------------------------------------------
# x1_estimate
# ---------------------------------------------------------------------------


def test_x1_estimate_recovers_x1_at_any_tau() -> None:
    rng = np.random.default_rng(0)
    x1 = jnp.asarray(rng.standard_normal((2, 4)), dtype=jnp.float32)
    x0 = jnp.asarray(rng.standard_normal((2, 4)), dtype=jnp.float32)
    for tau in (0.0, 0.3, 0.99):
        a_t = tau * x1 + (1.0 - tau) * x0
        v = x1 - x0  # exact velocity
        np.testing.assert_allclose(np.asarray(x1_estimate(a_t, v, tau)), np.asarray(x1), atol=1e-5)


def test_tau_scale() -> None:
    term = LossTerm(name="t", fn=lambda: (jnp.zeros(()), {}), dofs=(), tau_power=2.0)
    tau = jnp.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(np.asarray(term.tau_scale(tau)), [0.0, 0.25, 1.0])
    ungated = LossTerm(name="u", fn=lambda: (jnp.zeros(()), {}), dofs=())
    np.testing.assert_allclose(np.asarray(ungated.tau_scale(tau)), [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# dof stats table / denormalization
# ---------------------------------------------------------------------------


def _fake_stats() -> dict[str, SimpleNamespace]:
    return {
        "joints": SimpleNamespace(mean=np.arange(7.0), std=np.full(7, 2.0)),
        "gripper": SimpleNamespace(mean=np.array([9.0]), std=np.array([9.0])),
        "position": SimpleNamespace(mean=np.array([1.0, 2.0, 3.0]), std=np.array([0.5, 0.5, 0.5])),
        "orientation": SimpleNamespace(mean=np.zeros(3), std=np.ones(3)),
        "kp3dc_robot": SimpleNamespace(
            mean=np.arange(14 * 3, dtype=np.float64).reshape(14, 3),
            std=np.full((14, 3), 3.0),
        ),
    }


def test_dof_stats_table_and_denorm() -> None:
    table = dof_stats_table(_fake_stats(), SINGLE)

    # arm dofs carry their stats
    np.testing.assert_allclose(table[list(ARM_7DOF.dof_ids), 0], np.arange(7.0))
    np.testing.assert_allclose(table[list(ARM_7DOF.dof_ids), 1], 2.0)
    # gripper has norm_mask=False -> identity even though stats exist
    gid = GRIPPER.dof_ids[0]
    np.testing.assert_allclose(table[gid], [0.0, 1.0])
    # kp3dc flattened (14, 3) row-major matches KP_CHAIN xyz order
    np.testing.assert_allclose(table[list(KP3DC.dof_ids), 0], np.arange(42.0))
    np.testing.assert_allclose(table[list(KP3DC.dof_ids), 1], 3.0)
    # MASK id stays identity
    np.testing.assert_allclose(table[DOF["MASK"]], [0.0, 1.0])

    # denormalize a slot vector: joints j0 and kp first coord
    dof_ids = jnp.array([ARM_7DOF.dof_ids[0], KP3DC.dof_ids[0]], dtype=jnp.int32)
    x = jnp.array([1.0, 1.0])
    out = np.asarray(denormalize_by_dof(x, dof_ids, table))
    np.testing.assert_allclose(out, [0.0 + 2.0, 0.0 + 3.0])


# ---------------------------------------------------------------------------
# smoothness
# ---------------------------------------------------------------------------


def test_smoothness_zero_on_linear_trajectory() -> None:
    h = jnp.arange(6, dtype=jnp.float32)[:, None]  # (H, 1)
    x = jnp.tile(h, (1, 4))[None]  # (1, H, A) linear ramp -> zero accel
    mask = jnp.ones(x.shape, dtype=bool)
    loss, _ = smoothness_loss(x, mask, order=2)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


def test_smoothness_positive_and_mask_respected() -> None:
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((1, 6, 4)), dtype=jnp.float32)
    mask = jnp.ones(x.shape, dtype=bool)
    loss, _ = smoothness_loss(x, mask, order=2)
    assert float(loss) > 0.0

    # corrupting a fully-masked step must not change the loss
    mask2 = mask.at[:, 3].set(False)
    loss_a, _ = smoothness_loss(x, mask2, order=2)
    x_corrupt = x.at[:, 3].set(1e6)
    loss_b, _ = smoothness_loss(x_corrupt, mask2, order=2)
    np.testing.assert_allclose(float(loss_a), float(loss_b), rtol=1e-5)


# ---------------------------------------------------------------------------
# fk consistency
# ---------------------------------------------------------------------------


def test_fk_consistency_zero_when_consistent() -> None:
    rng = np.random.default_rng(2)
    joints = jnp.asarray(rng.standard_normal((2, 7)), dtype=jnp.float32)
    kp3dw = _fake_fk(joints)  # (2, 14, 3)

    # two views: identity extrinsics and a pure translation
    w2c = jnp.tile(jnp.eye(4, dtype=jnp.float32)[None, None], (2, 2, 1, 1))
    w2c = w2c.at[:, 1, :3, 3].set(jnp.array([1.0, -2.0, 3.0]))
    kp3dc = jnp.stack([kp3dw, kp3dw + jnp.array([1.0, -2.0, 3.0])], axis=1)  # (2, V, 14, 3)

    mask = jnp.ones((2, 2, 14), dtype=bool)
    loss, _ = fk_consistency_loss(joints, kp3dc, w2c, mask, _fake_fk)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-9)

    # perturb one view -> positive; mask that view out -> zero again
    kp_bad = kp3dc.at[:, 1].add(0.1)
    loss_bad, _ = fk_consistency_loss(joints, kp_bad, w2c, mask, _fake_fk)
    assert float(loss_bad) > 0.0
    mask_v0 = mask.at[:, 1].set(False)
    loss_masked, _ = fk_consistency_loss(joints, kp_bad, w2c, mask_v0, _fake_fk)
    np.testing.assert_allclose(float(loss_masked), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# yaml registry
# ---------------------------------------------------------------------------


def test_load_loss_terms_from_yaml() -> None:
    terms = load_loss_terms(_LOSS_YAML, fk_fn=_fake_fk)
    assert set(terms) == {"smoothness", "fk_consistency", "reprojection"}
    for term in terms.values():
        assert isinstance(term, LossTerm)
        assert term.tau_power > 0.0  # every aux term must be tau-gated
    assert isinstance(terms["smoothness"], SmoothnessTerm)
    assert isinstance(terms["fk_consistency"], FKConsistencyTerm)
    assert terms["fk_consistency"].fk_fn is _fake_fk
    assert terms["smoothness"].weight == pytest.approx(0.05)


def test_load_loss_terms_requires_fk() -> None:
    with pytest.raises((ValueError, TypeError)):
        load_loss_terms(_LOSS_YAML, fk_fn=None)
