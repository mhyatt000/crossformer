from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.dof import FactoredQueryEncoding
from crossformer.model.components.heads.io.attention import make_view_bias_weight
from crossformer.model.components.heads.xflow import XFlowHead

pytestmark = pytest.mark.nn


# ---------------------------------------------------------------------------
# make_view_bias_weight
# ---------------------------------------------------------------------------


def test_view_bias_weight_shape_and_values() -> None:
    q_view = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    kv_view = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

    w = make_view_bias_weight(q_view, kv_view)
    assert w.shape == (1, 3, 4)

    # NO_VIEW query (view 0) is neutral against every kv token.
    assert jnp.all(w[0, 0, :] == 1.0)

    # NO_VIEW kv token (view 0) is neutral for every query.
    assert jnp.all(w[:, :, 0] == 1.0)

    # Paired same view boosts, paired different view suppresses.
    assert w[0, 1, 1] == pytest.approx(1.2)  # q=1, kv=1
    assert w[0, 1, 2] == pytest.approx(0.8)  # q=1, kv=2
    assert w[0, 1, 3] == pytest.approx(0.8)  # q=1, kv=3
    assert w[0, 2, 2] == pytest.approx(1.2)  # q=2, kv=2


def test_view_bias_weight_custom_scales() -> None:
    q_view = jnp.array([[1, 2]], dtype=jnp.int32)
    kv_view = jnp.array([[1, 2]], dtype=jnp.int32)

    w = make_view_bias_weight(q_view, kv_view, same_view=2.0, other_view=0.5, neutral=1.0)
    assert w[0, 0, 0] == pytest.approx(2.0)  # same
    assert w[0, 0, 1] == pytest.approx(0.5)  # other
    assert w[0, 1, 0] == pytest.approx(0.5)  # other
    assert w[0, 1, 1] == pytest.approx(2.0)  # same


# ---------------------------------------------------------------------------
# FactoredQueryEncoding view embedding
# ---------------------------------------------------------------------------


def test_factored_query_encoding_view_none_equals_zeros() -> None:
    enc = FactoredQueryEncoding(num_channels=16)
    chunk_steps = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    dof_ids = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    slot_pos = jnp.array([[0.0, 1.0, 2.0]], dtype=jnp.float32)

    params = enc.init(jax.random.PRNGKey(0), chunk_steps, dof_ids, slot_pos)

    out_none = enc.apply(params, chunk_steps, dof_ids, slot_pos)
    out_zeros = enc.apply(params, chunk_steps, dof_ids, slot_pos, jnp.zeros_like(dof_ids))
    out_view = enc.apply(params, chunk_steps, dof_ids, slot_pos, jnp.array([[1, 2, 0]], dtype=jnp.int32))

    assert jnp.allclose(out_none, out_zeros)
    assert not jnp.allclose(out_none, out_view)


# ---------------------------------------------------------------------------
# XFlowHead forward with view ids
# ---------------------------------------------------------------------------


@pytest.fixture
def head() -> XFlowHead:
    return XFlowHead(
        readout_key="obs",
        max_dofs=5,
        max_horizon=2,
        num_query_channels=16,
        num_heads=2,
        num_blocks=1,
        num_self_attend_layers=1,
        dropout_prob=0.0,
        flow_steps=3,
        max_action=0.25,
    )


@pytest.fixture
def head_inputs() -> dict[str, jax.Array]:
    return {
        "time": jnp.full((2, 3, 1), 0.5, dtype=jnp.float32),
        "a_t": jnp.arange(2 * 3 * 2 * 5, dtype=jnp.float32).reshape(2, 3, 2, 5) / 100.0,
        "dof_ids": jnp.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=jnp.int32),
        "chunk_steps": jnp.array([[0.0, 1.0], [0.0, -1.0]], dtype=jnp.float32),
    }


def _outputs(view: jax.Array | None = None) -> dict[str, TokenGroup]:
    tokens = jnp.linspace(0.0, 1.0, 2 * 3 * 4 * 8, dtype=jnp.float32).reshape(2, 3, 4, 8)
    mask = jnp.ones((2, 3, 4), dtype=jnp.bool_)
    return {"obs": TokenGroup(tokens=tokens, mask=mask, view=view)}


def test_xflow_view_ids_zero_matches_none(head: XFlowHead, head_inputs: dict[str, jax.Array]) -> None:
    outputs = _outputs()
    params = head.init(jax.random.PRNGKey(0), outputs, train=False)

    view_ids = jnp.zeros_like(head_inputs["dof_ids"])
    out_none = head.apply(params, outputs, train=False, view_ids=None, **head_inputs)
    out_zeros = head.apply(params, outputs, train=False, view_ids=view_ids, **head_inputs)

    assert out_none.shape == (2, 3, 10)
    assert jnp.all(jnp.isfinite(out_none))
    assert jnp.allclose(out_none, out_zeros)


def test_xflow_view_ids_change_output(head: XFlowHead, head_inputs: dict[str, jax.Array]) -> None:
    # Token group carries per-token view ids so the bias is non-neutral.
    view = jnp.array([[1, 2, 3, 0]] * 3, dtype=jnp.int32)[None].repeat(2, axis=0)
    outputs = _outputs(view=view)
    params = head.init(jax.random.PRNGKey(0), outputs, train=False)

    view_ids = jnp.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=jnp.int32)
    out_view = head.apply(params, outputs, train=False, view_ids=view_ids, **head_inputs)
    out_zero = head.apply(params, outputs, train=False, view_ids=None, **head_inputs)

    assert out_view.shape == (2, 3, 10)
    assert jnp.all(jnp.isfinite(out_view))
    assert not jnp.allclose(out_view, out_zero)
