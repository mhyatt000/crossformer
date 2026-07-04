from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.xflow import XFlowHead

pytestmark = pytest.mark.nn

MAX_H, MAX_A = 3, 5


def _head(factor_attn: bool) -> XFlowHead:
    return XFlowHead(
        readout_key="obs",
        max_dofs=MAX_A,
        max_horizon=MAX_H,
        num_query_channels=16,
        num_heads=2,
        num_blocks=1,
        num_self_attend_layers=2,
        dropout_prob=0.0,
        flow_steps=2,
        factor_attn=factor_attn,
    )


def _outputs() -> dict[str, TokenGroup]:
    tokens = jnp.linspace(0.0, 1.0, 2 * 3 * 4 * 8, dtype=jnp.float32).reshape(2, 3, 4, 8)
    mask = jnp.ones((2, 3, 4), dtype=jnp.bool_)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


def _inputs() -> dict[str, jax.Array]:
    return {
        "time": jnp.full((2, 3, 1), 0.5, dtype=jnp.float32),
        "a_t": jnp.zeros((2, 3, MAX_H * MAX_A), dtype=jnp.float32),
        "dof_ids": jnp.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=jnp.int32),
        "chunk_steps": jnp.arange(MAX_H, dtype=jnp.float32)[None].repeat(2, axis=0),
    }


def test_factor_attn_forward_shape_finite() -> None:
    head = _head(factor_attn=True)
    outputs = _outputs()
    params = head.init(jax.random.PRNGKey(0), outputs, train=False)
    out = head.apply(params, outputs, train=False, **_inputs())
    assert out.shape == (2, 3, MAX_H * MAX_A)
    assert jnp.all(jnp.isfinite(out))


def test_factor_attn_param_tree() -> None:
    factored = _head(factor_attn=True).init(jax.random.PRNGKey(0), _outputs(), train=False)
    full = _head(factor_attn=False).init(jax.random.PRNGKey(0), _outputs(), train=False)
    fk = set(factored["params"]["decoder"])
    uk = set(full["params"]["decoder"])
    # factored path replaces each self-attn layer with an A-axis and an H-axis layer
    assert {"self_attend_a_0_0", "self_attend_h_0_0", "self_attend_a_0_1", "self_attend_h_0_1"} <= fk
    assert not any(k.startswith(("self_attend_a", "self_attend_h")) for k in uk)
