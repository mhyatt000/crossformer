from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.pio import PerceiverIOHead

pytestmark = pytest.mark.nn


@pytest.fixture
def transformer_outputs():
    tokens = jnp.linspace(0.0, 1.0, 2 * 3 * 4 * 8, dtype=jnp.float32).reshape(2, 3, 4, 8)
    mask = jnp.ones((2, 3, 4), dtype=jnp.bool_)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


@pytest.fixture(params=[0, 4], ids=["single-encode", "act-fuse"])
def head(request):
    return PerceiverIOHead(
        readout_key="obs",
        max_dofs=5,
        max_horizon=2,
        num_query_channels=16,
        num_heads=2,
        num_self_attend_layers=1,
        num_latents=6,
        num_act_latents=request.param,
        num_fuse_layers=1,
        dropout_prob=0.0,
        flow_steps=3,
        max_action=0.25,
    )


@pytest.fixture
def head_inputs():
    return {
        "time": jnp.full((2, 3, 1), 0.5, dtype=jnp.float32),
        "a_t": jnp.arange(2 * 3 * 2 * 5, dtype=jnp.float32).reshape(2, 3, 2, 5) / 100.0,
        "dof_ids": jnp.array(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 0, 0, 0],
            ],
            dtype=jnp.int32,
        ),
        "chunk_steps": jnp.array(
            [
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=jnp.float32,
        ),
    }


def test_pio_forward_accepts_rank4_and_flat_actions(transformer_outputs, head, head_inputs):
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)

    out_4d = head.apply(params, transformer_outputs, train=False, **head_inputs)
    out_flat = head.apply(
        params,
        transformer_outputs,
        time=head_inputs["time"],
        a_t=head_inputs["a_t"].reshape(2, 3, -1),
        dof_ids=head_inputs["dof_ids"],
        chunk_steps=head_inputs["chunk_steps"],
        train=False,
    )

    assert out_4d.shape == (2, 3, 10)
    assert jnp.allclose(out_4d, out_flat)


def test_pio_forward_requires_flow_inputs(transformer_outputs, head):
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)

    with pytest.raises(ValueError, match="Must provide time, a_t, dof_ids, chunk_steps"):
        head.apply(params, transformer_outputs, train=False)


def test_pio_no_dense_decoder_params(transformer_outputs, head):
    """The inherited dense decoder must not materialize params (never called)."""
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)
    assert "decoder" not in params["params"], sorted(params["params"])


def test_pio_loss_ignores_padded_queries(transformer_outputs, head):
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=True)
    dof_ids = jnp.array([[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]], dtype=jnp.int32)
    chunk_steps = jnp.array([[0.0, 1.0], [0.0, -1.0]], dtype=jnp.float32)

    actions = jnp.zeros((2, 3, 2, 5), dtype=jnp.float32)
    changed = actions.at[:, :, :, 2:].set(99.0)
    changed = changed.at[1, :, 1, :].set(-99.0)

    rng = jax.random.PRNGKey(42)
    loss_a, metrics_a = head.apply(
        params,
        transformer_outputs,
        actions,
        dof_ids,
        chunk_steps,
        train=True,
        method=head.loss,
        rngs={"dropout": rng},
    )
    loss_b, metrics_b = head.apply(
        params,
        transformer_outputs,
        changed,
        dof_ids,
        chunk_steps,
        train=True,
        method=head.loss,
        rngs={"dropout": rng},
    )

    assert jnp.allclose(loss_a, loss_b)
    assert jnp.allclose(metrics_a["mse"], metrics_b["mse"])


def test_pio_remat_is_equivalent(transformer_outputs, head, head_inputs):
    """remat=True must share the param tree and produce identical outputs/grads."""
    head_r = head.clone(remat=True)
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)
    params_r = head_r.init(jax.random.PRNGKey(0), transformer_outputs, train=False)
    assert jax.tree.structure(params) == jax.tree.structure(params_r)

    out = head.apply(params, transformer_outputs, train=False, **head_inputs)
    out_r = head_r.apply(params, transformer_outputs, train=False, **head_inputs)
    assert jnp.allclose(out, out_r, atol=1e-5)

    def loss_of(h):
        def f(p):
            return h.apply(p, transformer_outputs, train=False, **head_inputs).sum()

        return jax.grad(f)(params)

    g, g_r = loss_of(head), loss_of(head_r)
    flat, flat_r = jax.tree.leaves(g), jax.tree.leaves(g_r)
    assert all(jnp.allclose(a, b, atol=1e-5) for a, b in zip(flat, flat_r))


def test_pio_predict_action_respects_sample_shape_and_clipping(transformer_outputs, head, head_inputs):
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)

    pred = head.apply(
        params,
        transformer_outputs,
        rng=jax.random.PRNGKey(7),
        dof_ids=head_inputs["dof_ids"],
        chunk_steps=head_inputs["chunk_steps"],
        sample_shape=(2,),
        method=head.predict_action,
    )

    assert pred.shape == (2, 2, 3, 2, 5)
    assert jnp.all(pred <= head.max_action + 1e-6)
    assert jnp.all(pred >= -head.max_action - 1e-6)
