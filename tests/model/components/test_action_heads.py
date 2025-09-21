import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.action_heads import (
    ContinuousActionHead,
    DiffusionActionHead,
    L1ActionHead,
    MSEActionHead,
    continuous_loss,
)
from crossformer.model.components.base import TokenGroup


@pytest.fixture
def transformer_outputs():
    tokens = jnp.linspace(0, 1, 2 * 3 * 4 * 8).reshape(2, 3, 4, 8)
    mask = jnp.ones((2, 3, 4))
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


def test_continuous_loss_mse_and_l1():
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.array([[2.0, 2.0], [1.0, 6.0]])
    mask = jnp.array([[1.0, 0.0], [1.0, 1.0]])

    mse_loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
    assert mse_loss == pytest.approx(3.0, rel=1e-5)
    assert set(metrics.keys()) == {"loss", "mse", "lsign"}

    l1_loss, _ = continuous_loss(pred, target, mask, loss_type="l1")
    assert l1_loss < mse_loss


@pytest.mark.parametrize("pool_strategy", ["mean", "pass"])
def test_continuous_action_head_forward_and_loss(transformer_outputs, pool_strategy):
    if pool_strategy == "pass":
        tokens = transformer_outputs["obs"].tokens.reshape(2, 3, 2, 16)
        mask = jnp.ones((2, 3, 2))
        transformer_outputs = {"obs": TokenGroup(tokens=tokens, mask=mask)}

    head = ContinuousActionHead(
        readout_key="obs",
        pool_strategy=pool_strategy,
        action_horizon=2,
        action_dim=3,
    )

    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=True)
    preds = head.apply(params, transformer_outputs, train=True)

    if pool_strategy == "pass":
        expected_last_dim = head.num_preds or head.action_horizon * head.action_dim
        assert preds.shape == (2, 3, head.action_horizon, expected_last_dim)
    else:
        assert preds.shape == (2, 3, head.action_horizon, head.action_dim)
    assert jnp.all(jnp.abs(preds) <= head.max_action + 1e-5)

    actions = jnp.zeros_like(preds)
    timestep_mask = jnp.ones((2, 3), dtype=bool)
    action_pad_mask = jnp.ones_like(preds, dtype=bool)

    if pool_strategy != "pass":
        loss, metrics = head.apply(
            params,
            transformer_outputs,
            actions,
            timestep_mask,
            action_pad_mask,
            method=head.loss,
        )
        assert loss.shape == ()
        assert metrics["loss"].shape == ()


def test_l1_action_head_inherits_loss_type():
    head = L1ActionHead(readout_key="obs")
    assert head.loss_type == "l1"


def test_mse_action_head_uses_map_head(transformer_outputs):
    head = MSEActionHead(readout_key="obs", action_horizon=1, action_dim=2)
    params = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)
    preds = head.apply(params, transformer_outputs, train=False)
    assert preds.shape == (2, 3, 1, 2)


def test_diffusion_action_head_forward_loss_and_sampling(transformer_outputs):
    head = DiffusionActionHead(
        readout_key="obs",
        action_horizon=1,
        action_dim=2,
        time_dim=8,
        num_blocks=1,
        hidden_dim=16,
        diffusion_steps=4,
        dropout_rate=0.0,
    )

    params = head.init(jax.random.PRNGKey(1), transformer_outputs, train=True)

    dummy_time = jnp.zeros((2, 3, 1), dtype=jnp.int32)
    dummy_noisy = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    output = head.apply(
        params,
        transformer_outputs,
        dummy_time,
        dummy_noisy,
        train=False,
    )
    assert output.shape == (2, 3, 2)

    actions = jnp.zeros((2, 3, 1, 2))
    timestep_mask = jnp.ones((2, 3), dtype=bool)
    action_pad_mask = jnp.ones_like(actions, dtype=bool)

    loss, metrics = head.apply(
        params,
        transformer_outputs,
        actions,
        timestep_mask,
        action_pad_mask,
        method=head.loss,
        rngs={"dropout": jax.random.PRNGKey(2)},
    )

    assert loss.shape == ()
    assert "mse" in metrics

    sampled = head.apply(
        params,
        transformer_outputs,
        rng=jax.random.PRNGKey(3),
        sample_shape=(2,),
        method=head.predict_action,
    )

    assert sampled.shape == (2, 2, 3, 1, 2)
