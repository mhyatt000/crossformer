"""Focused action head snippets inspired by :mod:`tests.model.components.test_action_heads`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.action_heads import (
    ContinuousActionHead,
    DiffusionActionHead,
    L1ActionHead,
    MSEActionHead,
    continuous_loss,
)
from crossformer.model.components.base import TokenGroup


def _make_token_group(batch: int = 2, window: int = 3, tokens: int = 4, dim: int = 8) -> dict[str, TokenGroup]:
    """Utility that mirrors the fixture used throughout the tests."""
    values = jnp.linspace(0, 1, batch * window * tokens * dim).reshape(batch, window, tokens, dim)
    mask = jnp.ones((batch, window, tokens), dtype=bool)
    return {"obs": TokenGroup(tokens=values, mask=mask)}


def loss_metric_demo() -> dict[str, jnp.ndarray]:
    """Compare mean squared and L1 losses with shared masks."""
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.array([[2.0, 2.0], [1.0, 6.0]])
    mask = jnp.array([[1.0, 0.0], [1.0, 1.0]])
    mse_loss, mse_metrics = continuous_loss(pred, target, mask, loss_type="mse")
    l1_loss, _ = continuous_loss(pred, target, mask, loss_type="l1")
    return {"mse": mse_loss, "metrics": mse_metrics["lsign"], "l1": l1_loss}


def continuous_head_pool_demo(pool_strategy: str = "mean") -> jnp.ndarray:
    """Forward the regression head with either mean or pass pooling."""
    outputs = _make_token_group()
    if pool_strategy == "pass":
        tokens = outputs["obs"].tokens.reshape(2, 3, 2, 16)
        mask = jnp.ones((2, 3, 2), dtype=bool)
        outputs = {"obs": TokenGroup(tokens=tokens, mask=mask)}

    head = ContinuousActionHead(
        readout_key="obs",
        pool_strategy=pool_strategy,
        action_horizon=2,
        action_dim=3,
    )
    variables = head.init(jax.random.PRNGKey(0), outputs, train=True)
    preds = head.apply(variables, outputs, train=True)
    return preds


def specialized_head_demo() -> tuple[int, int]:
    """Confirm the light wrappers swap the loss function or pooling strategy."""
    l1_head = L1ActionHead(readout_key="obs")
    mse_head = MSEActionHead(readout_key="obs", action_horizon=1, action_dim=2)
    outputs = _make_token_group()
    l1_params = l1_head.init(jax.random.PRNGKey(1), outputs, train=False)
    mse_params = mse_head.init(jax.random.PRNGKey(2), outputs, train=False)
    l1_preds = l1_head.apply(l1_params, outputs, train=False)
    mse_preds = mse_head.apply(mse_params, outputs, train=False)
    return l1_preds.shape[-1], mse_preds.shape[-1]


def diffusion_head_pipeline_demo() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform score estimation, loss computation, and sampling in one go."""
    outputs = _make_token_group()
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
    params = head.init(jax.random.PRNGKey(3), outputs, train=True)
    time = jnp.zeros((2, 3, 1), dtype=jnp.int32)
    noisy = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    score = head.apply(params, outputs, time, noisy, train=False)

    actions = jnp.zeros((2, 3, 1, 2))
    timestep_mask = jnp.ones((2, 3), dtype=bool)
    action_mask = jnp.ones_like(actions, dtype=bool)
    loss, metrics = head.apply(
        params,
        outputs,
        actions,
        timestep_mask,
        action_mask,
        method=head.loss,
        rngs={"dropout": jax.random.PRNGKey(4)},
    )
    samples = head.apply(
        params,
        outputs,
        rng=jax.random.PRNGKey(5),
        sample_shape=(2,),
        method=head.predict_action,
    )
    return score, loss, samples


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("loss demo", loss_metric_demo())
    print("continuous mean", continuous_head_pool_demo("mean").shape)
    print("continuous pass", continuous_head_pool_demo("pass").shape)
    print("specialized", specialized_head_demo())
    score, loss, samples = diffusion_head_pipeline_demo()
    print("diffusion score", score.shape, "loss", loss.shape, "samples", samples.shape)
