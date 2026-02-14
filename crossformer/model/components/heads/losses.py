"""Loss functions and utilities for action heads."""

from __future__ import annotations

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


def masked_mean(x, mask):
    """Compute masked mean."""
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> tuple[Array, dict[str, Array]]:
    """Compute continuous action loss with optional L1/MSE.

    Args:
        pred_value: predicted values, shape (batch_dims...)
        ground_truth_value: continuous ground truth values, shape (batch_dims...)
        mask: boolean mask broadcastable to ground_truth shape
        loss_type: "mse" or "l1"

    Returns:
        loss: scalar loss value
        metrics: dict with loss, mse, and sign error metrics
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)

    sign_deltas = jnp.logical_or(
        jnp.logical_and(ground_truth_value > 0, pred_value <= 0),
        jnp.logical_and(ground_truth_value <= 0, pred_value > 0),
    )
    lsign = masked_mean(sign_deltas, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "lsign": lsign,
    }


def sample_tau(key: jax.random.PRNGKey, shape: tuple[int, ...], s: float = 0.999) -> Array:
    """Sample time steps from a beta distribution for flow matching.

    Args:
        key: JAX random key
        shape: output shape, e.g. (N,) for N samples
        s: cutoff parameter, default 0.999

    Returns:
        τ samples in [0, s]
    """
    alpha, beta = 1.5, 1.0
    # sample x ~ Beta(alpha, beta) on [0,1]
    x = jax.random.beta(key, alpha, beta, shape=shape)
    # map back: τ = s * (1 - x)
    tau = s * (1.0 - x)
    return tau
