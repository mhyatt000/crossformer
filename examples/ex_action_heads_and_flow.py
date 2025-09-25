"""Action head walkthrough mirroring :mod:`tests.test_action_heads_and_flow`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.action_heads import (
    ContinuousActionHead,
    DiffusionActionHead,
    FlowMatchingActionHead,
)
from crossformer.model.components.base import TokenGroup


def _make_transformer_outputs(
    batch_size: int = 2,
    window_size: int = 3,
    num_tokens: int = 4,
    embed_dim: int = 6,
) -> dict[str, TokenGroup]:
    """Assemble a tiny token stream that mimics encoder outputs."""
    tokens = jnp.linspace(
        0.0,
        1.0,
        batch_size * window_size * num_tokens * embed_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, window_size, num_tokens, embed_dim)
    mask = jnp.ones((batch_size, window_size, num_tokens), dtype=bool)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


def continuous_head_demo() -> dict[str, jnp.ndarray]:
    """Show mean prediction, sampling, and loss masking for regression heads."""
    outputs = _make_transformer_outputs()
    head = ContinuousActionHead(
        readout_key="obs",
        action_horizon=2,
        action_dim=3,
        max_action=1.5,
    )
    variables = head.init(jax.random.PRNGKey(0), outputs, train=False)
    mean = head.apply(variables, outputs, train=False)

    timestep_mask = jnp.ones(mean.shape[:2], dtype=bool)
    action_mask = jnp.ones_like(mean, dtype=bool)
    loss, metrics = head.apply(
        variables,
        outputs,
        mean,
        timestep_mask,
        action_mask,
        method=head.loss,
    )
    samples = head.apply(
        variables,
        outputs,
        sample_shape=(2,),
        train=False,
        method=head.predict_action,
    )
    return {"mean": mean, "samples": samples, "loss": loss, "metrics": metrics}


def diffusion_head_demo() -> dict[str, jnp.ndarray]:
    """Trace the denoising score network used for diffusion policies."""
    outputs = _make_transformer_outputs(batch_size=2, window_size=2, num_tokens=3)
    head = DiffusionActionHead(
        readout_key="obs",
        action_horizon=1,
        action_dim=2,
        diffusion_steps=4,
        num_blocks=1,
        time_dim=8,
        hidden_dim=16,
        dropout_rate=0.0,
    )
    variables = head.init(jax.random.PRNGKey(1), outputs, train=False)
    time = jnp.zeros((2, 2, 1), dtype=jnp.float32)
    noisy = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    eps = head.apply(variables, outputs, time=time, noisy_actions=noisy, train=False)

    timestep_mask = jnp.ones((2, 2), dtype=bool)
    action_mask = jnp.ones((2, 2, 1, 2), dtype=bool)
    loss, metrics = head.apply(
        variables,
        outputs,
        jnp.zeros_like(action_mask, dtype=jnp.float32),
        timestep_mask,
        action_mask,
        method=head.loss,
        rngs={"dropout": jax.random.PRNGKey(2)},
    )
    predicted = head.apply(
        variables,
        outputs,
        sample_shape=(3,),
        rng=jax.random.PRNGKey(3),
        train=False,
        method=head.predict_action,
    )
    return {"score": eps, "loss": loss, "metrics": metrics, "samples": predicted}


def flow_matching_head_demo() -> dict[str, jnp.ndarray]:
    """Simulate velocity predictions for the flow-matching action head."""
    outputs = _make_transformer_outputs(batch_size=2, window_size=2, num_tokens=3)
    head = FlowMatchingActionHead(
        readout_key="obs",
        action_horizon=2,
        action_dim=2,
        flow_steps=3,
        num_blocks=1,
        time_dim=8,
        hidden_dim=16,
        dropout_rate=0.0,
    )
    variables = head.init(jax.random.PRNGKey(4), outputs, train=False)
    velocity = head.apply(
        variables,
        outputs,
        time=jnp.full((2, 2, 1), 0.5, dtype=jnp.float32),
        current=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        train=False,
    )

    timestep_mask = jnp.ones((2, 2), dtype=bool)
    action_mask = jnp.ones((2, 2, 2, 2), dtype=bool)
    loss, metrics = head.apply(
        variables,
        outputs,
        jnp.zeros_like(action_mask, dtype=jnp.float32),
        timestep_mask,
        action_mask,
        method=head.loss,
        rngs={"dropout": jax.random.PRNGKey(5)},
    )
    samples = head.apply(
        variables,
        outputs,
        sample_shape=(2,),
        rng=jax.random.PRNGKey(6),
        train=False,
        method=head.predict_action,
    )
    return {"velocity": velocity, "loss": loss, "metrics": metrics, "samples": samples}


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    demos = {
        "continuous": continuous_head_demo(),
        "diffusion": diffusion_head_demo(),
        "flow_matching": flow_matching_head_demo(),
    }
    for name, outputs in demos.items():
        summary = {key: tuple(value.shape) for key, value in outputs.items() if hasattr(value, "shape")}
        print(f"{name}: {summary}")
