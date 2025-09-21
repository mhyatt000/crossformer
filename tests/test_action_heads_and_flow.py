import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

# @codex what is the purpose of this block? jax>=0.5.3
if not hasattr(jax.config, "define_bool_state"):
    try:
        from jax._src.config import bool_state as _bool_state
    except ImportError:  # pragma: no cover - fallback for older versions

        def _bool_state(name, default, _help):
            setattr(jax.config, name, default)
            return default

    jax.config.define_bool_state = _bool_state

# @codex re: above
if "jax.experimental.maps" not in sys.modules:
    dummy_env = SimpleNamespace(
        physical_mesh=SimpleNamespace(devices=SimpleNamespace(shape=()))
    )
    maps_module = SimpleNamespace(thread_resources=SimpleNamespace(env=dummy_env))
    sys.modules["jax.experimental.maps"] = maps_module
    setattr(jax.experimental, "maps", maps_module)

if not hasattr(jax.nn, "normalize"):

    def _normalize(x, axis=-1, epsilon=1e-12):
        norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
        return x / jnp.maximum(norm, epsilon)

    jax.nn.normalize = _normalize

from crossformer.model.components.action_heads import ContinuousActionHead
from crossformer.model.components.action_heads import DiffusionActionHead
from crossformer.model.components.action_heads import FlowMatchingActionHead
from crossformer.model.components.base import TokenGroup


def make_transformer_outputs(
    batch_size: int,
    window_size: int,
    num_tokens: int,
    embedding_dim: int,
):
    tokens = jnp.arange(
        batch_size * window_size * num_tokens * embedding_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, window_size, num_tokens, embedding_dim)
    mask = jnp.ones((batch_size, window_size, num_tokens), dtype=bool)
    return {"obs": TokenGroup.create(tokens=tokens, mask=mask)}


def make_masks(batch_size: int, window_size: int, horizon: int, action_dim: int):
    timestep_mask = jnp.ones((batch_size, window_size), dtype=bool)
    action_mask = jnp.ones((batch_size, window_size, horizon, action_dim), dtype=bool)
    return timestep_mask, action_mask


def test_continuous_action_head_forward_and_masks():
    batch_size, window_size = 2, 3
    horizon, action_dim = 2, 3
    transformer_outputs = make_transformer_outputs(batch_size, window_size, 4, 6)

    head = ContinuousActionHead(
        readout_key="obs",
        action_horizon=horizon,
        action_dim=action_dim,
        max_action=1.5,
        pool_strategy="mean",
        clip_pred=True,
    )

    variables = head.init(jax.random.PRNGKey(0), transformer_outputs, train=False)
    mean = head.apply(variables, transformer_outputs, train=False)

    assert mean.shape == (batch_size, window_size, horizon, action_dim)
    assert jnp.max(jnp.abs(mean)) <= head.max_action + 1e-6

    samples = head.apply(
        variables,
        transformer_outputs,
        train=False,
        sample_shape=(5,),
        method=head.predict_action,
    )
    assert samples.shape == (5, batch_size, window_size, horizon, action_dim)
    assert jnp.allclose(samples[0], mean)

    actions = mean
    actions_modified = actions.at[1].add(0.5)

    timestep_mask, action_mask = make_masks(
        batch_size, window_size, horizon, action_dim
    )

    masked_loss, metrics = head.apply(
        variables,
        transformer_outputs,
        actions,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, False]),
        train=False,
        method=head.loss,
    )
    masked_loss_modified, _ = head.apply(
        variables,
        transformer_outputs,
        actions_modified,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, False]),
        train=False,
        method=head.loss,
    )
    unmasked_loss, _ = head.apply(
        variables,
        transformer_outputs,
        actions_modified,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, True]),
        train=False,
        method=head.loss,
    )

    assert jnp.allclose(masked_loss, masked_loss_modified)
    assert unmasked_loss > masked_loss
    assert metrics["loss"] == masked_loss
    assert set(metrics) == {"loss", "mse", "lsign"}


def test_diffusion_action_head_end_to_end():
    batch_size, window_size = 2, 2
    horizon, action_dim = 1, 2
    transformer_outputs = make_transformer_outputs(batch_size, window_size, 3, 4)

    head = DiffusionActionHead(
        readout_key="obs",
        action_horizon=horizon,
        action_dim=action_dim,
        diffusion_steps=4,
        num_blocks=1,
        time_dim=8,
        hidden_dim=16,
        dropout_rate=0.0,
    )

    variables = head.init(jax.random.PRNGKey(42), transformer_outputs, train=False)

    with pytest.raises(ValueError):
        head.apply(variables, transformer_outputs, train=False)

    time = jnp.zeros((batch_size, window_size, 1), dtype=jnp.float32)
    noisy = jnp.zeros(
        (batch_size, window_size, horizon * action_dim), dtype=jnp.float32
    )
    eps = head.apply(
        variables,
        transformer_outputs,
        time=time,
        noisy_actions=noisy,
        train=False,
    )
    assert eps.shape == (batch_size, window_size, horizon * action_dim)

    timestep_mask, action_mask = make_masks(
        batch_size, window_size, horizon, action_dim
    )
    actions = jnp.zeros(
        (batch_size, window_size, horizon, action_dim), dtype=jnp.float32
    )
    actions_modified = actions.at[1].add(1.0)

    rng = jax.random.PRNGKey(0)
    masked_loss, metrics = head.apply(
        variables,
        transformer_outputs,
        actions,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, False]),
        train=False,
        rngs={"dropout": rng},
        method=head.loss,
    )
    masked_loss_modified, _ = head.apply(
        variables,
        transformer_outputs,
        actions_modified,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, False]),
        train=False,
        rngs={"dropout": rng},
        method=head.loss,
    )
    unmasked_loss, _ = head.apply(
        variables,
        transformer_outputs,
        actions_modified,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, True]),
        train=False,
        rngs={"dropout": rng},
        method=head.loss,
    )

    assert jnp.allclose(masked_loss, masked_loss_modified)
    assert unmasked_loss > masked_loss
    assert metrics["loss"] == masked_loss
    assert set(metrics) == {"loss", "mse", "lsign"}

    sampled = head.apply(
        variables,
        transformer_outputs,
        rng=jax.random.PRNGKey(123),
        train=False,
        sample_shape=(3,),
        method=head.predict_action,
    )
    assert sampled.shape == (3, batch_size, window_size, horizon, action_dim)
    assert jnp.max(jnp.abs(sampled)) <= head.max_action + 1e-6


def test_flow_matching_action_head_end_to_end():
    batch_size, window_size = 2, 2
    horizon, action_dim = 2, 2
    transformer_outputs = make_transformer_outputs(batch_size, window_size, 3, 5)

    head = FlowMatchingActionHead(
        readout_key="obs",
        action_horizon=horizon,
        action_dim=action_dim,
        flow_steps=3,
        num_blocks=1,
        time_dim=8,
        hidden_dim=16,
        dropout_rate=0.0,
    )

    variables = head.init(jax.random.PRNGKey(7), transformer_outputs, train=False)

    with pytest.raises(ValueError):
        head.apply(variables, transformer_outputs, train=False)

    time = jnp.full((batch_size, window_size, 1), 0.5, dtype=jnp.float32)
    current = jnp.zeros(
        (batch_size, window_size, horizon, action_dim), dtype=jnp.float32
    )
    velocity = head.apply(
        variables,
        transformer_outputs,
        time=time,
        current=current,
        train=False,
    )
    assert velocity.shape == (batch_size, window_size, horizon * action_dim)

    timestep_mask, action_mask = make_masks(
        batch_size, window_size, horizon, action_dim
    )
    actions = jnp.zeros(
        (batch_size, window_size, horizon, action_dim), dtype=jnp.float32
    )
    actions_modified = actions.at[1].add(1.0)

    rng = jax.random.PRNGKey(5)
    masked_loss, metrics = head.apply(
        variables,
        transformer_outputs,
        actions,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, False]),
        train=False,
        rngs={"dropout": rng},
        method=head.loss,
    )
    masked_loss_modified, _ = head.apply(
        variables,
        transformer_outputs,
        actions_modified,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, False]),
        train=False,
        rngs={"dropout": rng},
        method=head.loss,
    )
    unmasked_loss, _ = head.apply(
        variables,
        transformer_outputs,
        actions_modified,
        timestep_mask,
        action_mask,
        action_head_mask=jnp.array([True, True]),
        train=False,
        rngs={"dropout": rng},
        method=head.loss,
    )

    assert jnp.allclose(masked_loss, masked_loss_modified)
    assert unmasked_loss > masked_loss
    assert metrics["loss"] == masked_loss
    assert set(metrics) == {"loss", "mse", "lsign"}

    samples = head.apply(
        variables,
        transformer_outputs,
        rng=jax.random.PRNGKey(11),
        train=False,
        sample_shape=(2,),
        method=head.predict_action,
    )
    assert samples.shape == (2, batch_size, window_size, horizon, action_dim)
    assert jnp.max(jnp.abs(samples)) <= head.max_action + 1e-6
