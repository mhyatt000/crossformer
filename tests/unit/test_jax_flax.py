from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import pytest


class SimpleModel(nn.Module):
    """Minimal one-layer Flax model for GPU testing."""

    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x.mean()  # scalar loss


@pytest.fixture
def dummy_batch():
    """Create small dummy batch (4, 16) for testing."""
    return jnp.ones((4, 16), dtype=jnp.float32)


@pytest.fixture
def model_state(dummy_batch):
    """Initialize model and return (params, model)."""
    model = SimpleModel(hidden_dim=32)
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, dummy_batch)
    return params, model


def test_jax_devices_available():
    """Verify JAX can see available devices."""
    devices = jax.devices()
    assert len(devices) > 0, "No JAX devices found"
    print(f"\n✓ JAX devices: {devices}")
    print(f"  Device count: {len(devices)}")


def test_forward_pass(model_state, dummy_batch):
    """Test forward pass on dummy data."""
    params, model = model_state
    loss = model.apply(params, dummy_batch)
    assert jnp.isfinite(loss), "Loss is not finite"
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    print(f"\n✓ Forward pass OK, loss={loss:.4f}")


def test_backward_pass(model_state, dummy_batch):
    """Test backward pass (gradient computation)."""
    params, model = model_state

    def loss_fn(p):
        return model.apply(p, dummy_batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Verify grads structure matches params
    assert jax.tree.structure(grads) == jax.tree.structure(params)

    # Verify grads are finite
    flat_grads = jax.tree.leaves(grads)
    for grad in flat_grads:
        assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN or Inf"

    print(f"\n✓ Backward pass OK, loss={loss:.4f}")
    print(f"  Grad keys: {list(grads.keys())}")


def test_jit_compiled_fwd_bwd(model_state, dummy_batch):
    """Test JIT-compiled forward and backward pass."""
    params, model = model_state

    @jax.jit
    def train_step(p):
        def loss_fn(p_inner):
            return model.apply(p_inner, dummy_batch)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        return loss, grads

    loss, _grads = train_step(params)

    assert jnp.isfinite(loss), "JIT loss is not finite"
    print(f"\n✓ JIT forward+backward OK, loss={loss:.4f}")


def test_optimizer_step(model_state, dummy_batch):
    """Test one optimizer step with jit."""
    params, model = model_state

    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(p, opt_st):
        def loss_fn(p_inner):
            return model.apply(p_inner, dummy_batch)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt_st = tx.update(grads, opt_st, p)
        new_p = jax.tree.map(lambda x, u: x + u, p, updates)
        return loss, new_p, new_opt_st

    loss, _new, opt_state = train_step(params, opt_state)
    assert jnp.isfinite(loss), "Optimizer step loss is not finite"
    print(f"\n✓ Optimizer step OK, loss={loss:.4f}")


def test_data_parallel_sharding(model_state, dummy_batch):
    """Test data parallel sharding (simulates multi-GPU setup)."""
    params, model = model_state
    devices = jax.devices()

    # Create mesh with batch axis for data parallelism
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # Replicate params across all devices
    params_sharded = jax.device_put(params, replicated_sharding)

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(p, batch):
        def loss_fn(p_inner):
            return model.apply(p_inner, batch)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        return loss, grads

    # For single device, batch sharding has no effect, but test NCCL communication
    loss, _grads = train_step(params_sharded, dummy_batch)

    assert jnp.isfinite(loss), "Sharded loss is not finite"
    print(f"\n✓ Data parallel sharding OK, loss={loss:.4f}")
    print(f"  Devices in mesh: {len(devices)}")


def test_nccl_multi_device_training():
    """Test multi-GPU training with data parallelism (exercises NCCL)."""
    devices = jax.devices()
    num_devices = len(devices)

    if num_devices < 2:
        pytest.skip("Multi-GPU test requires at least 2 GPUs")

    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # Create model with batch size equal to number of devices
    model = SimpleModel(hidden_dim=16)
    rng = jax.random.PRNGKey(42)
    dummy_data = jnp.ones((num_devices, 8), dtype=jnp.float32)
    params = model.init(rng, dummy_data)

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(p, batch):
        def loss_fn(p_inner):
            logits = model.apply(p_inner, batch)
            return jnp.mean(logits)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        return loss, grads

    # This will trigger NCCL communication for sharding the batch across devices
    loss, _grads = train_step(params, dummy_data)

    assert jnp.isfinite(loss), "Loss is not finite"
    print(f"\n✓ Multi-GPU training OK on {num_devices} GPUs")
    print("  Data parallelism with sharding layer exercises NCCL")
    print(f"  Loss: {loss:.4f}")


pytestmark = pytest.mark.unit
