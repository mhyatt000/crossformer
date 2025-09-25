"""Diffusion module cookbook aligned with :mod:`tests.model.components.test_diffusion`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.diffusion import (
    FourierFeatures,
    MLP,
    MLPResNet,
    MLPResNetBlock,
    ScoreActor,
    cosine_beta_schedule,
    create_diffusion_model,
)


def beta_schedule_demo() -> jnp.ndarray:
    """Return a cosine beta schedule for a small number of timesteps."""
    return cosine_beta_schedule(10)


def fourier_features_demo() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Contrast learnable and fixed Fourier feature projections."""
    inputs = jnp.ones((2, 1))
    learnable = FourierFeatures(output_size=4, learnable=True)
    params_learnable = learnable.init(jax.random.PRNGKey(0), inputs)
    outputs_learnable = learnable.apply(params_learnable, inputs)

    fixed = FourierFeatures(output_size=6, learnable=False)
    params_fixed = fixed.init(jax.random.PRNGKey(1), inputs)
    outputs_fixed = fixed.apply(params_fixed, inputs)
    return outputs_learnable, outputs_fixed


def mlp_and_resnet_demo() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a feed-forward network and a residual block with dropout RNGs."""
    mlp = MLP((8, 4), use_layer_norm=True, dropout_rate=0.1, activate_final=True)
    variables = mlp.init(
        {"params": jax.random.PRNGKey(2), "dropout": jax.random.PRNGKey(3)},
        jnp.ones((3, 2)),
        train=True,
    )
    mlp_out = mlp.apply(
        variables,
        jnp.ones((3, 2)),
        train=True,
        rngs={"dropout": jax.random.PRNGKey(4)},
    )

    block = MLPResNetBlock(features=4, act=jax.nn.relu, dropout_rate=0.1, use_layer_norm=True)
    block_vars = block.init(
        {"params": jax.random.PRNGKey(5), "dropout": jax.random.PRNGKey(6)},
        jnp.ones((3, 4)),
        train=True,
    )
    block_out = block.apply(
        block_vars,
        jnp.ones((3, 4)),
        train=True,
        rngs={"dropout": jax.random.PRNGKey(7)},
    )
    return mlp_out, block_out


def resnet_stack_demo() -> jnp.ndarray:
    """Stack residual blocks with :class:`MLPResNet`."""
    resnet = MLPResNet(num_blocks=2, out_dim=5, hidden_dim=16, dropout_rate=0.0, use_layer_norm=True)
    vars_resnet = resnet.init(jax.random.PRNGKey(8), jnp.ones((2, 4)), train=False)
    return resnet.apply(vars_resnet, jnp.ones((2, 4)), train=False)


def score_actor_demo() -> jnp.ndarray:
    """Combine Fourier features, conditioning MLPs, and a reverse network."""
    module = ScoreActor(
        time_preprocess=FourierFeatures(output_size=4),
        cond_encoder=MLP((8, 4)),
        reverse_network=MLPResNet(num_blocks=1, out_dim=3, hidden_dim=8),
    )
    variables = module.init(
        jax.random.PRNGKey(9),
        obs_enc=jnp.ones((2, 3)),
        actions=jnp.ones((2, 3)),
        time=jnp.ones((2, 1)),
        train=False,
    )
    return module.apply(
        variables,
        obs_enc=jnp.ones((2, 3)),
        actions=jnp.ones((2, 3)),
        time=jnp.ones((2, 1)),
        train=True,
    )


def diffusion_model_demo() -> jnp.ndarray:
    """Instantiate the bundled diffusion policy model."""
    model = create_diffusion_model(
        out_dim=4,
        time_dim=4,
        num_blocks=1,
        dropout_rate=0.0,
        hidden_dim=8,
        use_layer_norm=True,
    )
    variables = model.init(
        jax.random.PRNGKey(10),
        obs_enc=jnp.ones((2, 4)),
        actions=jnp.ones((2, 4)),
        time=jnp.ones((2, 1)),
        train=False,
    )
    return model.apply(
        variables,
        obs_enc=jnp.ones((2, 4)),
        actions=jnp.ones((2, 4)),
        time=jnp.ones((2, 1)),
        train=True,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("betas", beta_schedule_demo().shape)
    learnable, fixed = fourier_features_demo()
    print("fourier", learnable.shape, fixed.shape)
    mlp_out, block_out = mlp_and_resnet_demo()
    print("mlp", mlp_out.shape, "block", block_out.shape)
    print("resnet", resnet_stack_demo().shape)
    print("score actor", score_actor_demo().shape)
    print("diffusion", diffusion_model_demo().shape)
