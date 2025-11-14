from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crossformer.model.components.diffusion import (
    cosine_beta_schedule,
    create_diffusion_model,
    FourierFeatures,
    MLP,
    MLPResNet,
    MLPResNetBlock,
    ScoreActor,
)


def test_cosine_beta_schedule_properties():
    betas = cosine_beta_schedule(10)
    assert betas.shape == (10,)
    assert jnp.all((betas > 0) & (betas < 1))
    assert jnp.all(betas[1:] >= betas[:-1] - 1e-6)


def test_fourier_features_learnable_and_fixed():
    inputs = jnp.ones((2, 1))
    learnable = FourierFeatures(output_size=4, learnable=True)
    params = learnable.init(jax.random.PRNGKey(0), inputs)
    outputs = learnable.apply(params, inputs)
    assert outputs.shape == (2, 4)

    fixed = FourierFeatures(output_size=6, learnable=False)
    params_fixed = fixed.init(jax.random.PRNGKey(1), inputs)
    outputs_fixed = fixed.apply(params_fixed, inputs)
    assert outputs_fixed.shape == (2, 6)
    np.testing.assert_allclose(outputs_fixed[0, :3], outputs_fixed[1, :3])


def test_mlp_and_resnet_blocks():
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
    assert mlp_out.shape == (3, 4)

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
    assert block_out.shape == (3, 4)


def test_mlp_resnet_stack():
    resnet = MLPResNet(num_blocks=2, out_dim=5, hidden_dim=16, dropout_rate=0.0, use_layer_norm=True)
    vars_resnet = resnet.init(jax.random.PRNGKey(6), jnp.ones((2, 4)), train=False)
    out = resnet.apply(vars_resnet, jnp.ones((2, 4)), train=False)
    assert out.shape == (2, 5)


def test_score_actor_composition():
    module = ScoreActor(
        time_preprocess=FourierFeatures(output_size=4),
        cond_encoder=MLP((8, 4)),
        reverse_network=MLPResNet(num_blocks=1, out_dim=3, hidden_dim=8),
    )
    variables = module.init(
        jax.random.PRNGKey(7),
        obs_enc=jnp.ones((2, 3)),
        actions=jnp.ones((2, 3)),
        time=jnp.ones((2, 1)),
        train=False,
    )
    out = module.apply(
        variables,
        obs_enc=jnp.ones((2, 3)),
        actions=jnp.ones((2, 3)),
        time=jnp.ones((2, 1)),
        train=True,
    )
    assert out.shape == (2, 3)


def test_create_diffusion_model_pipeline():
    model = create_diffusion_model(
        out_dim=4,
        time_dim=4,
        num_blocks=1,
        dropout_rate=0.0,
        hidden_dim=8,
        use_layer_norm=True,
    )
    variables = model.init(
        jax.random.PRNGKey(8),
        obs_enc=jnp.ones((2, 4)),
        actions=jnp.ones((2, 4)),
        time=jnp.ones((2, 1)),
        train=False,
    )
    outputs = model.apply(
        variables,
        obs_enc=jnp.ones((2, 4)),
        actions=jnp.ones((2, 4)),
        time=jnp.ones((2, 1)),
        train=True,
    )
    assert outputs.shape == (2, 4)
