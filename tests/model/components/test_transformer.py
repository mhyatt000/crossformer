import jax
import jax.numpy as jnp
import numpy as np

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.transformer import (
    AddPositionEmbs,
    Encoder1DBlock,
    MAPHead,
    MlpBlock,
    Transformer,
    common_transformer_sizes,
)


def test_add_position_embs_adds_bias():
    module = AddPositionEmbs(
        posemb_init=lambda key, shape, dtype=jnp.float32: jnp.ones(shape, dtype)
    )
    inputs = jnp.zeros((2, 3, 4))
    variables = module.init(jax.random.PRNGKey(0), inputs)
    outputs = module.apply(variables, inputs)
    np.testing.assert_array_equal(outputs, jnp.ones_like(inputs))


def test_mlp_block_output_shape():
    block = MlpBlock(mlp_dim=8, dropout_rate=0.0)
    variables = block.init(jax.random.PRNGKey(1), jnp.ones((2, 4)), deterministic=True)
    outputs = block.apply(variables, jnp.ones((2, 4)), deterministic=True)
    assert outputs.shape == (2, 4)


def test_map_head_with_token_group_and_array():
    tokens = jnp.ones((2, 5, 6))
    mask = jnp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    group = TokenGroup(tokens=tokens, mask=mask)

    head = MAPHead(num_heads=2, num_readouts=2)
    variables = head.init(jax.random.PRNGKey(2), group, train=False)
    out_group = head.apply(variables, group, train=False)
    assert out_group.shape == (2, 2, 6)

    out_array = head.apply(variables, tokens, train=False)
    assert out_array.shape == (2, head.num_readouts, tokens.shape[-1])


def test_encoder_block_respects_mask_and_positional_encoding():
    block = Encoder1DBlock(
        mlp_dim=8,
        num_heads=2,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        repeat_pos_enc=True,
    )
    x = jnp.ones((2, 4, 6))
    pos_enc = jnp.linspace(0, 1, 2 * 4 * 6).reshape(2, 4, 6)
    attn_mask = jnp.ones((2, 1, 4, 4))
    variables = block.init(jax.random.PRNGKey(3), x, pos_enc, attn_mask, deterministic=True)
    out = block.apply(variables, x, pos_enc, attn_mask, deterministic=True)
    assert out.shape == x.shape


def test_transformer_encoder_stack():
    transformer = Transformer(
        num_layers=2,
        mlp_dim=16,
        num_attention_heads=2,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        repeat_pos_enc=False,
    )
    x = jnp.ones((1, 4, 8))
    pos_enc = jnp.zeros_like(x)
    attn_mask = jnp.ones((1, 1, 4, 4))
    variables = transformer.init(jax.random.PRNGKey(4), x, pos_enc, attn_mask, train=False)
    out = transformer.apply(variables, x, pos_enc, attn_mask, train=False)
    assert out.shape == x.shape


def test_common_transformer_sizes_return_expected_values():
    dim, config = common_transformer_sizes("vit_b")
    assert dim == 768
    assert config["num_layers"] == 12
    assert config["num_attention_heads"] == 12

    dim_dummy, config_dummy = common_transformer_sizes("dummy")
    assert dim_dummy == 256
    assert config_dummy["num_layers"] == 1
