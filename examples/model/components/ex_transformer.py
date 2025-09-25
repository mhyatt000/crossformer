"""Transformer component examples aligned with :mod:`tests.model.components.test_transformer`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.transformer import (
    AddPositionEmbs,
    Encoder1DBlock,
    MAPHead,
    MlpBlock,
    Transformer,
    common_transformer_sizes,
)


def positional_embedding_demo() -> jnp.ndarray:
    """Apply learned position biases to a flat token grid."""
    module = AddPositionEmbs(
        posemb_init=lambda key, shape, dtype=jnp.float32: jnp.ones(shape, dtype)
    )
    inputs = jnp.zeros((2, 3, 4))
    variables = module.init(jax.random.PRNGKey(0), inputs)
    return module.apply(variables, inputs)


def mlp_block_demo() -> jnp.ndarray:
    """Run the feed-forward sub-block with deterministic dropout."""
    block = MlpBlock(mlp_dim=8, dropout_rate=0.0)
    variables = block.init(jax.random.PRNGKey(1), jnp.ones((2, 4)), deterministic=True)
    return block.apply(variables, jnp.ones((2, 4)), deterministic=True)


def map_head_demo() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pool a :class:`TokenGroup` and a raw array with the same parameters."""
    tokens = jnp.ones((2, 5, 6))
    mask = jnp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
    group = TokenGroup(tokens=tokens, mask=mask)

    head = MAPHead(num_heads=2, num_readouts=2)
    variables = head.init(jax.random.PRNGKey(2), group, train=False)
    pooled_group = head.apply(variables, group, train=False)
    pooled_array = head.apply(variables, tokens, train=False)
    return pooled_group, pooled_array


def encoder_block_demo() -> jnp.ndarray:
    """Forward a causal encoder block with explicit attention masks."""
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
    return block.apply(variables, x, pos_enc, attn_mask, deterministic=True)


def transformer_stack_demo() -> jnp.ndarray:
    """Stack multiple encoder blocks using :class:`Transformer`."""
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
    return transformer.apply(variables, x, pos_enc, attn_mask, train=False)


def preset_lookup_demo() -> dict[str, int | tuple[int, ...]]:
    """Inspect bundled configuration presets for convenience sizing."""
    dim, config = common_transformer_sizes("vit_b")
    fallback_dim, fallback_cfg = common_transformer_sizes("dummy")
    return {
        "vit_b_dim": dim,
        "vit_b_layers": config["num_layers"],
        "fallback_dim": fallback_dim,
        "fallback_layers": fallback_cfg["num_layers"],
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("positional", positional_embedding_demo().shape)
    print("mlp", mlp_block_demo().shape)
    pooled_group, pooled_array = map_head_demo()
    print("map_head group", pooled_group.shape)
    print("map_head array", pooled_array.shape)
    print("encoder", encoder_block_demo().shape)
    print("transformer", transformer_stack_demo().shape)
    print("presets", preset_lookup_demo())
