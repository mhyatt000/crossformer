import jax
import jax.numpy as jnp
import numpy as np
import pytest
from crossformer.model.components.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
    TokenMetadata,
)


def _make_groups(batch_size=2, horizon=3, embed_dim=8):
    prefix_tokens = jnp.ones((batch_size, 1, embed_dim))
    prefix_pos = jnp.zeros_like(prefix_tokens)
    prefix_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
    prefix_group = PrefixGroup(
        tokens=prefix_tokens,
        mask=prefix_mask,
        pos_enc=prefix_pos,
        name="task",
        attention_rules={"task": AttentionRule.CAUSAL},
    )

    timestep_tokens = jnp.ones((batch_size, horizon, 2, embed_dim))
    timestep_pos = jnp.zeros_like(timestep_tokens)
    timestep_mask = jnp.ones((batch_size, horizon, 2), dtype=jnp.bool_)
    timestep_group = TimestepGroup(
        tokens=timestep_tokens,
        mask=timestep_mask,
        pos_enc=timestep_pos,
        name="obs",
        attention_rules={"task": AttentionRule.CAUSAL, "obs": AttentionRule.CAUSAL},
    )
    return [prefix_group], [timestep_group]


def test_token_metadata_rules():
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=2, embed_dim=4)
    prefix_meta = TokenMetadata.create(prefix_groups[0], timestep=-1)
    timestep_meta_now = TokenMetadata.create(timestep_groups[0], timestep=1)
    timestep_meta_past = TokenMetadata.create(timestep_groups[0], timestep=0)

    assert prefix_meta.should_attend_to(prefix_meta) is True
    assert prefix_meta.should_attend_to(timestep_meta_now) is False
    assert timestep_meta_now.should_attend_to(prefix_meta) is True
    assert timestep_meta_now.should_attend_to(timestep_meta_past) is True
    assert timestep_meta_past.should_attend_to(timestep_meta_now) is False


def test_block_transformer_call_and_split_roundtrip():
    prefix_groups, timestep_groups = _make_groups()
    block = BlockTransformer(
        transformer_kwargs=dict(
            num_layers=1,
            mlp_dim=16,
            num_attention_heads=2,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            repeat_pos_enc=False,
        )
    )

    params = block.init(jax.random.PRNGKey(0), prefix_groups, timestep_groups, train=False)
    prefix_out, timestep_out = block.apply(
        params, prefix_groups, timestep_groups, train=False
    )

    assert len(prefix_out) == 1 and len(timestep_out) == 1
    assert prefix_out[0].tokens.shape == prefix_groups[0].tokens.shape
    assert timestep_out[0].tokens.shape == timestep_groups[0].tokens.shape

    tokens, pos = block.assemble_input_tokens(prefix_groups, timestep_groups)
    rebuilt_prefix, rebuilt_timestep = block.split_output_tokens(
        tokens, prefix_groups, timestep_groups
    )
    np.testing.assert_allclose(rebuilt_prefix[0].tokens, prefix_groups[0].tokens)
    np.testing.assert_allclose(
        rebuilt_timestep[0].tokens, timestep_groups[0].tokens
    )


def test_attention_mask_generation():
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=2, embed_dim=4)
    block = BlockTransformer(transformer_kwargs=dict(num_layers=1, mlp_dim=8, num_attention_heads=2))
    mask = block.generate_attention_mask(prefix_groups, timestep_groups)

    assert mask.shape == (1, 1, 5, 5)

    # Expected pattern: prefix attends to prefix, timestep tokens attend to prefix and causal past.
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask[0, 0], expected)


def test_generate_pad_attention_mask_respects_padding():
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=2, embed_dim=2)
    timestep_groups[0] = timestep_groups[0].replace(
        mask=jnp.array([[[True, False], [True, True]]])
    )

    block = BlockTransformer(transformer_kwargs=dict(num_layers=1, mlp_dim=8, num_attention_heads=2))
    generated = block.generate_pad_attention_mask(prefix_groups, timestep_groups)

    assert generated.shape == (1, 1, 5, 5)
    pad_vector = jnp.concatenate(
        [prefix_groups[0].mask, timestep_groups[0].mask.reshape(1, -1)], axis=1
    )
    expected = jnp.broadcast_to(pad_vector[:, None, :], (1, 5, 5))[0]
    np.testing.assert_array_equal(generated[0, 0], expected)


def test_verify_causality_raises_for_future_attention():
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=1)
    timestep_groups[0] = timestep_groups[0].replace(
        attention_rules={"task": AttentionRule.ALL, "obs": AttentionRule.ALL}
    )
    block = BlockTransformer(transformer_kwargs=dict(num_layers=1, mlp_dim=8, num_attention_heads=2))

    with pytest.raises(AssertionError):
        block.verify_causality(prefix_groups, timestep_groups)
