"""Block-transformer walkthrough guided by :mod:`tests.model.components.test_block_transformer`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crossformer.model.components.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
    TokenMetadata,
)


def _make_groups(batch_size: int = 2, horizon: int = 3, embed_dim: int = 8):
    """Build prefix and timestep token groups with causal attention defaults."""
    prefix_tokens = jnp.ones((batch_size, 1, embed_dim))
    prefix_group = PrefixGroup(
        tokens=prefix_tokens,
        mask=jnp.ones((batch_size, 1), dtype=bool),
        pos_enc=jnp.zeros_like(prefix_tokens),
        name="task",
        attention_rules={"task": AttentionRule.CAUSAL},
    )
    timestep_tokens = jnp.ones((batch_size, horizon, 2, embed_dim))
    timestep_group = TimestepGroup(
        tokens=timestep_tokens,
        mask=jnp.ones((batch_size, horizon, 2), dtype=bool),
        pos_enc=jnp.zeros_like(timestep_tokens),
        name="obs",
        attention_rules={"task": AttentionRule.CAUSAL, "obs": AttentionRule.CAUSAL},
    )
    return [prefix_group], [timestep_group]


def metadata_rules_demo() -> dict[str, bool]:
    """Check directed attention decisions between prefix and timestep metadata."""
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=2, embed_dim=4)
    prefix_meta = TokenMetadata.create(prefix_groups[0], timestep=-1)
    timestep_now = TokenMetadata.create(timestep_groups[0], timestep=1)
    timestep_past = TokenMetadata.create(timestep_groups[0], timestep=0)
    return {
        "prefix_to_prefix": prefix_meta.should_attend_to(prefix_meta),
        "prefix_to_now": prefix_meta.should_attend_to(timestep_now),
        "now_to_prefix": timestep_now.should_attend_to(prefix_meta),
        "now_to_past": timestep_now.should_attend_to(timestep_past),
        "past_to_now": timestep_past.should_attend_to(timestep_now),
    }


def block_roundtrip_demo() -> dict[str, tuple[int, ...]]:
    """Run the transformer and confirm the split/merge keeps shapes intact."""
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
    prefix_out, timestep_out = block.apply(params, prefix_groups, timestep_groups, train=False)
    tokens, _ = block.assemble_input_tokens(prefix_groups, timestep_groups)
    rebuilt_prefix, rebuilt_timestep = block.split_output_tokens(tokens, prefix_groups, timestep_groups)
    return {
        "prefix_out": prefix_out[0].tokens.shape,
        "timestep_out": timestep_out[0].tokens.shape,
        "rebuilt_prefix": rebuilt_prefix[0].tokens.shape,
        "rebuilt_timestep": rebuilt_timestep[0].tokens.shape,
    }


def attention_mask_demo() -> np.ndarray:
    """Generate the dense attention mask used by the block transformer."""
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=2, embed_dim=4)
    block = BlockTransformer(transformer_kwargs=dict(num_layers=1, mlp_dim=8, num_attention_heads=2))
    mask = block.generate_attention_mask(prefix_groups, timestep_groups)
    return np.array(mask[0, 0])


def pad_attention_mask_demo() -> np.ndarray:
    """Blend causal and padding masks into a single attention tensor."""
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=2, embed_dim=2)
    timestep_groups[0] = timestep_groups[0].replace(mask=jnp.array([[[True, False], [True, True]]]))
    block = BlockTransformer(transformer_kwargs=dict(num_layers=1, mlp_dim=8, num_attention_heads=2))
    generated = block.generate_pad_attention_mask(prefix_groups, timestep_groups)
    return np.array(generated[0, 0])


def causality_check_demo() -> bool:
    """Verify that relaxing causal rules triggers the internal assertion."""
    prefix_groups, timestep_groups = _make_groups(batch_size=1, horizon=1)
    timestep_groups[0] = timestep_groups[0].replace(
        attention_rules={"task": AttentionRule.ALL, "obs": AttentionRule.ALL}
    )
    block = BlockTransformer(transformer_kwargs=dict(num_layers=1, mlp_dim=8, num_attention_heads=2))
    try:
        block.verify_causality(prefix_groups, timestep_groups)
    except AssertionError:
        return True
    return False


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("metadata", metadata_rules_demo())
    shapes = block_roundtrip_demo()
    print("roundtrip", shapes)
    print("attn mask", attention_mask_demo().shape)
    print("pad mask", pad_attention_mask_demo().shape)
    print("causality guard", causality_check_demo())
