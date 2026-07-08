from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from crossformer.model.bela import BELAModel, BELATransformer
from crossformer.model.components.base import TokenGroup
from crossformer.model.crossformer_model import CrossFormerModel

pytestmark = pytest.mark.nn

B, W, N_TOK, E = 2, 3, 5, 8


class StubTokenizer(nn.Module):
    """Emits observations['x'] as a TokenGroup with a view id per token."""

    @nn.compact
    def __call__(self, observations, tasks=None, train: bool = False) -> TokenGroup:
        tokens = observations["x"]  # (B, W, N_TOK, E)
        mask = jnp.ones(tokens.shape[:-1], dtype=bool)
        view = jnp.ones(tokens.shape[:-1], dtype=jnp.int32)
        return TokenGroup(tokens, mask, view=view)


@pytest.fixture
def trunk():
    return BELATransformer(
        observation_tokenizers={"stub": StubTokenizer()},
        task_tokenizers={},
        readouts={"action": 4},
        token_embedding_size=16,
        num_layers=1,
        num_heads=2,
    )


@pytest.fixture
def obs():
    return {
        "x": jnp.linspace(0.0, 1.0, B * W * N_TOK * E, dtype=jnp.float32).reshape(B, W, N_TOK, E),
    }


def test_bela_trunk_shapes_and_pad_mask(trunk, obs):
    pad_mask = jnp.array([[True, True, False], [True, True, True]])
    params = trunk.init(jax.random.PRNGKey(0), obs, {}, pad_mask, train=False)
    out = trunk.apply(params, obs, {}, pad_mask, train=False)

    assert set(out) == {"readout_action"}
    group = out["readout_action"]
    assert group.tokens.shape == (B, W, 4, 16)
    assert group.mask.shape == (B, W, 4)
    # latent mask mirrors the timestep pad mask
    assert bool(group.mask[0, 2, 0]) is False
    assert bool(group.mask[0, 1, 0]) is True
    assert jnp.isfinite(group.tokens).all()


def test_bela_trunk_ignores_masked_tokens(trunk, obs):
    """Latents must be invariant to content at tokenizer-masked positions."""

    class HalfMaskedTokenizer(StubTokenizer):
        @nn.compact
        def __call__(self, observations, tasks=None, train: bool = False) -> TokenGroup:
            g = super().__call__(observations, tasks, train=train)
            mask = g.mask.at[..., N_TOK // 2 :].set(False)
            return TokenGroup(g.tokens, mask, view=g.view)

    trunk_m = BELATransformer(
        observation_tokenizers={"stub": HalfMaskedTokenizer()},
        task_tokenizers={},
        readouts={"action": 4},
        token_embedding_size=16,
        num_layers=1,
        num_heads=2,
    )
    pad_mask = jnp.ones((B, W), dtype=bool)
    params = trunk_m.init(jax.random.PRNGKey(0), obs, {}, pad_mask, train=False)
    out_a = trunk_m.apply(params, obs, {}, pad_mask, train=False)["readout_action"].tokens

    obs_b = {"x": obs["x"].at[:, :, N_TOK // 2 :, :].set(1e3)}
    out_b = trunk_m.apply(params, obs_b, {}, pad_mask, train=False)["readout_action"].tokens
    assert jnp.allclose(out_a, out_b, atol=1e-5)


def test_bela_model_module_cls_hook():
    assert BELAModel.module_cls.__name__ == "BELAModule"
    assert CrossFormerModel.module_cls.__name__ == "CrossFormerModule"
    # pytree registration: BELAModel must flatten like its parent (params and
    # example_batch are pytree fields; the rest are static). An unregistered
    # subclass would flatten to itself as a single leaf.
    model = BELAModel(
        module=None,
        text_processor=None,
        config={},
        params={"w": jnp.zeros(3)},
        example_batch={},
        dataset_statistics=None,
    )
    leaves = jax.tree.leaves(model)
    assert len(leaves) == 1 and leaves[0].shape == (3,)
