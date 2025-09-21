import jax.numpy as jnp

from crossformer.model.components.base import TokenGroup


def test_token_group_create_default_mask():
    tokens = jnp.ones((2, 3, 4))
    group = TokenGroup.create(tokens)

    assert group.tokens.shape == (2, 3, 4)
    assert group.mask.shape == (2, 3)
    assert jnp.all(group.mask == 1)


def test_token_group_concatenate():
    g1 = TokenGroup.create(jnp.zeros((1, 2, 4)), jnp.array([[1, 0]]))
    g2 = TokenGroup.create(jnp.ones((1, 3, 4)), jnp.array([[1, 1, 0]]))

    combined = TokenGroup.concatenate([g1, g2])

    assert combined.tokens.shape == (1, 5, 4)
    assert combined.mask.shape == (1, 5)
    assert jnp.allclose(combined.tokens[:, :2], 0)
    assert jnp.allclose(combined.tokens[:, 2:], 1)
    assert jnp.array_equal(combined.mask, jnp.array([[1, 0, 1, 1, 0]]))
