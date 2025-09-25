"""TokenGroup helpers drawn from :mod:`tests.model.components.test_base`."""

from __future__ import annotations

import jax.numpy as jnp

from crossformer.model.components.base import TokenGroup


def create_group_demo() -> TokenGroup:
    """Create a token group while letting the helper synthesize a mask."""
    tokens = jnp.ones((2, 3, 4))
    return TokenGroup.create(tokens)


def concatenate_demo() -> TokenGroup:
    """Concatenate two groups with different mask patterns."""
    g1 = TokenGroup.create(jnp.zeros((1, 2, 4)), jnp.array([[1, 0]]))
    g2 = TokenGroup.create(jnp.ones((1, 3, 4)), jnp.array([[1, 1, 0]]))
    return TokenGroup.concatenate([g1, g2])


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("create", create_group_demo().mask.shape)
    print("concat", concatenate_demo().tokens.shape)
