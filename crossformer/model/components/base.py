from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from crossformer.utils.mytyping import Sequence


@flax.struct.dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: jax.Array of shape (..., n_tokens, token_dim)
        mask: jax.Array of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
        view: optional jax.Array of shape (..., n_tokens) int32 camera-view id per
            token (0 = NO_VIEW / global, 1..MAX_VIEWS = camera). None ≡ all zeros.
            Random per-forward view permutations make group order unreliable, so
            view identity must travel with the tokens.
    """

    tokens: jax.typing.ArrayLike
    mask: jax.typing.ArrayLike
    view: jax.typing.ArrayLike | None = flax.struct.field(default=None, kw_only=True)

    @classmethod
    def create(cls, tokens: jax.typing.ArrayLike, mask: jax.typing.ArrayLike = None, **kwargs):
        if mask is None:
            mask = jnp.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence[TokenGroup], axis=-2):
        data = jnp.concatenate([t.tokens for t in group_list], axis=axis)
        mask = jnp.concatenate([t.mask for t in group_list], axis=axis + 1)
        view = None
        if any(t.view is not None for t in group_list):
            view = jnp.concatenate(
                [t.view if t.view is not None else jnp.zeros(jnp.shape(t.mask), dtype=jnp.int32) for t in group_list],
                axis=axis + 1,
            )
        return cls(data, mask, view=view)
