from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.film_conditioning_layer import FilmConditioning


def test_film_conditioning_applies_shift_and_scale():
    conv = jnp.ones((2, 4, 4, 3))
    conditioning = jnp.array([[1.0, -1.0], [-0.5, 0.5]])
    module = FilmConditioning()
    variables = module.init(jax.random.PRNGKey(0), conv, conditioning)

    output = module.apply(variables, conv, conditioning)
    assert output.shape == conv.shape

    zero_cond = module.apply(variables, conv, jnp.zeros_like(conditioning))
    assert jnp.allclose(zero_cond, conv)
