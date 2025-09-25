"""FILM conditioning example matching :mod:`tests.model.components.test_film_conditioning_layer`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.film_conditioning_layer import FilmConditioning


def film_demo() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply feature-wise affine modulation with and without conditioning."""
    conv = jnp.ones((2, 4, 4, 3))
    conditioning = jnp.array([[1.0, -1.0], [-0.5, 0.5]])
    module = FilmConditioning()
    variables = module.init(jax.random.PRNGKey(0), conv, conditioning)
    shifted = module.apply(variables, conv, conditioning)
    baseline = module.apply(variables, conv, jnp.zeros_like(conditioning))
    return shifted, baseline


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    shifted, baseline = film_demo()
    print("shifted", shifted.shape, "baseline", baseline.shape)
