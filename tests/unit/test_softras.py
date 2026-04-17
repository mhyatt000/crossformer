"""Unit tests for the SoftRas silhouette rasterizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.utils.softras import silhouette


def _unit_cube():
    """Axis-aligned cube in clip-space (w=1), 12 tris. xy spans [-0.3, 0.3]."""
    v = jnp.array(
        [
            [-0.3, -0.3, 0.0, 1.0],
            [0.3, -0.3, 0.0, 1.0],
            [0.3, 0.3, 0.0, 1.0],
            [-0.3, 0.3, 0.0, 1.0],
            [-0.3, -0.3, 0.2, 1.0],
            [0.3, -0.3, 0.2, 1.0],
            [0.3, 0.3, 0.2, 1.0],
            [-0.3, 0.3, 0.2, 1.0],
        ],
        dtype=jnp.float32,
    )
    t = jnp.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=jnp.int32,
    )
    return v, t


@pytest.mark.unit
def test_mask_shape_and_range():
    v, t = _unit_cube()
    m = silhouette(v, t, 64, 64, sigma=1e-3)
    assert m.shape == (64, 64)
    assert m.dtype == jnp.float32
    assert float(m.min()) >= 0.0
    assert float(m.max()) <= 1.0
    assert bool(jnp.isfinite(m).all())


@pytest.mark.unit
def test_center_covered_corners_empty():
    v, t = _unit_cube()
    m = silhouette(v, t, 64, 64, sigma=1e-3)
    assert float(m[32, 32]) > 0.9  # center of image is inside the cube silhouette
    assert float(m[0, 0]) < 0.05  # corner of image is well outside
    assert float(m[63, 63]) < 0.05


@pytest.mark.unit
def test_gradient_is_finite_and_nonzero():
    v, t = _unit_cube()

    def area(vc):
        return silhouette(vc, t, 64, 64, sigma=1e-2).sum()

    g = jax.grad(area)(v)
    assert g.shape == v.shape
    assert bool(jnp.isfinite(g).all())
    assert float(jnp.linalg.norm(g)) > 0.0


@pytest.mark.unit
def test_outward_vertex_motion_grows_mask():
    """Pushing every rightmost vertex outward should increase total mask area."""
    v, t = _unit_cube()
    rng = np.random.default_rng(0)

    def area(vc):
        return silhouette(vc, t, 64, 64, sigma=1e-2).sum()

    a0 = float(area(v))
    # Shift the four x=+0.3 vertices by +0.05 in x.
    idx = jnp.array([1, 2, 5, 6])
    v_big = v.at[idx, 0].add(0.05)
    a1 = float(area(v_big))
    assert a1 > a0


@pytest.mark.unit
def test_jit_compiles():
    v, t = _unit_cube()
    f = jax.jit(lambda vc: silhouette(vc, t, 64, 64, sigma=1e-3))
    m = f(v)
    _ = f(v)  # second call hits cache
    assert m.shape == (64, 64)
