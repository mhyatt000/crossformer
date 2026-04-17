"""Pure-JAX SoftRas silhouette rasterizer.

Boolean-mask-only differentiable rasterizer. Renders a soft silhouette in
[0, 1] from triangle meshes in OpenGL clip space. No z-buffer, no textures,
no colour -- gradients flow to vertex positions via smooth per-triangle
sigmoid occupancy composed by multiplicative union.

Reference: Liu et al. 2019, "Soft Rasterizer" (silhouette term only).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_EPS = 1e-8


def _edge_sdist(p0: jax.Array, p1: jax.Array, px: jax.Array) -> jax.Array:
    """Signed dist from each pixel px(H,W,2) to oriented line p0->p1.

    Safe against zero-length edges (rsqrt with eps inside).
    """
    d = p1 - p0
    inv_len = jax.lax.rsqrt((d * d).sum() + _EPS)
    n = jnp.stack([-d[1], d[0]]) * inv_len
    return jnp.einsum("hwi,i->hw", px - p0, n)


def _tri_log_outside(tri: jax.Array, px: jax.Array, sigma: float) -> jax.Array:
    """log(1 - occupancy) per pixel for one triangle (3,2). Winding-agnostic.

    Edge-on triangles (projected area ~= 0) are gated out so they contribute
    no occupancy -- otherwise a zero-area tri would drop the mask to 0.5.
    """
    a, b, c = tri[0], tri[1], tri[2]
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = _edge_sdist(a, b, px)
    d2 = _edge_sdist(b, c, px)
    d3 = _edge_sdist(c, a, px)
    dmin = jnp.minimum(jnp.minimum(d1, d2), d3)  # >0 inside a CCW tri
    dmax = jnp.maximum(jnp.maximum(d1, d2), d3)
    inside = jnp.maximum(dmin, -dmax)  # >0 inside either winding
    # Area gate: saturates to 1 when |area| >> sigma^2, 0 when edge-on.
    gate = jnp.tanh(jnp.abs(area2) / (sigma * sigma + _EPS))
    # log(1 - sigmoid(x)) = -softplus(x). Gate scales the log-outside.
    return -jax.nn.softplus(inside / sigma) * gate


def silhouette(
    verts_clip: jax.Array,
    tris: jax.Array,
    H: int,
    W: int,
    *,
    sigma: float = 1e-3,
    chunk: int = 256,
) -> jax.Array:
    """Soft silhouette mask (H, W) in [0, 1].

    verts_clip: (V, 4) OpenGL clip-space xyzw.
    tris:       (T, 3) int vertex indices.
    sigma:      soft-edge width in NDC units (viewport span = 2).
    chunk:      triangles per scan step; lower = less peak memory.

    Row 0 is the top of the image (y = +1 in NDC).
    """
    ndc = verts_clip[:, :2] / verts_clip[:, 3:4]  # (V, 2)
    tri_ndc = ndc[tris]  # (T, 3, 2)

    y = jnp.linspace(1.0, -1.0, H)
    x = jnp.linspace(-1.0, 1.0, W)
    px = jnp.stack(jnp.meshgrid(x, y, indexing="xy"), axis=-1)  # (H, W, 2)

    # Pad with a far-away non-degenerate triangle so scan chunks are uniform.
    T = tri_ndc.shape[0]
    pad = (-T) % chunk
    if pad:
        far = jnp.array([[1e3, 1e3], [1e3 + 1.0, 1e3], [1e3, 1e3 + 1.0]], dtype=tri_ndc.dtype)
        tri_ndc = jnp.concatenate([tri_ndc, jnp.broadcast_to(far, (pad, 3, 2))], axis=0)
    chunks = tri_ndc.reshape(-1, chunk, 3, 2)

    per_tri = jax.vmap(_tri_log_outside, in_axes=(0, None, None))

    def body(log_out, c):
        return log_out + jnp.sum(per_tri(c, px, sigma), axis=0), None

    log_out0 = jnp.zeros((H, W), dtype=jnp.float32)
    log_out, _ = jax.lax.scan(body, log_out0, chunks)
    return 1.0 - jnp.exp(log_out)
