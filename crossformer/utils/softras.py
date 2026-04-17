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


def _point_to_segment_sq(px: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Squared distance from pixels px(H,W,2) to segment a->b (unsigned)."""
    ab = b - a
    ab_sq = (ab * ab).sum() + _EPS
    t = jnp.einsum("hwi,i->hw", px - a, ab) / ab_sq
    t = jnp.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    diff = px - closest
    return (diff * diff).sum(axis=-1)


def _tri_log_outside(tri: jax.Array, px: jax.Array, sigma: float) -> jax.Array:
    """log(1 - occupancy) per pixel for one triangle (3,2). Winding-agnostic.

    Occupancy = sigmoid(signed_dist / sigma) where signed_dist is positive
    inside the triangle, negative outside. Distance is to the triangle as a
    2D region (closest edge *segment*, not infinite line), so occupancy
    decays with actual Euclidean distance from the mesh instead of leaking
    along edge extensions.
    """
    a, b, c = tri[0], tri[1], tri[2]
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # Inside test via cross-product signs (no normalization; signs only).
    c1 = (b[0] - a[0]) * (px[..., 1] - a[1]) - (b[1] - a[1]) * (px[..., 0] - a[0])
    c2 = (c[0] - b[0]) * (px[..., 1] - b[1]) - (c[1] - b[1]) * (px[..., 0] - b[0])
    c3 = (a[0] - c[0]) * (px[..., 1] - c[1]) - (a[1] - c[1]) * (px[..., 0] - c[0])
    inside = ((c1 >= 0) & (c2 >= 0) & (c3 >= 0)) | ((c1 <= 0) & (c2 <= 0) & (c3 <= 0))

    # Unsigned distance to triangle boundary via closest edge segment.
    d_sq = jnp.minimum(
        jnp.minimum(_point_to_segment_sq(px, a, b), _point_to_segment_sq(px, b, c)),
        _point_to_segment_sq(px, c, a),
    )
    d = jnp.sqrt(d_sq + _EPS)
    signed = jnp.where(inside, d, -d)

    gate = jnp.tanh(jnp.abs(area2) / (sigma * sigma + _EPS))
    return -jax.nn.softplus(signed / sigma) * gate


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

    # jax.checkpoint: backward rematerializes per-step activations instead of
    # saving them for every step of the scan (O(steps) memory -> O(1)).
    @jax.checkpoint
    def body(log_out, c):
        return log_out + jnp.sum(per_tri(c, px, sigma), axis=0), None

    log_out0 = jnp.zeros((H, W), dtype=jnp.float32)
    log_out, _ = jax.lax.scan(body, log_out0, chunks)
    return 1.0 - jnp.exp(log_out)
