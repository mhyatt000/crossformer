"""
Dynamic Time Warping in JAX.

Two implementations:
  1. dtw_numpy_ref   — wraps dtw-python for validation
  2. dtw_jax         — pure JAX, jit/vmap/grad compatible
  3. dtw_jax_batched — vmapped over batch dim

Notes:
  - JAX DTW uses `jax.lax.scan` over anti-diagonals for parallelism
    (standard row-major scan is sequential and slow under jit).
  - Alternatively: simple nested scan (sequential) for clarity/correctness first.
  - Both are provided: `dtw_jax_sequential` and `dtw_jax_scan`.
"""

from __future__ import annotations

from rich import print
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple


np.set_printoptions(precision=2, linewidth=200)

# ---------------------------------------------------------------------------
# 1. Reference: dtw-python wrapper
# ---------------------------------------------------------------------------

def dtw_numpy_ref(x: np.ndarray, y: np.ndarray) -> float:
    """
    Reference implementation using dtw-python.
    Args:
        x: (T,) or (T, D) array
        y: (S,) or (S, D) array
    Returns:
        normalized DTW distance (float)
    """
    from dtw import dtw as _dtw
    result = _dtw(x, y, keep_internals=False)
    return result.normalizedDistance


# ---------------------------------------------------------------------------
# 2. JAX: Sequential (row-major scan) — correct, jit-able, not vmap-friendly
#         over sequence dims (vmap over batch is fine)
# ---------------------------------------------------------------------------

def _cost_matrix(x: jax.Array, y: jax.Array) -> jax.Array:
    """Euclidean distance matrix. Shape: (T, S)."""
    # x: (T, D), y: (S, D)
    return jnp.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))


@partial(jax.jit, static_argnames=())
def dtw_jax_sequential(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Standard DTW via sequential scan over rows.
    
    Args:
        x: (T, D) — query sequence
        y: (S, D) — reference sequence
    Returns:
        scalar DTW distance (unnormalized)

    Complexity: O(T*S) time, O(T*S) space (cost matrix materialized).
    jit: Yes. vmap over batch: Yes (outer dim). grad: Yes.
    Static shapes required (no dynamic T/S).
    """
    C = _cost_matrix(x, y)  # (T, S)
    T, S = C.shape

    INF = jnp.inf

    def scan_row(carry, c_row):
        # carry: (S,) — previous dtw row
        # c_row: (S,) — current cost row
        def scan_col(prev_val, j):
            diag = carry[j - 1]  # D[i-1, j-1]
            left = prev_val      # D[i, j-1]
            up   = carry[j]      # D[i-1, j]
            best = jnp.minimum(jnp.minimum(diag, left), up)
            val  = c_row[j] + best
            return val, val

        # j=0: only up neighbor is valid
        d0 = c_row[0] + carry[0]
        # j=1..S-1: full scan
        _, rest = jax.lax.scan(scan_col, d0, jnp.arange(1, S))
        row = jnp.concatenate([d0[None], rest])
        return row, row

    # Initialize: first row — only left neighbors valid
    init_row = jnp.concatenate([
        C[0, :1],
        jnp.cumsum(C[0, 1:]) + C[0, 0] * 0  # will redo properly below
    ])
    # Proper first row: D[0,j] = sum(C[0,0:j+1])
    init_row = jnp.cumsum(C[0])

    # Scan over rows 1..T-1
    _, all_rows = jax.lax.scan(scan_row, init_row, C[1:])
    # all_rows: (T-1, S)

    final_row = jax.lax.cond(
        T > 1,
        lambda: all_rows[-1],
        lambda: init_row,
    )
    return final_row[-1]


# ---------------------------------------------------------------------------
# 3. JAX: Anti-diagonal scan — more parallelism within each diagonal
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def dtw_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    DTW via anti-diagonal wavefront scan (preferred for GPU).

    Each anti-diagonal d = i+j = const can be computed in parallel.
    Uses a (T+1, S+1) accumulator with INF boundaries.

    Args:
        x: (T, D)
        y: (S, D)
    Returns:
        scalar DTW distance
    """
    C = _cost_matrix(x, y)  # (T, S)
    T, S = C.shape
    INF = jnp.finfo(jnp.float32).max / 2

    # Pad accumulator: D[i,j] corresponds to acc[i+1, j+1]
    # acc[0,:] = INF, acc[:,0] = INF, acc[0,0] = 0 (virtual origin)
    acc_init = jnp.full((T + 1, S + 1), INF)
    acc_init = acc_init.at[0, 0].set(0.0)

    n_diags = T + S - 1

    def step(acc, d):
        # d: anti-diagonal index (0-indexed), i+j = d+1 in acc coords
        # Valid (i,j) in cost matrix: i in [0,T), j in [0,S), i+j = d
        i_min = jnp.maximum(0, d - (S - 1))
        i_max = jnp.minimum(T - 1, d)
        n = i_max - i_min + 1

        is_ = jnp.arange(T)  # broadcast over all possible i
        js_ = d - is_        # j = d - i

        valid = (is_ >= i_min) & (is_ <= i_max) & (js_ >= 0) & (js_ < S)

        # Acc indices: [i+1, j+1]
        ai = is_ + 1
        aj = js_ + 1

        diag = acc[ai - 1, aj - 1]
        up   = acc[ai - 1, aj]
        left = acc[ai,     aj - 1]

        best = jnp.minimum(jnp.minimum(diag, up), left)
        cost = C[is_, js_]
        new_val = cost + best

        # Only update valid cells
        new_val = jnp.where(valid, new_val, acc[ai, aj])
        acc = acc.at[ai, aj].set(new_val)
        return acc, None

    final_acc, _ = jax.lax.scan(step, acc_init, jnp.arange(n_diags))
    return final_acc[T, S]


# ---------------------------------------------------------------------------
# 4. Batched: vmap over leading batch dim
# ---------------------------------------------------------------------------

dtw_jax_batched = jax.jit(
    jax.vmap(dtw_jax, in_axes=(0, 0))
)
"""
dtw_jax_batched(x_batch, y_batch) -> (B,) distances
x_batch: (B, T, D), y_batch: (B, S, D)
Note: T, S must be identical across batch (static shapes).
"""


# ---------------------------------------------------------------------------
# 5. Soft-DTW (differentiable, grad through distance)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("gamma",))
def soft_dtw_jax(x: jax.Array, y: jax.Array, gamma: float = 1.0) -> jax.Array:
    """
    Soft-DTW: smooth approximation via log-sum-exp min.
    Differentiable w.r.t. x and y.
    
    soft_min(a,b,c) = -gamma * logsumexp([-a,-b,-c] / gamma)
    
    Args:
        x: (T, D)
        y: (S, D)
        gamma: smoothing (→0 recovers hard DTW)
    Returns:
        scalar soft-DTW value
    """
    C = _cost_matrix(x, y)
    T, S = C.shape
    INF = jnp.finfo(jnp.float32).max / 2

    def soft_min3(a, b, c):
        vals = jnp.stack([-a, -b, -c]) / gamma
        return -gamma * jax.nn.logsumexp(vals)

    def scan_row(carry, c_row):
        def scan_col(prev_val, j):
            sm = soft_min3(carry[j - 1], prev_val, carry[j])
            val = c_row[j] + sm
            return val, val

        d0 = c_row[0] + carry[0]
        _, rest = jax.lax.scan(scan_col, d0, jnp.arange(1, S))
        row = jnp.concatenate([d0[None], rest])
        return row, row

    init_row = jnp.cumsum(C[0])
    _, all_rows = jax.lax.scan(scan_row, init_row, C[1:])
    final_row = jax.lax.cond(T > 1, lambda: all_rows[-1], lambda: init_row)
    return final_row[-1]


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    T, S, D = 50, 60, 16

    x_np = rng.standard_normal((T, D)).astype(np.float32)
    y_np = rng.standard_normal((S, D)).astype(np.float32)
    x_jx = jnp.array(x_np)
    y_jx = jnp.array(y_np)

    # Reference
    # ref = dtw_numpy_ref(x_np, y_np)
    # print(f"numpy-ref (normalized): {ref:.6f}")

    # JAX sequential
    d_seq = dtw_jax_sequential(x_jx, y_jx).block_until_ready()
    print(f"jax-sequential:         {d_seq:.6f}")

    # JAX anti-diagonal
    d_ad = dtw_jax(x_jx, y_jx).block_until_ready()
    print(f"jax-antidiag:           {d_ad:.6f}")
    print(f"match: {jnp.allclose(d_seq, d_ad, atol=1e-4)}")

    # Soft-DTW gradient
    grad_fn = jax.grad(soft_dtw_jax, argnums=0)
    g = grad_fn(x_jx, y_jx, gamma=0.1)
    print(f"soft-dtw grad shape:    {g.shape}")

    # Batched timing
    B = 32
    xb = jnp.array(rng.standard_normal((B, T, D)).astype(np.float32))
    yb = jnp.array(rng.standard_normal((B, S, D)).astype(np.float32))
    _ = dtw_jax_batched(xb, yb).block_until_ready()  # warmup
    t0 = time.perf_counter()
    for _ in range(100):
        dtw_jax_batched(xb, yb).block_until_ready()
    print(f"batched B={B}: {(time.perf_counter()-t0)/100*1000:.2f} ms/call")
