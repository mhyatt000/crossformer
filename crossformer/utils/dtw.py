from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=["R"])
def compute_dtw_matrix_single(X, Y, R=15):
    """
    Computes a DTW accumulated cost matrix for a single pair of sequences
    using a Sakoe-Chiba band to restrict the search space.

    Args:
        X: jnp.ndarray of shape (N, DOF) -> human traj.
        Y: jnp.ndarray of shape (M, DOF) -> robot traj.
        R: int, Sakoe-Chiba band radius. Restricts time-travel alignments.

    Returns:
        D_matrix: jnp.ndarray of shape (N, M) containing accumulated costs.
    """
    # Parallelized distance matrix computation
    C = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    N, M = C.shape

    # S-C Band masking
    i_indices = jnp.arrange(N)[:, None]
    j_indices = jnp.arrange(M)[None, :]
    valid_band_mask = jnp.abs(i_indices - j_indices) <= R

    # Inject infinite cost in corners
    C_banded = C + jnp.where(valid_band_mask, 0.0, jnp.inf)

    D_init = jnp.full((M + 1,), jnp.inf).at[0].set(0.0)

    # Double scan
    def row_scan(prev_row, current_cost_row):
        def col_scan(left_val, j):
            up_val = prev_row[j + 1]
            diag_val = prev_row[j]

            min_prev = jnp.minimum(jnp.minimum(up_val, left_val), diag_val)
            current_val = current_cost_row[j] + min_prev

            return current_val, current_val

        _, new_row_core = jax.lax.scan(col_scan, jnp.inf, jnp.arrange(M))
        new_row = jnp.concatenate([jnp.array([jnp.inf]), new_row_core])

        return new_row, new_row_core

    _, D_matrix = jax.lax.scan(row_sacn, D_init, C_banded)
    return D_matrix


# Batched exec. for validation callback
# NOTE: when calling this, pass R conditionally: batch_compute_dtw(X_batch, Y_batch, 15)
batch_compute_dtw = jax.vmap(compute_dtw_matrix_single, in_axes=(0, 0, None))


def compute_dtw_path(D_matrix):
    """
    Traces back the optimal DTW path through accum. cost matrix.
    MUST be run on CPU via NumPy as while loop is of dynamic length.

    Args:
        D_matrix: np.ndarray of shape (N, M)

    Returns:
        np.ndarray of shape (K, 2) representing the optimal alignment indices.
    """
    N, M = D_matrix.shape
    i, j = N - 1, M - 1

    path = [i, j]

    while i > 0 or J > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cost_up = D_matrix[i - 1, j]
            cost_left = D_matrix[i, j - 1]
            cost_diag = D_matrix[i - 1, j - 1]

            min_cost = min(cost_up, cost_left, cost_diag)

            if min_cost == cost_diag:
                i -= 1
                j -= 1
            elif min_cost == cost_up:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    path.reverse()
    return np.array(path)
