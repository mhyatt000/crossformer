"""Back-propagatable PnP helpers in JAX."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

Array = jax.Array


def _skew(v: Array) -> Array:
    z = jnp.zeros_like(v[..., 0])
    return jnp.stack(
        [
            jnp.stack([z, -v[..., 2], v[..., 1]], axis=-1),
            jnp.stack([v[..., 2], z, -v[..., 0]], axis=-1),
            jnp.stack([-v[..., 1], v[..., 0], z], axis=-1),
        ],
        axis=-2,
    )


def axis_angle_to_matrix(rvec: Array, eps: float = 1e-8) -> Array:
    """Convert axis-angle vectors (..., 3) to rotation matrices (..., 3, 3)."""
    rvec = jnp.asarray(rvec)
    theta = jnp.linalg.norm(rvec, axis=-1, keepdims=True)
    axis = jnp.where(theta > eps, rvec / theta, rvec)
    K = _skew(axis)
    I = jnp.broadcast_to(jnp.eye(3, dtype=rvec.dtype), (*rvec.shape[:-1], 3, 3))
    theta = theta[..., None]
    R = I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)
    return jnp.where(theta > eps, R, I + _skew(rvec))


def matrix_to_axis_angle(R: Array, eps: float = 1e-7) -> Array:
    """Convert rotation matrices (..., 3, 3) to axis-angle vectors (..., 3)."""
    R = jnp.asarray(R)
    trace = jnp.trace(R, axis1=-2, axis2=-1)
    cos_theta = jnp.clip((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(cos_theta)
    vee = jnp.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    scale = theta / (2.0 * jnp.sin(theta))
    small = 0.5 * vee
    return jnp.where(theta[..., None] > 1e-4, scale[..., None] * vee, small)


def transform_points(pose: Array, pts3d: Array, angle_axis: bool = True) -> Array:
    """Transform 3D points by pose (..., 6) or matrices (..., 3, 4)."""
    pose = jnp.asarray(pose)
    pts3d = jnp.asarray(pts3d)
    if angle_axis:
        R = axis_angle_to_matrix(pose[..., :3])
        t = pose[..., 3:6]
        return jnp.einsum("...ij,...nj->...ni", R, pts3d) + t[..., None, :]

    R = pose[..., :3, :3]
    t = pose[..., :3, 3]
    return jnp.einsum("...ij,...nj->...ni", R, pts3d) + t[..., None, :]


def batch_transform_3d(pose: Array, pts3d: Array, angle_axis: bool = True) -> Array:
    """Compatibility wrapper matching RoboPEPP's batched transform helper."""
    return transform_points(pose, pts3d, angle_axis=angle_axis)


def project_points(
    pose: Array,
    pts3d: Array,
    K: Array,
    angle_axis: bool = True,
    z_min: float = 1e-6,
) -> Array:
    """Project 3D points to pixels with a pinhole camera."""
    pts_cam = transform_points(pose, pts3d, angle_axis=angle_axis)
    pix_h = jnp.einsum("...ij,...nj->...ni", K, pts_cam)
    z = jnp.where(jnp.abs(pix_h[..., 2:3]) > z_min, pix_h[..., 2:3], z_min)
    return pix_h[..., :2] / z


def batch_project(pose: Array, pts3d: Array, K: Array, angle_axis: bool = True) -> Array:
    """Compatibility wrapper matching RoboPEPP's batched projection helper."""
    return project_points(pose, pts3d, K, angle_axis=angle_axis)


def get_coefs(pose: Array, pts3d: Array, K: Array, angle_axis: bool = True) -> Array:
    """Return RoboPEPP-style ``-2 * d(project_points) / d(pose)`` coefficients."""
    pose = jnp.asarray(pose)
    pose0 = pose[0] if pose.ndim == 2 else pose
    jac = jax.jacfwd(lambda p: project_points(p, pts3d, K, angle_axis=angle_axis))(pose0)
    return -2.0 * jac


def _pose_from_dlt(pts2d: Array, pts3d: Array, K: Array) -> Array:
    """DLT initialization in normalized camera coordinates."""
    ones = jnp.ones((*pts3d.shape[:-1], 1), dtype=pts3d.dtype)
    X = jnp.concatenate([pts3d, ones], axis=-1)
    K_inv = jnp.linalg.inv(K)
    uv1 = jnp.concatenate([pts2d, jnp.ones((*pts2d.shape[:-1], 1), dtype=pts2d.dtype)], axis=-1)
    xy1 = uv1 @ K_inv.T
    x = xy1[..., 0] / xy1[..., 2]
    y = xy1[..., 1] / xy1[..., 2]
    z = jnp.zeros_like(x)

    row_x = jnp.concatenate([X, z[..., None].repeat(4, axis=-1), -x[..., None] * X], axis=-1)
    row_y = jnp.concatenate([z[..., None].repeat(4, axis=-1), X, -y[..., None] * X], axis=-1)
    A = jnp.reshape(jnp.stack([row_x, row_y], axis=-2), (-1, 12))

    _, _, vh = jnp.linalg.svd(A, full_matrices=False)
    P = vh[-1].reshape(3, 4)
    M = P[:, :3]
    U, s, Vh = jnp.linalg.svd(M)
    R0 = U @ Vh
    sign = jnp.where(jnp.linalg.det(R0) < 0.0, -1.0, 1.0)
    R = sign * R0
    scale = sign * jnp.mean(s)
    t = P[:, 3] / scale
    return jnp.concatenate([matrix_to_axis_angle(R), t], axis=-1)


def _weighted_residual(pose: Array, pts2d: Array, pts3d: Array, K: Array, weights: Array) -> Array:
    err = project_points(pose, pts3d, K) - pts2d
    return (err * jnp.sqrt(weights[..., None])).reshape(-1)


def _lm_step(pose: Array, pts2d: Array, pts3d: Array, K: Array, weights: Array, damping: float) -> Array:
    def residual(p):
        return _weighted_residual(p, pts2d, pts3d, K, weights)

    r = residual(pose)
    J = jax.jacfwd(residual)(pose)
    H = J.T @ J
    g = J.T @ r
    eye = jnp.eye(6, dtype=pose.dtype)
    reg = damping * (jnp.mean(jnp.diag(H)) + 1.0)
    step = jnp.linalg.solve(H + reg * eye, g)
    return pose - step


@partial(jax.jit, static_argnames=("n_iters",))
def solve_pnp_single(
    pts2d: Array,
    pts3d: Array,
    K: Array,
    init_pose: Array | None = None,
    weights: Array | None = None,
    n_iters: int = 10,
    damping: float = 1e-4,
) -> Array:
    """Solve one differentiable PnP problem as angle-axis xyz + translation."""
    pts2d = jnp.asarray(pts2d)
    pts3d = jnp.asarray(pts3d)
    K = jnp.asarray(K)
    pose = _pose_from_dlt(pts2d, pts3d, K) if init_pose is None else jnp.asarray(init_pose)
    weights = jnp.ones(pts2d.shape[:-1], dtype=pts2d.dtype) if weights is None else jnp.asarray(weights)

    for _ in range(n_iters):
        pose = _lm_step(pose, pts2d, pts3d, K, weights, damping)
    return pose


def _broadcast_batch(x: Array, batch: int, event_ndim: int) -> Array:
    x = jnp.asarray(x)
    if x.ndim == event_ndim:
        return jnp.broadcast_to(x, (batch, *x.shape))
    return x


def solve_pnp(
    pts2d: Array,
    pts3d: Array,
    K: Array,
    init_pose: Array | None = None,
    weights: Array | None = None,
    n_iters: int = 10,
    damping: float = 1e-4,
) -> Array:
    """Batched differentiable PnP with shared or per-sample 3D keypoints."""
    pts2d = jnp.asarray(pts2d)
    single = pts2d.ndim == 2
    pts2d_b = pts2d[None] if single else pts2d
    B = pts2d_b.shape[0]
    pts3d_b = _broadcast_batch(pts3d, B, event_ndim=2)
    K_b = _broadcast_batch(K, B, event_ndim=2)
    init_b = None if init_pose is None else _broadcast_batch(init_pose, B, event_ndim=1)
    w_b = None if weights is None else _broadcast_batch(weights, B, event_ndim=1)

    def solve_i(i_pts2d, i_pts3d, i_K, i_init, i_w):
        return solve_pnp_single(i_pts2d, i_pts3d, i_K, i_init, i_w, n_iters, damping)

    if init_b is None:
        if w_b is None:
            w_b = jnp.ones(pts2d_b.shape[:-1], dtype=pts2d_b.dtype)
        poses = jax.vmap(lambda x, z, k, w: solve_pnp_single(x, z, k, None, w, n_iters, damping))(
            pts2d_b, pts3d_b, K_b, w_b
        )
    else:
        if w_b is None:
            w_b = jnp.ones(pts2d_b.shape[:-1], dtype=pts2d_b.dtype)
        poses = jax.vmap(solve_i)(pts2d_b, pts3d_b, K_b, init_b, w_b)
    return poses[0] if single else poses


def solve_pnp_m3d(
    pts2d: Array,
    pts3d: Array,
    K: Array,
    init_pose: Array | None = None,
    weights: Array | None = None,
    n_iters: int = 10,
    damping: float = 1e-4,
) -> Array:
    """Alias for per-sample 3D keypoints, matching RoboPEPP's BPnP_m3d."""
    return solve_pnp(pts2d, pts3d, K, init_pose, weights, n_iters, damping)


bpnp = solve_pnp
bpnp_m3d = solve_pnp_m3d
BPnP = solve_pnp
BPnP_m3d = solve_pnp_m3d
BPnP_fast = solve_pnp
