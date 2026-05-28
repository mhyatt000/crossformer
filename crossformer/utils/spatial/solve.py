from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie


def levenberg_marquardt(
    residual_fn,
    theta0,
    *,
    max_iters=50,
    lambda0=1e-3,
    tol=1e-6,
):
    """
    Minimize 0.5 * ||residual_fn(theta)||^2 using Levenberg-Marquardt.

    residual_fn: theta -> residual vector (... flattened)
    theta0: parameter vector, shape (D,)
    """

    def loss(theta):
        r = residual_fn(theta)
        return 0.5 * jnp.sum(r**2)

    def step(state, _):
        theta, damping, prev_loss = state

        r = residual_fn(theta)  # (N,)
        J = jax.jacrev(residual_fn)(theta)  # (N, D)

        A = J.T @ J + damping * jnp.eye(theta.shape[0])
        b = -J.T @ r

        delta = jnp.linalg.solve(A, b)
        theta_new = theta + delta
        new_loss = loss(theta_new)

        accept = new_loss < prev_loss

        theta_next = jnp.where(accept, theta_new, theta)
        loss_next = jnp.where(accept, new_loss, prev_loss)

        # If step helped, trust Gauss-Newton more.
        # If step failed, increase damping toward gradient descent.
        damping_next = jnp.where(accept, damping * 0.3, damping * 10.0)

        return (theta_next, damping_next, loss_next), {
            "loss": loss_next,
            "damping": damping_next,
            "step_norm": jnp.linalg.norm(delta),
            "accepted": accept,
        }

    init_state = (theta0, lambda0, loss(theta0))
    final_state, history = jax.lax.scan(step, init_state, None, length=max_iters)

    theta_final, _damping_final, _final_loss = final_state
    return theta_final, history


def rotation_6d_to_matrix(r6):
    """
    Zhou et al. 6D rotation representation -> SO(3).
    r6: (..., 6)
    returns: (..., 3, 3)
    """
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]

    b1 = a1 / (jnp.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    a2_orth = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / (jnp.linalg.norm(a2_orth, axis=-1, keepdims=True) + 1e-8)

    b3 = jnp.cross(b1, b2, axis=-1)

    return jnp.stack([b1, b2, b3], axis=-1)


def project_points(K, R, t, X):
    """
    K: (3, 3)
    R: (3, 3)
    t: (3,)
    X: (N, 3)  world/robot points
    returns: (N, 2) pixel coordinates
    """
    X_cam = X @ R.T + t  # (N, 3)

    x = X_cam[:, 0]
    y = X_cam[:, 1]
    z = X_cam[:, 2]

    # avoid divide-by-zero
    z = jnp.clip(z, 1e-6, None)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy

    return jnp.stack([u, v], axis=-1)


def pnp_reprojection_residual(theta, K, X_3d, x_2d):
    """
    theta: (9,) = [tx, ty, tz, r6...]
    K: (3, 3)
    X_3d: (N, 3)
    x_2d: (N, 2)

    returns: (2N,) residual vector
    """
    t = theta[:3]
    r6 = theta[3:9]

    R = rotation_6d_to_matrix(r6)
    x_proj = project_points(K, R, t, X_3d)

    residual = x_proj - x_2d
    return residual.reshape(-1)


# residual_fn = lambda theta: pnp_reprojection_residual(
# theta, K=K, X_3d=X_3d, x_2d=x_2d,
# )
# theta_hat, hist = levenberg_marquardt(residual_fn, theta0)


def weighted_pnp_residual(theta, K, X_3d, x_2d, weights):
    """
    weights: (N,) confidence weights
    """
    t = theta[:3]
    r6 = theta[3:9]

    R = rotation_6d_to_matrix(r6)
    x_proj = project_points(K, R, t, X_3d)

    residual = x_proj - x_2d  # (N, 2)
    residual = _weight_residual(residual, weights)

    return residual.reshape(-1)


def _weight_residual(residual, weights):
    if weights is None:
        return residual
    return residual * weights[:, None] ** 0.5


def project_points_lie(K: jax.Array, X_cam: jax.Array, eps: float = 1e-6) -> jax.Array:
    """
    K: (3, 3)
    X_cam: (N, 3)
    returns: (N, 2)
    """
    z = jnp.clip(X_cam[:, 2], eps, None)

    u = K[0, 0] * X_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * X_cam[:, 1] / z + K[1, 2]

    return jnp.stack([u, v], axis=-1)


def pnp_residual(
    T_cam_world: jaxlie.SE3,
    K: jax.Array,
    X_world: jax.Array,
    x_obs: jax.Array,
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    T_cam_world: transform from world/robot frame to camera frame
    K: (3, 3)
    X_world: (N, 3)
    x_obs: (N, 2)

    returns: (2N,)
    """
    X_cam = jax.vmap(T_cam_world.apply)(X_world)
    x_proj = project_points_lie(K, X_cam)
    residual = x_proj - x_obs

    return _weight_residual(residual, weights).reshape(-1)


def lm_step_se3(
    T: jaxlie.SE3,
    K: jax.Array,
    X_world: jax.Array,
    x_obs: jax.Array,
    damping: float,
    weights: jax.Array | None = None,
):
    """
    One LM step:
        T_new = T @ SE3.exp(delta)
    """

    def residual_delta(delta):
        T_delta = jaxlie.manifold.rplus(T, delta)
        return pnp_residual(T_delta, K, X_world, x_obs, weights)

    r = residual_delta(jnp.zeros(6))
    J = jax.jacrev(residual_delta)(jnp.zeros(6))  # (2N, 6)

    A = J.T @ J + damping * jnp.eye(6)
    b = -J.T @ r

    delta = jnp.linalg.solve(A, b)
    T_candidate = jaxlie.manifold.rplus(T, delta)

    old_loss = 0.5 * jnp.sum(r**2)
    new_r = pnp_residual(T_candidate, K, X_world, x_obs, weights)
    new_loss = 0.5 * jnp.sum(new_r**2)

    accept = new_loss < old_loss

    T_next = jax.tree.map(
        lambda a, b: jnp.where(accept, a, b),
        T_candidate,
        T,
    )

    damping_next = jnp.where(accept, damping * 0.3, damping * 10.0)
    loss_next = jnp.where(accept, new_loss, old_loss)

    info = {
        "loss": loss_next,
        "step_norm": jnp.linalg.norm(delta),
        "accepted": accept,
        "damping": damping_next,
    }

    return T_next, damping_next, info


def levenberg_marquardt_se3_pnp(
    T0: jaxlie.SE3,
    K: jax.Array,
    X_world: jax.Array,
    x_obs: jax.Array,
    *,
    max_iters: int = 50,
    damping0: float = 1e-3,
    weights: jax.Array | None = None,
):
    def body(state, _):
        T, damping = state
        T_next, damping_next, info = lm_step_se3(
            T,
            K,
            X_world,
            x_obs,
            damping,
            weights,
        )
        return (T_next, damping_next), info

    init_state = (T0, jnp.asarray(damping0))
    (T_final, _damping_final), hist = jax.lax.scan(
        body,
        init_state,
        None,
        length=max_iters,
    )

    return T_final, hist
