"""Composable, named loss terms layered on top of bare loss fns in ``losses.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.embody import (
    ARM_7DOF,
    BodyPart,
    CAM_EXTR,
    CAM_INTR,
    KP2D_ARM10DOF,
)

from .losses import masked_mean

LossFn = Callable[..., tuple[Array, dict[str, Array]]]


def _gram_schmidt_6d(rep6d: ArrayLike) -> Array:
    """Zhou-6D → SO(3). ``rep6d`` is (..., 6); returns (..., 3, 3) with basis as columns."""
    a1 = rep6d[..., 0:3]
    a2 = rep6d[..., 3:6]
    b1 = a1 / jnp.clip(jnp.linalg.norm(a1, axis=-1, keepdims=True), a_min=1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / jnp.clip(jnp.linalg.norm(b2, axis=-1, keepdims=True), a_min=1e-8)
    b3 = jnp.cross(b1, b2)
    return jnp.stack([b1, b2, b3], axis=-1)


def geodesic_loss(
    pred_6d: ArrayLike,
    gt_6d: ArrayLike,
    mask: ArrayLike,
    eps: float = 1e-7,
) -> tuple[Array, dict[str, Array]]:
    """Geodesic angular distance on SO(3) between two Zhou-6D rotations.

    Args:
        pred_6d, gt_6d: (..., 6) Zhou-6D rotation representations.
        mask: broadcastable to the leading dims (one entry per rotation, not per scalar).
        eps: clamp on cos(θ) to keep ``arccos`` gradients finite at ±1.
    """
    R_pred = _gram_schmidt_6d(pred_6d)
    R_gt = _gram_schmidt_6d(gt_6d)
    R_rel = jnp.einsum("...ji,...jk->...ik", R_pred, R_gt)  # R_predᵀ @ R_gt
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = jnp.clip((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(cos_theta)  # radians, in [0, π]
    loss = masked_mean(theta, mask)
    return loss, {
        "loss": loss,
        "theta_deg": loss * (180.0 / jnp.pi),
    }


@dataclass
class LossTerm:
    """Thin wrapper that gives a loss fn a name, weight, and DOF scope.

    Provide *exactly one* of ``bodyparts`` or ``dofs``. ``bodyparts`` is preferred:
    it's symbolic (survives DOF-table reorderings) and self-documents intent.
    ``dofs`` is the escape hatch for slices that don't map to a registered part.
    After ``__post_init__``, ``self.dofs`` is always populated.
    """

    name: str
    fn: LossFn
    weight: float = 1.0
    bodyparts: tuple[BodyPart, ...] | None = None
    dofs: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (self.bodyparts is None) == (self.dofs is None):
            raise ValueError(f"{self.name}: exactly one of `bodyparts` or `dofs` must be set")
        if self.bodyparts is not None:
            self.dofs = tuple(d for bp in self.bodyparts for d in bp.dof_ids)

    def __call__(self, *args, **kwargs) -> tuple[Array, dict[str, Array]]:
        loss, metrics = self.fn(*args, **kwargs)
        return self.weight * loss, {f"{self.name}/{k}": v for k, v in metrics.items()}


@dataclass
class GeodesicLoss(LossTerm):
    """Geodesic distance on the Zhou-6D rotation dims of CAM_EXTR (default).

    Note: ``CAM_EXTR`` carries 3 translation dims followed by 6 rotation dims.
    The caller must slice ``[..., 3:]`` from the predicted/GT tensors before
    passing them to ``fn`` — ``geodesic_loss`` only consumes the 6D portion.
    """

    fn: LossFn = geodesic_loss
    bodyparts: tuple[BodyPart, ...] | None = (CAM_EXTR,)


# ---------------------------------------------------------------------------
# Reconstruction loss — joints → FK → world kp3d → extrinsics → camera kp3d
#                       → intrinsics → kp2d, vs GT kp2d (visibility-masked)
# ---------------------------------------------------------------------------


def _project_world_to_pixels(
    kp3d_world: ArrayLike,  # (..., N, 3)
    extr: ArrayLike,  # (..., 9): t_x, t_y, t_z, r6d_0..5  (world→cam)
    intr: ArrayLike,  # (..., 4): fx, fy, cx, cy
    z_min: float = 1e-3,
) -> Array:
    """Pinhole project N world-space 3D points into pixel coordinates."""
    R = _gram_schmidt_6d(extr[..., 3:9])  # (..., 3, 3)
    t = extr[..., 0:3]  # (..., 3)
    kp3d_cam = jnp.einsum("...ij,...nj->...ni", R, kp3d_world) + t[..., None, :]
    z = jnp.clip(kp3d_cam[..., 2], a_min=z_min)
    x_n = kp3d_cam[..., 0] / z
    y_n = kp3d_cam[..., 1] / z
    fx, fy, cx, cy = intr[..., 0], intr[..., 1], intr[..., 2], intr[..., 3]
    u = fx[..., None] * x_n + cx[..., None]
    v = fy[..., None] * y_n + cy[..., None]
    return jnp.stack([u, v], axis=-1)  # (..., N, 2)


def reconstruction_loss(
    joints_pred: ArrayLike,  # (..., 7)        — ARM_7DOF
    extr_pred: ArrayLike,  # (..., 9)        — CAM_EXTR (t_xyz + r6d)
    intr_pred: ArrayLike,  # (..., 4)        — CAM_INTR (fx, fy, cx, cy)
    kp2d_gt: ArrayLike,  # (..., N, 3)     — KP2D_ARM10DOF reshaped: (u, v, vis)
    mask: ArrayLike,  # broadcastable over leading dims
    fk_fn: Callable[[Array], Array],  # joints (..., 7) → kp3d_world (..., N, 3)
) -> tuple[Array, dict[str, Array]]:
    """Re-project predicted FK keypoints into the predicted camera, MSE vs GT 2D.

    The visibility channel of ``kp2d_gt`` is folded into the mask so occluded
    keypoints don't contribute. Caller can swap any of the predicted inputs for
    GT (e.g. fix intrinsics) by passing GT in place of the prediction.
    """
    kp3d_world = fk_fn(joints_pred)  # (..., N, 3)
    kp2d_pred = _project_world_to_pixels(kp3d_world, extr_pred, intr_pred)
    uv_gt = kp2d_gt[..., 0:2]
    vis = kp2d_gt[..., 2:3]  # (..., N, 1)
    err = jnp.square(kp2d_pred - uv_gt)  # (..., N, 2)
    full_mask = vis * jnp.broadcast_to(jnp.asarray(mask)[..., None, None], err.shape)
    loss = masked_mean(err, full_mask)
    pixel_rmse = jnp.sqrt(loss + 1e-12)
    return loss, {"loss": loss, "pixel_rmse": pixel_rmse}


@dataclass
class ReconstructionLoss(LossTerm):
    """Re-projection term: joints --FK--> kp3d_world --extr--> kp3d_cam --intr--> kp2d_pred.

    Default scope covers all four bodyparts the loss reads from. ``fk_fn`` is the
    embodiment-specific forward-kinematics callable; it must produce keypoints in
    the same order as ``KP2D_ARM10DOF`` (base, j1..j7, eef, tcp).
    """

    fn: LossFn = reconstruction_loss
    bodyparts: tuple[BodyPart, ...] | None = (
        ARM_7DOF,
        CAM_EXTR,
        CAM_INTR,
        KP2D_ARM10DOF,
    )
    fk_fn: Callable[[Array], Array] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fk_fn is None:
            raise ValueError(f"{self.name}: `fk_fn` is required (embodiment-specific FK)")

    def __call__(self, *args, **kwargs) -> tuple[Array, dict[str, Array]]:
        kwargs.setdefault("fk_fn", self.fk_fn)
        return super().__call__(*args, **kwargs)
