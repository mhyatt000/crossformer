"""Composable, named loss terms layered on top of bare loss fns in ``losses.py``.

Auxiliary (non-flow) terms operate on the one-step denoised estimate
``x1_estimate`` and are τ-gated via ``LossTerm.tau_power``: nonlinear
consistency constraints (FK, reprojection, smoothness) are only well-posed
when x̂₁ approaches an actual sample (τ → 1); at low τ the optimal x̂₁ is a
conditional average that need not satisfy them (Jensen gap). The flow MSE
itself stays un-gated — it is linear in the target, so the conditional mean
is its correct answer at every τ.

Physical terms need raw units: build a per-DOF (mean, std) table once with
``dof_stats_table`` and denormalize slot values with ``denormalize_by_dof``.

Terms are instantiated from ``config/loss.yaml`` via ``load_loss_terms``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from crossformer.data.grain.embody import PART_TO_ACTION_KEY
from crossformer.embody import (
    ARM_7DOF,
    BodyPart,
    CAM_EXTR,
    CAM_INTR,
    Embodiment,
    KP2D_ARM10DOF,
    KP3DC,
    MASK_ID,
    NO_NORM_DOF_IDS,
    VOCAB_SIZE,
)
from crossformer.utils.spec import ModuleFile, ModuleSpec

from .losses import masked_mean

LossFn = Callable[..., tuple[Array, dict[str, Array]]]


# ---------------------------------------------------------------------------
# Flow-time helpers — x̂₁ estimate
# ---------------------------------------------------------------------------


def x1_estimate(a_t: ArrayLike, v_pred: ArrayLike, tau: ArrayLike) -> Array:
    """One-step denoised action estimate.

    Under the head's convention ``a_t = τ·x₁ + (1-τ)·x₀`` with velocity target
    ``v = x₁ - x₀``, the estimate ``x̂₁ = a_t + (1-τ)·v`` recovers x₁ exactly
    when v is exact, at any τ.
    """
    return jnp.asarray(a_t) + (1.0 - jnp.asarray(tau)) * jnp.asarray(v_pred)


# ---------------------------------------------------------------------------
# Per-DOF denormalization — bridge normalized slots to raw units
# ---------------------------------------------------------------------------


def dof_stats_table(
    action_stats: Mapping[str, Any],
    embodiment: Embodiment,
    key_map: dict[str, str] | None = None,
) -> np.ndarray:
    """(VOCAB_SIZE, 2) [mean, std] per DOF id from per-action-key statistics.

    Mirrors the normalization applied in the data pipeline: entries default to
    identity (0, 1); DOF ids in ``NO_NORM_DOF_IDS`` (norm_mask False parts) and
    MASK stay identity. Per-view parts (kp3dc) share one pooled stat across all
    view copies, matching the ``agg`` stats adapter.

    Args:
        action_stats: per-action-key stats with ``.mean`` / ``.std`` arrays
            (e.g. ``DatasetStatistics.action``). Stats are per *dataset* — for
            multi-dataset mixes build one table per dataset.
        embodiment: catalog embodiment (``parts``, not ``expanded`` — per-view
            copies share DOF ids so one write covers all views).
        key_map: override for PART_TO_ACTION_KEY.
    """
    table = np.zeros((VOCAB_SIZE, 2), dtype=np.float32)
    table[:, 1] = 1.0

    km = key_map or PART_TO_ACTION_KEY
    parts_by_key: dict[str, list[BodyPart]] = {}
    for part in embodiment.parts:
        key = km.get(part.name)
        if key is None:
            raise KeyError(f"no action key mapping for body part {part.name!r}")
        parts_by_key.setdefault(key, []).append(part)

    for key, parts in parts_by_key.items():
        stats = action_stats.get(key)
        if stats is None:
            continue
        mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1)
        std = np.asarray(stats.std, dtype=np.float32).reshape(-1)
        dims = sum(p.action_dim for p in parts)
        assert mean.size == dims, f"{key!r}: stats size {mean.size} != part dims {dims}"
        off = 0
        for p in parts:
            ids_ = np.asarray(p.dof_ids)
            table[ids_, 0] = mean[off : off + p.action_dim]
            table[ids_, 1] = std[off : off + p.action_dim]
            off += p.action_dim

    skip = np.asarray([*sorted(NO_NORM_DOF_IDS), MASK_ID])
    table[skip, 0] = 0.0
    table[skip, 1] = 1.0
    return table


def denormalize_by_dof(x: ArrayLike, dof_ids: ArrayLike, table: ArrayLike) -> Array:
    """Map normalized slot values (..., A) to raw units via per-slot DOF ids.

    ``dof_ids`` broadcasts against x's trailing slot axis ((A,), (B, A), ...).
    """
    t = jnp.asarray(table)
    ids = jnp.asarray(dof_ids)
    return jnp.asarray(x) * t[ids, 1] + t[ids, 0]


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
    # τ-gate exponent: the effective weight is weight * τ^tau_power. 0 = ungated
    # (flow MSE); aux consistency terms should use > 0 so they only bite where
    # the one-step x̂₁ estimate is well-posed (τ → 1).
    tau_power: float = 0.0

    def __post_init__(self) -> None:
        if (self.bodyparts is None) == (self.dofs is None):
            raise ValueError(f"{self.name}: exactly one of `bodyparts` or `dofs` must be set")
        if self.bodyparts is not None:
            self.dofs = tuple(d for bp in self.bodyparts for d in bp.dof_ids)

    def tau_scale(self, tau: ArrayLike) -> Array:
        """Per-sample τ-gate factor, broadcastable like ``tau``."""
        if self.tau_power == 0.0:
            return jnp.ones_like(jnp.asarray(tau))
        return jnp.asarray(tau) ** self.tau_power

    def __call__(self, *args: object, **kwargs: object) -> tuple[Array, dict[str, Array]]:
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

    def __call__(self, *args: object, **kwargs: object) -> tuple[Array, dict[str, Array]]:
        kwargs.setdefault("fk_fn", self.fk_fn)
        return super().__call__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Smoothness — finite-difference penalty along the action chunk
# ---------------------------------------------------------------------------


def smoothness_loss(
    x1: ArrayLike,
    mask: ArrayLike,
    order: int = 2,
    axis: int = -2,
) -> tuple[Array, dict[str, Array]]:
    """Mean squared finite difference of ``order`` along the chunk (H) axis.

    Works in normalized space (per-dim scaling doesn't change the structure).
    order=2 penalizes acceleration, order=3 jerk. The mask shrinks with each
    difference so chunk-boundary/invalid steps never contribute.

    Args:
        x1: denoised action estimate (..., H, A).
        mask: bool, broadcastable to x1 (invalid slots/steps False).
        order: number of finite differences.
        axis: the chunk axis.
    """
    d = jnp.asarray(x1)
    m = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), d.shape)
    for _ in range(order):
        d = jnp.diff(d, axis=axis)
        lo = [slice(None)] * d.ndim
        hi = [slice(None)] * d.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        m = m[tuple(lo)] & m[tuple(hi)]
    loss = masked_mean(jnp.square(d), m)
    return loss, {"loss": loss}


@dataclass
class SmoothnessTerm(LossTerm):
    """Accel/jerk penalty on the denoised chunk. ``dofs=()`` = all slots."""

    fn: LossFn = smoothness_loss
    dofs: tuple[int, ...] | None = ()
    order: int = 2

    def __call__(self, *args: object, **kwargs: object) -> tuple[Array, dict[str, Array]]:
        kwargs.setdefault("order", self.order)
        return super().__call__(*args, **kwargs)


# ---------------------------------------------------------------------------
# FK consistency — FK(denoised joints) must match denoised per-view kp3dc
# ---------------------------------------------------------------------------


def fk_consistency_loss(
    joints_pred: ArrayLike,
    kp3dc_pred: ArrayLike,
    w2c: ArrayLike,
    mask: ArrayLike,
    fk_fn: Callable[[Array], Array],
) -> tuple[Array, dict[str, Array]]:
    """MSE between FK of denoised joints and denoised camera-frame keypoints.

    Args:
        joints_pred: (..., 7) denoised joint angles, raw units (radians).
        kp3dc_pred: (..., V, N, 3) denoised camera-frame keypoints, raw units.
        w2c: (..., V, 4, 4) world→camera extrinsics per view.
        mask: bool, broadcastable to (..., V, N) — invalid views/keypoints False.
        fk_fn: joints (..., 7) → world keypoints (..., N, 3), dream order.
    """
    kp3dw = fk_fn(jnp.asarray(joints_pred))  # (..., N, 3)
    w2c = jnp.asarray(w2c)
    ones = jnp.ones((*kp3dw.shape[:-1], 1), dtype=kp3dw.dtype)
    kp_h = jnp.concatenate([kp3dw, ones], axis=-1)  # (..., N, 4)
    kp3dc_fk = jnp.einsum("...vij,...nj->...vni", w2c, kp_h)[..., :3]
    err = jnp.square(kp3dc_fk - jnp.asarray(kp3dc_pred))  # (..., V, N, 3)
    m = jnp.broadcast_to(jnp.asarray(mask, dtype=bool)[..., None], err.shape)
    loss = masked_mean(err, m)
    return loss, {"loss": loss, "rmse_m": jnp.sqrt(loss + 1e-12)}


@dataclass
class FKConsistencyTerm(LossTerm):
    """FK(joints) ↔ kp3dc coherence across views. ``fk_fn`` is embodiment FK."""

    fn: LossFn = fk_consistency_loss
    bodyparts: tuple[BodyPart, ...] | None = (ARM_7DOF, KP3DC)
    fk_fn: Callable[[Array], Array] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fk_fn is None:
            raise ValueError(f"{self.name}: `fk_fn` is required (embodiment-specific FK)")

    def __call__(self, *args: object, **kwargs: object) -> tuple[Array, dict[str, Array]]:
        kwargs.setdefault("fk_fn", self.fk_fn)
        return super().__call__(*args, **kwargs)


# ---------------------------------------------------------------------------
# YAML registry — config/loss.yaml
# ---------------------------------------------------------------------------


def load_loss_terms(
    path: str | Path,
    fk_fn: Callable[[Array], Array] | None = None,
) -> dict[str, LossTerm]:
    """Instantiate LossTerms from a yaml of ``_target_`` specs.

    ``fk_fn`` is injected into terms whose dataclass declares an ``fk_fn``
    field (FK-dependent terms are skipped with a clear error if it's missing).
    """
    specs = ModuleFile.load(Path(path))
    terms: dict[str, LossTerm] = {}
    for key, spec in specs.items():
        make = ModuleSpec.instantiate(spec)
        target = make.func if hasattr(make, "func") else make
        needs_fk = any(f.name == "fk_fn" for f in fields(target))
        term = make(fk_fn=fk_fn) if needs_fk and fk_fn is not None else make()
        assert isinstance(term, LossTerm), f"{key}: {target} is not a LossTerm"
        terms[key] = term
    return terms
