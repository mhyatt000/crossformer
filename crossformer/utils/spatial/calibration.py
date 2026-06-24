from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StackedPnPResult:
    w2c: np.ndarray | None
    valid: np.ndarray
    reproj_px: float
    n_points: int

    @property
    def success(self) -> bool:
        return self.w2c is not None


def project_points(w2c: np.ndarray, pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts_cam = (w2c[:3, :3] @ pts_3d.T).T + w2c[:3, 3]
    pix = (K @ pts_cam.T).T
    return pix[:, :2] / np.maximum(pix[:, 2:3], 1e-8)


def solve_stacked_pnp(
    pts_3d: np.ndarray,
    uv_px: np.ndarray,
    K: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    min_points: int = 12,
    reproj_thresh_px: float = 30.0,
) -> StackedPnPResult:
    """Solve one camera pose from keypoints stacked across robot poses."""
    pts_3d = np.asarray(pts_3d, dtype=np.float64)
    uv_px = np.asarray(uv_px, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if pts_3d.shape[:-1] != uv_px.shape[:-1] or pts_3d.shape[-1] != 3 or uv_px.shape[-1] != 2:
        raise ValueError(f"expected pts_3d (...,3) and uv_px (...,2), got {pts_3d.shape=} {uv_px.shape=}")
    if K.shape != (3, 3):
        raise ValueError(f"expected K shape (3,3), got {K.shape}")

    finite = np.isfinite(pts_3d).all(axis=-1) & np.isfinite(uv_px).all(axis=-1) & (uv_px[..., 0] > -999.0)
    valid = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    flat_valid = valid.reshape(-1)
    pts = pts_3d.reshape(-1, 3)[flat_valid]
    uv = uv_px.reshape(-1, 2)[flat_valid]
    if pts.shape[0] < min_points:
        return StackedPnPResult(None, valid, float("inf"), int(pts.shape[0]))

    w2c = _solve_pnp_sqpnp_iter(pts, uv, K)
    if w2c is None:
        return StackedPnPResult(None, valid, float("inf"), int(pts.shape[0]))

    errs = np.linalg.norm(project_points(w2c, pts, K) - uv, axis=-1)
    keep = errs <= reproj_thresh_px
    if not keep.all():
        if int(keep.sum()) < min_points:
            valid_refined = np.zeros_like(flat_valid)
            valid_refined[np.where(flat_valid)[0][keep]] = True
            return StackedPnPResult(None, valid_refined.reshape(valid.shape), float("inf"), int(keep.sum()))
        w2c_refined = _solve_pnp_sqpnp_iter(pts[keep], uv[keep], K)
        if w2c_refined is None:
            return StackedPnPResult(None, valid, float("inf"), int(keep.sum()))
        w2c = w2c_refined
        valid_refined = np.zeros_like(flat_valid)
        valid_refined[np.where(flat_valid)[0][keep]] = True
        valid = valid_refined.reshape(valid.shape)
        pts = pts[keep]
        uv = uv[keep]

    reproj = float(np.linalg.norm(project_points(w2c, pts, K) - uv, axis=-1).mean())
    if reproj > reproj_thresh_px:
        return StackedPnPResult(None, valid, reproj, int(pts.shape[0]))
    return StackedPnPResult(w2c, valid, reproj, int(pts.shape[0]))


def _solve_pnp_sqpnp_iter(pts_3d: np.ndarray, uv_px: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    if pts_3d.shape[0] < 4:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts_3d.astype(np.float64),
            uv_px.astype(np.float64),
            K.astype(np.float64),
            np.array([]),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if ok:
            ok, rvec, tvec = cv2.solvePnP(
                pts_3d.astype(np.float64),
                uv_px.astype(np.float64),
                K.astype(np.float64),
                np.array([]),
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    return w2c
