from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from crossformer.data.geometry import denormalize_kp2d
from crossformer.utils.callbacks.synth_viz import composite_robot, fk_keypoints, rasterize_robot

from .metrics import _mask_iou, _reprojection_errors, extract_keypoints

KP_MISSING_VALUE = -999.999


@dataclass
class SessionCalibrationConfig:
    """Session-level per-camera extrinsic calibration."""

    enabled: bool = False
    camera_keys: tuple[str, ...] | None = None
    max_candidate_frames_per_camera: int = 512
    max_selected_frames_per_camera: int = 64
    min_selected_frames: int = 2
    min_total_correspondences: int = 12
    min_total_inliers: int = 12
    min_frame_inliers: int = 4
    keypoint_conf_threshold: float = 0.03
    reproj_threshold_px: float = 30.0
    frame_reproj_threshold_px: float = 30.0
    ransac_iterations: int = 1000
    ransac_confidence: float = 0.999
    max_refine_iters: int = 4
    require_same_K: bool = True
    use_mask_iou_scoring: bool = True
    min_mask_iou: float | None = None
    diversity_weight_q: float = 1.0
    diversity_weight_uv: float = 0.5
    confidence_weight: float = 1.0
    q_degrees: bool = True


MultiFramePnPConfig = SessionCalibrationConfig


@dataclass
class FramePrediction:
    cam_name: str
    frame_idx: int
    image: np.ndarray | None
    q: np.ndarray
    K: np.ndarray
    keypoints_px: np.ndarray
    keypoints_conf: np.ndarray
    pred_mask: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiFrameCorrespondences:
    pts_3d: np.ndarray
    uv_px: np.ndarray
    conf: np.ndarray
    frame_idx: np.ndarray
    kp_idx: np.ndarray
    K: np.ndarray
    cam_name: str


@dataclass
class CameraCalibrationResult:
    cam_name: str
    success: bool
    failure_reason: str | None
    w2c: np.ndarray | None
    K: np.ndarray | None
    num_candidate_frames: int
    num_selected_frames: int
    num_used_frames: int
    num_candidate_points: int
    num_inlier_points: int
    mean_reproj_px: float
    median_reproj_px: float
    per_frame_reproj_px: dict[int, float] = field(default_factory=dict)
    per_frame_inlier_counts: dict[int, int] = field(default_factory=dict)
    used_frame_indices: list[int] = field(default_factory=list)
    rejected_frame_indices: list[int] = field(default_factory=list)
    mean_mask_iou: float = float("nan")
    per_frame_mask_iou: dict[int, float] = field(default_factory=dict)
    rasterized_masks: dict[int, np.ndarray] = field(default_factory=dict)
    overlays: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class SessionCalibrationResult:
    session_id: str | None
    camera_results: dict[str, CameraCalibrationResult]
    summary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustStackedPnPResult:
    success: bool
    failure_reason: str | None
    w2c: np.ndarray | None
    inliers: np.ndarray
    reproj_errors: np.ndarray


def calibrate_session_cameras(session: Any, cfg: SessionCalibrationConfig) -> SessionCalibrationResult:
    if not cfg.enabled:
        return SessionCalibrationResult(_session_id(session), {}, {"session_calib/enabled": 0.0})

    camera_results = {}
    for cam in get_camera_keys(session, cfg):
        frames = decode_camera_predictions_for_session(session, cam, cfg)
        selected = select_diverse_reliable_frames(frames, cfg)
        camera_results[cam] = solve_camera_extrinsics_from_frames(cam, selected, cfg, num_candidate_frames=len(frames))

    result = SessionCalibrationResult(_session_id(session), camera_results)
    result.summary = summarize_session_calibration(result)
    return result


def get_camera_keys(session: Any, cfg: SessionCalibrationConfig) -> list[str]:
    if cfg.camera_keys is not None:
        return list(cfg.camera_keys)
    session_d = _as_dict(session)
    keys = []
    for key, val in session_d.items():
        if key.startswith("image[") and key.endswith("]"):
            keys.append(key[6:-1])
        elif key == "image" and isinstance(val, dict):
            keys.extend(str(k) for k in val)
    return sorted(dict.fromkeys(keys))


def decode_camera_predictions_for_session(
    session: Any, cam_name: str, cfg: SessionCalibrationConfig
) -> list[FramePrediction]:
    session_d = _as_dict(session)
    image = _camera_value(session_d, "image", cam_name, default=None)
    q = _value(session_d, ("q", "joint_positions", "proprio"))
    K = _camera_value(session_d, "K", cam_name, default=None)
    uv = _camera_value(session_d, "keypoints_px", cam_name, default=None)
    conf = _camera_value(session_d, "keypoints_conf", cam_name, default=None)
    if uv is None or conf is None:
        uv, conf = _decode_heatmaps(session_d, cam_name, image)
    pred_mask = _camera_value(session_d, "pred_mask", cam_name, default=None)
    gt_mask = _camera_value(session_d, "gt_mask", cam_name, default=None)
    if gt_mask is None:
        gt_mask = _camera_value(session_d, "mask", cam_name, default=None)
    timestamp = _value(session_d, ("timestamp", "time"), default=None)

    if q is None:
        raise ValueError("session is missing q/joint_positions/proprio")
    if K is None:
        raise ValueError(f"session is missing K for camera {cam_name}")
    if uv is None or conf is None:
        raise ValueError(f"session is missing keypoints or heatmaps for camera {cam_name}")

    q = np.asarray(q)
    K = np.asarray(K)
    uv = np.asarray(uv)
    conf = np.asarray(conf)
    n = min(_len_frames(q), _len_frames(uv), _len_frames(conf), _len_frames(K))
    if image is not None:
        n = min(n, _len_frames(np.asarray(image)))
    if pred_mask is not None:
        n = min(n, _len_frames(np.asarray(pred_mask)))
    if gt_mask is not None:
        n = min(n, _len_frames(np.asarray(gt_mask)))
    n = min(n, cfg.max_candidate_frames_per_camera)

    return [
        FramePrediction(
            cam_name=cam_name,
            frame_idx=i,
            timestamp=_frame_item(timestamp, i, default=None),
            image=_frame_item(image, i, default=None),
            q=np.asarray(_frame_item(q, i), dtype=np.float64),
            K=np.asarray(_frame_item(K, i), dtype=np.float64),
            keypoints_px=np.asarray(_frame_item(uv, i), dtype=np.float64),
            keypoints_conf=np.asarray(_frame_item(conf, i), dtype=np.float64),
            pred_mask=_frame_item(pred_mask, i, default=None),
            gt_mask=_frame_item(gt_mask, i, default=None),
        )
        for i in range(n)
    ]


def select_diverse_reliable_frames(
    frames: list[FramePrediction], cfg: SessionCalibrationConfig
) -> list[FramePrediction]:
    scored = [(f, _frame_score(f, cfg)) for f in frames]
    scored = [(f, score) for f, score in scored if score > 0.0]
    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [scored[0][0]]
    remaining = scored[1:]
    while remaining and len(selected) < cfg.max_selected_frames_per_camera:
        best_i = max(
            range(len(remaining)), key=lambda i: _selection_score(remaining[i][0], remaining[i][1], selected, cfg)
        )
        selected.append(remaining.pop(best_i)[0])
    return selected


def build_multiframe_correspondences(
    frames: list[FramePrediction], cfg: SessionCalibrationConfig
) -> MultiFrameCorrespondences:
    if not frames:
        raise ValueError("no selected frames")
    K0 = np.asarray(frames[0].K, dtype=np.float64)
    if K0.shape != (3, 3):
        raise ValueError(f"expected K shape (3,3), got {K0.shape}")
    pts_all, uv_all, conf_all, frame_all, kp_all = [], [], [], [], []
    for frame in frames:
        K = np.asarray(frame.K, dtype=np.float64)
        if cfg.require_same_K and not np.allclose(K, K0, rtol=1e-5, atol=1e-5):
            raise ValueError(f"incompatible K for camera {frame.cam_name} frame {frame.frame_idx}")
        q = np.asarray(frame.q, dtype=np.float64)
        joints = np.deg2rad(q[:7]) if cfg.q_degrees else q[:7]
        pts_3d = fk_keypoints(joints)
        uv = np.asarray(frame.keypoints_px, dtype=np.float64)
        conf = np.asarray(frame.keypoints_conf, dtype=np.float64)
        valid = _valid_keypoints(frame, cfg)
        idx = np.where(valid)[0]
        pts_all.append(pts_3d[idx])
        uv_all.append(uv[idx])
        conf_all.append(conf[idx])
        frame_all.append(np.full(idx.shape, frame.frame_idx, dtype=np.int64))
        kp_all.append(idx.astype(np.int64))
    if not pts_all:
        raise ValueError("no correspondences")
    return MultiFrameCorrespondences(
        pts_3d=np.concatenate(pts_all, axis=0),
        uv_px=np.concatenate(uv_all, axis=0),
        conf=np.concatenate(conf_all, axis=0),
        frame_idx=np.concatenate(frame_all, axis=0),
        kp_idx=np.concatenate(kp_all, axis=0),
        K=K0,
        cam_name=frames[0].cam_name,
    )


def solve_camera_extrinsics_from_frames(
    cam_name: str,
    frames: list[FramePrediction],
    cfg: SessionCalibrationConfig,
    *,
    num_candidate_frames: int | None = None,
) -> CameraCalibrationResult:
    n_candidates = len(frames) if num_candidate_frames is None else num_candidate_frames
    result = _failed_result(cam_name, n_candidates, len(frames), "not_run")
    if len(frames) < cfg.min_selected_frames:
        result.failure_reason = "insufficient_frames"
        return result
    try:
        corr = build_multiframe_correspondences(frames, cfg)
    except ValueError as exc:
        result.failure_reason = str(exc)
        return result

    result.num_candidate_points = int(corr.pts_3d.shape[0])
    if corr.pts_3d.shape[0] < cfg.min_total_correspondences:
        result.failure_reason = "insufficient_points"
        return result

    pnp = solve_robust_stacked_pnp(corr, cfg)
    result.K = corr.K
    result.num_inlier_points = int(pnp.inliers.sum())
    if not pnp.success:
        result.failure_reason = pnp.failure_reason
        return result

    diagnostics = score_camera_calibration(frames, pnp.w2c, cfg, pnp.inliers, corr)
    used = list(diagnostics["used_frame_indices"])
    rejected = [f.frame_idx for f in frames if f.frame_idx not in used]
    return CameraCalibrationResult(
        cam_name=cam_name,
        success=True,
        failure_reason=None,
        w2c=pnp.w2c,
        K=corr.K,
        num_candidate_frames=n_candidates,
        num_selected_frames=len(frames),
        num_used_frames=len(used),
        num_candidate_points=int(corr.pts_3d.shape[0]),
        num_inlier_points=int(pnp.inliers.sum()),
        mean_reproj_px=float(np.mean(pnp.reproj_errors[pnp.inliers])),
        median_reproj_px=float(np.median(pnp.reproj_errors[pnp.inliers])),
        per_frame_reproj_px=diagnostics["per_frame_reproj_px"],
        per_frame_inlier_counts=diagnostics["per_frame_inlier_counts"],
        used_frame_indices=used,
        rejected_frame_indices=rejected,
        mean_mask_iou=diagnostics["mean_mask_iou"],
        per_frame_mask_iou=diagnostics["per_frame_mask_iou"],
        rasterized_masks=diagnostics["rasterized_masks"],
        overlays=diagnostics["overlays"],
    )


def solve_robust_stacked_pnp(
    correspondences: MultiFrameCorrespondences, cfg: SessionCalibrationConfig
) -> RobustStackedPnPResult:
    pts = np.asarray(correspondences.pts_3d, dtype=np.float64)
    uv = np.asarray(correspondences.uv_px, dtype=np.float64)
    K = np.asarray(correspondences.K, dtype=np.float64)
    if pts.shape[0] < cfg.min_total_correspondences:
        return _robust_fail("insufficient_points", pts.shape[0])

    ransac = _solve_pnp_ransac(pts, uv, K, cfg)
    if ransac is None:
        return _robust_fail("failed_pnp", pts.shape[0])
    rvec, tvec, inliers = ransac
    if int(inliers.sum()) < cfg.min_total_inliers:
        return _robust_fail("insufficient_inliers", pts.shape[0], inliers)

    prev = inliers.copy()
    for _ in range(max(cfg.max_refine_iters, 1)):
        refined = _refine_pnp(pts[inliers], uv[inliers], K, rvec, tvec)
        if refined is None:
            return _robust_fail("failed_pnp_refine", pts.shape[0], inliers)
        rvec, tvec = refined
        w2c = _w2c_from_rvec_tvec(rvec, tvec)
        errs = _reprojection_errors(w2c, pts, uv, K)
        keep = errs <= cfg.reproj_threshold_px
        keep = _drop_bad_frames(keep, errs, correspondences, cfg)
        if int(keep.sum()) < cfg.min_total_inliers:
            return RobustStackedPnPResult(False, "insufficient_inliers", None, keep, errs)
        if np.array_equal(keep, prev):
            inliers = keep
            break
        prev = keep.copy()
        inliers = keep

    w2c = _w2c_from_rvec_tvec(rvec, tvec)
    errs = _reprojection_errors(w2c, pts, uv, K)
    if float(np.mean(errs[inliers])) > cfg.reproj_threshold_px:
        return RobustStackedPnPResult(False, "excessive_reprojection_error", None, inliers, errs)
    return RobustStackedPnPResult(True, None, w2c, inliers, errs)


def _solve_pnp_ransac(
    pts: np.ndarray, uv: np.ndarray, K: np.ndarray, cfg: SessionCalibrationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        ok, rvec, tvec, inliers_cv = cv2.solvePnPRansac(
            pts,
            uv,
            K,
            np.array([]),
            iterationsCount=cfg.ransac_iterations,
            reprojectionError=cfg.reproj_threshold_px,
            confidence=cfg.ransac_confidence,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None
    if not ok or inliers_cv is None:
        return None
    inliers = np.zeros(pts.shape[0], dtype=bool)
    inliers[inliers_cv.reshape(-1)] = True
    return rvec, tvec, inliers


def _refine_pnp(
    pts: np.ndarray, uv: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts,
            uv,
            K,
            np.array([]),
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec,
            tvec=tvec,
        )
    except cv2.error:
        return None
    return (rvec, tvec) if ok else None


def score_camera_calibration(
    frames: list[FramePrediction],
    w2c: np.ndarray,
    cfg: SessionCalibrationConfig,
    inliers: np.ndarray | None = None,
    correspondences: MultiFrameCorrespondences | None = None,
) -> dict[str, Any]:
    per_frame_reproj, per_frame_inliers = {}, {}
    per_frame_iou, rasterized, overlays = {}, {}, {}
    used = []

    if correspondences is not None and inliers is not None:
        errs = _reprojection_errors(w2c, correspondences.pts_3d, correspondences.uv_px, correspondences.K)
        for frame in frames:
            mask = (correspondences.frame_idx == frame.frame_idx) & inliers
            per_frame_inliers[frame.frame_idx] = int(mask.sum())
            per_frame_reproj[frame.frame_idx] = float(np.mean(errs[mask])) if mask.any() else float("inf")
            if (
                mask.sum() >= cfg.min_frame_inliers
                and per_frame_reproj[frame.frame_idx] <= cfg.frame_reproj_threshold_px
            ):
                used.append(frame.frame_idx)

    for frame in frames:
        ref_mask = frame.gt_mask if frame.gt_mask is not None else frame.pred_mask
        if ref_mask is None:
            continue
        try:
            q = np.asarray(frame.q, dtype=np.float64)
            joints = np.deg2rad(q[:7]) if cfg.q_degrees else q[:7]
            gripper = float(q[7]) if q.shape[-1] > 7 else None
            h, w = np.asarray(ref_mask).shape[-2:]
            rast = rasterize_robot(joints, w2c, frame.K, w, h, gripper_rad=gripper)
            iou = _mask_iou(rast, np.asarray(ref_mask))
            per_frame_iou[frame.frame_idx] = iou
            rasterized[frame.frame_idx] = rast
            if frame.image is not None:
                overlays[frame.frame_idx] = composite_robot(frame.image, rast)
        except Exception:
            continue

    ious = np.asarray([v for v in per_frame_iou.values() if np.isfinite(v)], dtype=np.float64)
    mean_iou = float(ious.mean()) if len(ious) else float("nan")
    if cfg.use_mask_iou_scoring and cfg.min_mask_iou is not None:
        used = [idx for idx in used if per_frame_iou.get(idx, cfg.min_mask_iou) >= cfg.min_mask_iou]
    return {
        "used_frame_indices": used,
        "per_frame_reproj_px": per_frame_reproj,
        "per_frame_inlier_counts": per_frame_inliers,
        "mean_mask_iou": mean_iou,
        "per_frame_mask_iou": per_frame_iou,
        "rasterized_masks": rasterized,
        "overlays": overlays,
    }


def summarize_session_calibration(result: SessionCalibrationResult, prefix: str = "session_calib") -> dict[str, float]:
    rows = list(result.camera_results.values())
    ok = [r for r in rows if r.success]
    summary = {
        f"{prefix}/num_cameras": float(len(rows)),
        f"{prefix}/num_success": float(len(ok)),
        f"{prefix}/success_rate": float(len(ok) / len(rows)) if rows else float("nan"),
    }
    for key, attr in (
        ("mean_reproj_px", "mean_reproj_px"),
        ("median_reproj_px", "median_reproj_px"),
        ("inlier_points", "num_inlier_points"),
        ("used_frames", "num_used_frames"),
        ("mask_iou", "mean_mask_iou"),
    ):
        xs = np.asarray([getattr(r, attr) for r in ok], dtype=np.float64)
        summary[f"{prefix}/{key}"] = float(np.nanmean(xs)) if len(xs) else float("nan")
    for cam, res in result.camera_results.items():
        cam_prefix = f"camera_calib/{cam}"
        summary[f"{cam_prefix}/success"] = float(res.success)
        summary[f"{cam_prefix}/mean_reproj_px"] = float(res.mean_reproj_px)
        summary[f"{cam_prefix}/num_inlier_points"] = float(res.num_inlier_points)
        summary[f"{cam_prefix}/num_used_frames"] = float(res.num_used_frames)
        summary[f"{cam_prefix}/mean_mask_iou"] = float(res.mean_mask_iou)
    return summary


def _valid_keypoints(frame: FramePrediction, cfg: SessionCalibrationConfig) -> np.ndarray:
    uv = np.asarray(frame.keypoints_px, dtype=np.float64)
    conf = np.asarray(frame.keypoints_conf, dtype=np.float64)
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(conf) & (conf >= cfg.keypoint_conf_threshold)
    valid &= uv[:, 0] > KP_MISSING_VALUE * 0.5
    if frame.image is not None:
        h, w = np.asarray(frame.image).shape[:2]
        valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] < w) & (uv[:, 1] >= 0.0) & (uv[:, 1] < h)
    return valid


def _frame_score(frame: FramePrediction, cfg: SessionCalibrationConfig) -> float:
    valid = _valid_keypoints(frame, cfg)
    n = int(valid.sum())
    if n < cfg.min_frame_inliers:
        return 0.0
    conf = np.asarray(frame.keypoints_conf, dtype=np.float64)
    score = float(n) + cfg.confidence_weight * float(np.mean(conf[valid]))
    mask = frame.gt_mask if frame.gt_mask is not None else frame.pred_mask
    if mask is not None:
        area = float(np.mean(np.asarray(mask) > 0.5))
        if area <= 0.0005 or area >= 0.95:
            return 0.0
        score += min(area * 100.0, 5.0)
    return score


def _selection_score(
    frame: FramePrediction, reliability: float, selected: list[FramePrediction], cfg: SessionCalibrationConfig
) -> float:
    q = np.asarray(frame.q[:7], dtype=np.float64)
    uv = np.asarray(frame.keypoints_px, dtype=np.float64)
    valid = _valid_keypoints(frame, cfg)
    q_div = min(np.linalg.norm(q - np.asarray(s.q[:7], dtype=np.float64)) for s in selected)
    uv_divs = []
    for s in selected:
        s_valid = _valid_keypoints(s, cfg)
        both = valid & s_valid
        if both.any():
            uv_divs.append(float(np.linalg.norm(uv[both] - np.asarray(s.keypoints_px)[both], axis=-1).mean()))
    uv_div = min(uv_divs) if uv_divs else 0.0
    return reliability + cfg.diversity_weight_q * q_div + cfg.diversity_weight_uv * uv_div


def _drop_bad_frames(
    keep: np.ndarray, errs: np.ndarray, corr: MultiFrameCorrespondences, cfg: SessionCalibrationConfig
) -> np.ndarray:
    out = keep.copy()
    for frame_idx in np.unique(corr.frame_idx):
        mask = (corr.frame_idx == frame_idx) & out
        if mask.sum() == 0:
            continue
        if mask.sum() < cfg.min_frame_inliers or float(np.mean(errs[mask])) > cfg.frame_reproj_threshold_px:
            out[corr.frame_idx == frame_idx] = False
    return out


def _decode_heatmaps(
    session_d: dict[str, Any], cam_name: str, image: Any
) -> tuple[np.ndarray | None, np.ndarray | None]:
    heatmaps = _camera_value(session_d, "pred_heatmaps", cam_name, default=None)
    if heatmaps is None:
        heatmaps = _camera_value(session_d, "heatmaps", cam_name, default=None)
    if heatmaps is None:
        return None, None
    hm = jnp.asarray(heatmaps)
    uv, conf = extract_keypoints(hm)
    _, _, out_h, out_w = hm.shape
    if image is None:
        return np.asarray(jax.device_get(uv), dtype=np.float64), np.asarray(jax.device_get(conf), dtype=np.float64)
    image_shape = np.asarray(image).shape
    h, w = image_shape[1:3] if len(image_shape) >= 4 else image_shape[:2]
    uv_px = denormalize_kp2d(uv / jnp.array([out_w, out_h], dtype=jnp.float32), h, w)
    return np.asarray(jax.device_get(uv_px), dtype=np.float64), np.asarray(jax.device_get(conf), dtype=np.float64)


def _failed_result(
    cam_name: str, num_candidate_frames: int, num_selected_frames: int, reason: str
) -> CameraCalibrationResult:
    return CameraCalibrationResult(
        cam_name=cam_name,
        success=False,
        failure_reason=reason,
        w2c=None,
        K=None,
        num_candidate_frames=num_candidate_frames,
        num_selected_frames=num_selected_frames,
        num_used_frames=0,
        num_candidate_points=0,
        num_inlier_points=0,
        mean_reproj_px=float("inf"),
        median_reproj_px=float("inf"),
    )


def _robust_fail(reason: str, n: int, inliers: np.ndarray | None = None) -> RobustStackedPnPResult:
    inliers = np.zeros(n, dtype=bool) if inliers is None else inliers
    return RobustStackedPnPResult(False, reason, None, inliers, np.full(n, np.inf, dtype=np.float64))


def _w2c_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.reshape(3)
    return w2c


def _as_dict(session: Any) -> dict[str, Any]:
    return session if isinstance(session, dict) else vars(session)


def _session_id(session: Any) -> str | None:
    session_d = _as_dict(session)
    val = session_d.get("session_id", session_d.get("id"))
    return None if val is None else str(val)


def _value(session_d: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in session_d:
            return session_d[key]
    return default


def _camera_value(session_d: dict[str, Any], base: str, cam: str, default: Any = None) -> Any:
    bracket = f"{base}[{cam}]"
    if bracket in session_d:
        return session_d[bracket]
    val = session_d.get(base, default)
    if isinstance(val, dict):
        return val.get(cam, default)
    return val


def _len_frames(x: np.ndarray) -> int:
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape == (3, 3):
        return 10**9
    return int(arr.shape[0])


def _frame_item(x: Any, i: int, default: Any = ...):
    if x is None:
        if default is ...:
            raise IndexError(i)
        return default
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr
    if arr.ndim == 0:
        return arr.item()
    return arr[i]
