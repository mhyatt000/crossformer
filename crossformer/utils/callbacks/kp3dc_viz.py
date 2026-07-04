"""Eval callback: project kp3dc_robot keypoints onto each camera view.

``kp3dc_robot`` slots are camera-frame 3D keypoints (one 42-dim copy per view,
disambiguated by ``act.view``; view id v+1 <-> index v of the ``state`` view
axis). Projection is therefore intrinsics-only: ``uv = (K @ xyz) / z``, drawn
on the view's image resized to ``(2*cy, 2*cx)`` so K applies in pixel units.

Every horizon step is drawn (GT green, prediction red->yellow with a thin
trajectory trace per keypoint) so the predicted future motion is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from crossformer.embody import DOF, KP3DC, KP_CHAIN
from crossformer.utils.callbacks.base import EvalContext, maybe_getpath
import wandb

KP3DC_IDS = tuple(DOF[name] for name in KP3DC.dof_names)  # (42,)
_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(KP3DC_IDS)}
N_KP = len(KP_CHAIN)


def project_kp3dc(xyz: np.ndarray, K: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project camera-frame points (..., 3) to pixel uv (..., 2); NaN when z <= eps."""
    xyz = np.asarray(xyz, dtype=np.float32)
    pix = np.einsum("ij,...j->...i", np.asarray(K, dtype=np.float32), xyz)
    z = pix[..., 2:3]
    uv = pix[..., :2] / np.maximum(z, eps)
    return np.where(z > eps, uv, np.nan)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.asarray(img, dtype=np.float32)
        if img.size and float(img.max()) <= 1.001:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


@dataclass
class Kp3dcVizCallback:
    """Overlay GT + predicted kp3dc keypoints per view, all horizon steps."""

    name: str = "kp3dc"
    every: int = 0
    sample_idx: int = 0
    image_key: str = "image"  # stacked (B, W, V, h, w, C) observation key
    radius: int = 3
    max_px: int = 4096  # sanity cap on the K-derived render size

    def __call__(self, ctx: EvalContext) -> dict[str, Any]:
        batch = ctx.batch
        K_all = maybe_getpath(batch, ("state", "intr", "K"))
        images = maybe_getpath(batch, ("observation", self.image_key))
        if K_all is None or images is None:
            return {}

        base = np.asarray(batch["act"]["base"], dtype=np.float32)
        if base.ndim == 3:
            base = base[:, None]
        pred = np.asarray(ctx.pred, dtype=np.float32)
        b = min(self.sample_idx, base.shape[0] - 1)

        dof_ids = np.asarray(batch["act"]["id"])[b]  # (A,)
        view_ids = np.asarray(batch["act"].get("view", np.zeros_like(dof_ids)))[b]
        mask_act = batch.get("mask", {}).get("act")
        slot_ok = np.ones(dof_ids.shape[0], dtype=bool) if mask_act is None else np.asarray(mask_act)[b].astype(bool)

        K_views = np.asarray(K_all)[b, -1]  # (V, 3, 3) — last window step
        imgs = np.asarray(images)[b, -1]  # (V, h, w, C)
        ds_name = ctx.ds_names[b]

        out: dict[str, Any] = {}
        for v in range(K_views.shape[0]):
            gt = self._gather_kp(base[b, -1], dof_ids, view_ids, slot_ok, view_id=v + 1)  # (H, 42) or None
            pd = self._gather_kp(pred[b, -1], dof_ids, view_ids, slot_ok, view_id=v + 1)
            if gt is None or pd is None:
                continue
            gt = self._denorm(ctx, gt, ds_name).reshape(-1, N_KP, 3)
            pd = self._denorm(ctx, pd, ds_name).reshape(-1, N_KP, 3)

            K = K_views[v]
            cx, cy = float(K[0, 2]), float(K[1, 2])
            w_px, h_px = round(2 * cx), round(2 * cy)
            if not (8 <= w_px <= self.max_px and 8 <= h_px <= self.max_px):
                continue
            img = cv2.resize(_to_uint8(imgs[v]), (w_px, h_px))
            self._draw(img, project_kp3dc(gt, K), project_kp3dc(pd, K))
            out[f"view_{v}"] = wandb.Image(img, caption=f"sample {b} view {v} | GT=green, pred=red->yellow over horizon")
        return out

    def _gather_kp(
        self,
        arr: np.ndarray,
        dof_ids: np.ndarray,
        view_ids: np.ndarray,
        slot_ok: np.ndarray,
        view_id: int,
    ) -> np.ndarray | None:
        """Collect this view's kp3dc slots from (H, A) into canonical (H, 42); NaN-fill gaps."""
        out = np.full((arr.shape[0], len(KP3DC_IDS)), np.nan, dtype=np.float32)
        found = 0
        for slot, (dof_id, view) in enumerate(zip(dof_ids, view_ids)):
            dst = _ID_TO_IDX.get(int(dof_id))
            if dst is None or int(view) != view_id or not bool(slot_ok[slot]):
                continue
            out[:, dst] = arr[:, slot]
            found += 1
        return out if found >= 3 else None  # need at least one full xyz keypoint

    def _denorm(self, ctx: EvalContext, arr: np.ndarray, ds_name: str) -> np.ndarray:
        ids = np.asarray(KP3DC_IDS)
        return np.stack([ctx.denorm.denormalize_slot(row, ids, ds_name) for row in arr], axis=0)

    def _draw(self, img: np.ndarray, gt_uv: np.ndarray, pd_uv: np.ndarray) -> None:
        """Draw (H, K, 2) GT and pred pixel tracks in place, colored by horizon step."""
        horizon = gt_uv.shape[0]

        def _pt(uv: np.ndarray) -> tuple[int, int] | None:
            if not np.all(np.isfinite(uv)):
                return None
            return int(round(float(uv[0]))), int(round(float(uv[1])))

        # per-keypoint prediction trace so future motion reads as a path
        for k in range(pd_uv.shape[1]):
            for h in range(horizon - 1):
                p0, p1 = _pt(pd_uv[h, k]), _pt(pd_uv[h + 1, k])
                if p0 is None or p1 is None:
                    continue
                t = h / max(horizon - 1, 1)
                cv2.line(img, p0, p1, (255, int(255 * t), 0), 1)

        for h in range(horizon):
            t = h / max(horizon - 1, 1)
            for k in range(gt_uv.shape[1]):
                p = _pt(gt_uv[h, k])
                if p is not None:
                    cv2.circle(img, p, self.radius, (0, int(120 + 135 * t), 0), 1)
                p = _pt(pd_uv[h, k])
                if p is not None:
                    cv2.circle(img, p, max(self.radius - 1, 1), (255, int(255 * t), 0), -1)
