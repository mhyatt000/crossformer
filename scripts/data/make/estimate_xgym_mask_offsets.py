"""Estimate per-step mask offsets for real xgym records.

Scores candidate offsets k in [-K, K] by boundary-edge agreement:
    score(i, k) = mean Sobel(image[i]) on boundary(mask[i+k])

Then smooths the per-step argmax offsets with a local mode filter to produce
piecewise-stable offsets plus confidence, and writes a JSON map keyed by
global step index for prerender consumption.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record


def _open_writers(name: str, version: str, branch: str, root: Path) -> ArrayRecordBuilder:
    builder = ArrayRecordBuilder(name=name, version=version, branch=branch, root=str(root))
    meta = json.loads((builder.root / "meta.json").read_text())
    builder.writers = builder._normalize_writers(meta["writers"])
    builder.default_writer = "data" if "data" in builder.writers else next(iter(builder.writers))
    return builder


def _sobel_mag(img_u8: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


def _score_one(edge_mag: np.ndarray, mask: np.ndarray) -> float:
    boundary = _mask_boundary(mask)
    if not boundary.any():
        return float("nan")
    return float(edge_mag[boundary].mean())


def _extract_episode(info: dict) -> int | None:
    info_id = info.get("id") if isinstance(info, dict) else None
    if isinstance(info_id, dict) and "episode" in info_id:
        return int(np.asarray(info_id["episode"]).reshape(-1)[0])
    if info_id is not None:
        try:
            return int(np.asarray(info_id).reshape(-1)[0])
        except Exception:
            return None
    return None


def _mode_smooth(offsets: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return offsets.copy()
    radius = window // 2
    out = np.empty_like(offsets)
    for i in range(len(offsets)):
        lo = max(0, i - radius)
        hi = min(len(offsets), i + radius + 1)
        win = offsets[lo:hi]
        counts = Counter(int(x) for x in win)
        best_n = max(counts.values())
        tied = [k for k, n in counts.items() if n == best_n]
        # tie-break: keep value closest to unsmoothed offset at i
        out[i] = min(tied, key=lambda k: abs(k - int(offsets[i])))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="xgym_sweep_single")
    p.add_argument("--version", default="0.6.0")
    p.add_argument("--branch", default="main")
    p.add_argument("--root", type=Path, default=Path.home() / ".cache/arrayrecords")
    p.add_argument("--cam", default="side")
    p.add_argument("--max-offset", type=int, default=5)
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--max-records", type=int, default=0, help="0 = all")
    p.add_argument("--out-json", type=Path, default=Path("/tmp/xgym_mask_offsets.json"))
    p.add_argument("--low-conf-thresh", type=float, default=0.05)
    args = p.parse_args()

    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("smooth-window must be a positive odd integer")
    if args.max_offset < 0:
        raise ValueError("max-offset must be >= 0")

    builder = _open_writers(args.name, args.version, args.branch, args.root.expanduser())
    img_src = builder.get_source("image")
    pro_src = builder.get_source("proprio")
    n = len(img_src)
    if args.max_records > 0:
        n = min(n, args.max_records)
    print(f"scanning {n} records from {builder.root} cam={args.cam} max_offset={args.max_offset}")

    by_ep: dict[int, list[int]] = defaultdict(list)
    missing_ep = 0
    for i in tqdm(range(n), desc="index episodes"):
        pro = unpack_record(pro_src[i])
        ep = _extract_episode(pro.get("info", {}))
        if ep is None:
            missing_ep += 1
            continue
        by_ep[ep].append(i)
    if missing_ep:
        print(f"warning: skipped {missing_ep} records with no episode id")
    print(f"episodes: {len(by_ep)}")

    offsets = np.arange(-args.max_offset, args.max_offset + 1, dtype=np.int32)
    idx_payload: dict[str, dict] = {}
    ep_payload: list[dict] = []
    low_conf = 0
    n_scored = 0

    for ep, idxs in tqdm(sorted(by_ep.items()), desc="estimate"):
        m = len(idxs)
        if m == 0:
            continue

        edges: list[np.ndarray | None] = [None] * m
        masks: list[np.ndarray | None] = [None] * m
        for t, gi in enumerate(idxs):
            img_rec = unpack_record(img_src[gi])
            image = img_rec.get("image", {}).get(args.cam)
            if image is not None:
                img = np.asarray(image)
                if img.ndim == 4:
                    img = img[0]
                edges[t] = _sobel_mag(img)
            mask = img_rec.get("mask", {}).get(args.cam)
            if mask is not None:
                msk = np.asarray(mask)
                if msk.ndim == 3:
                    msk = msk[0]
                masks[t] = msk

        score = np.full((m, len(offsets)), np.nan, dtype=np.float64)
        for t in range(m):
            if edges[t] is None:
                continue
            for c, k in enumerate(offsets):
                q = t + int(k)
                if q < 0 or q >= m or masks[q] is None:
                    continue
                score[t, c] = _score_one(edges[t], masks[q])

        valid_row = np.isfinite(score).any(axis=1)
        raw_choice = np.argmax(np.where(np.isnan(score), -np.inf, score), axis=1)
        raw_k = offsets[raw_choice].astype(np.int32)
        raw_k = np.where(valid_row, raw_k, 0).astype(np.int32)
        smooth_k = _mode_smooth(raw_k, args.smooth_window).astype(np.int32)

        best_raw = np.nanmax(score, axis=1)
        smooth_score = np.full((m,), np.nan, dtype=np.float64)
        conf = np.full((m,), 0.0, dtype=np.float32)
        for t in range(m):
            # confidence from top-2 gap at this frame
            row = score[t]
            valid = row[~np.isnan(row)]
            if valid.size >= 2:
                top2 = np.partition(valid, -2)[-2:]
                conf_t = float((top2[-1] - top2[-2]) / (abs(top2[-1]) + 1e-6))
            elif valid.size == 1:
                conf_t = 1.0
            else:
                conf_t = 0.0
            conf[t] = np.float32(np.clip(conf_t, 0.0, 1.0))

            c = int(np.where(offsets == smooth_k[t])[0][0])
            if np.isfinite(score[t, c]):
                smooth_score[t] = score[t, c]
            else:
                smooth_score[t] = best_raw[t]

        for t, gi in enumerate(idxs):
            k = int(smooth_k[t])
            entry = {
                "episode": int(ep),
                "episode_step": int(t),
                "offset": k,
                "score": float(smooth_score[t]) if np.isfinite(smooth_score[t]) else None,
                "confidence": float(conf[t]),
            }
            idx_payload[str(gi)] = entry
            if conf[t] < args.low_conf_thresh:
                low_conf += 1
            n_scored += 1

        raw_mean = float(np.nanmean(best_raw)) if np.isfinite(best_raw).any() else float("nan")
        smooth_mean = float(np.nanmean(smooth_score)) if np.isfinite(smooth_score).any() else float("nan")
        hist = Counter(int(x) for x in smooth_k.tolist())
        ep_payload.append(
            {
                "episode": int(ep),
                "n_steps": int(m),
                "raw_best_score_mean": raw_mean,
                "smoothed_score_mean": smooth_mean,
                "offset_histogram": dict(sorted(hist.items())),
                "low_conf_count": int((conf < args.low_conf_thresh).sum()),
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "name": args.name,
            "version": args.version,
            "branch": args.branch,
            "cam": args.cam,
            "max_offset": int(args.max_offset),
            "smooth_window": int(args.smooth_window),
            "n_records": int(n),
            "low_conf_thresh": float(args.low_conf_thresh),
        },
        "summary": {
            "episodes": len(ep_payload),
            "steps": n_scored,
            "low_conf_steps": int(low_conf),
            "low_conf_ratio": (float(low_conf) / float(n_scored)) if n_scored > 0 else 0.0,
        },
        "by_episode": ep_payload,
        "by_global_index": idx_payload,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))

    print(f"wrote {args.out_json}")
    print(
        "summary:",
        f"episodes={payload['summary']['episodes']}",
        f"steps={payload['summary']['steps']}",
        f"low_conf={payload['summary']['low_conf_steps']} ({100.0 * payload['summary']['low_conf_ratio']:.1f}%)",
    )


if __name__ == "__main__":
    main()
