"""Pre-render robot masks + 2D keypoints for xgym_sweep_single (or any
xgym dataset on the same rig).

Input: existing arec at ``<root>/<name>/<src_version>/<branch>/`` with image+proprio.
Output: new arec at ``<root>/<name>/<dst_version>/<branch>/`` with extra
``mask`` and ``kp2d`` per-camera fields baked in.

Usage:
    # preview a few records to a tmp arec
    uv run scripts/data/make/prerender_xgym_masks.py \\
        --name xgym_sweep_single --src-version 0.5.6 --dst-version 0.6.0 \\
        --cams low side --n-preview 16

    # full build (this is the slow step — uses --workers parallelism)
    uv run scripts/data/make/prerender_xgym_masks.py \\
        --name xgym_sweep_single --src-version 0.5.6 --dst-version 0.6.0 \\
        --cams low side --workers 16

Optimisations over the original:
    1. FK is computed once per record, shared across cameras (was redundant).
    2. Rasterisation at 1/4 resolution (configurable --render-scale), then
       nearest-neighbour upscale.  Masks are binary silhouettes so this is
       visually lossless, but ~16x fewer pixels to shade.
    3. Mesh decimation via --decimate-ratio (default 0.1 = keep 10% of faces).
       Silhouettes are insensitive to interior triangle detail.
    4. Vectorised rasteriser: the per-triangle Python loop is replaced with a
       batched edge-function test over padded bounding boxes.
    5. Larger imap chunksize (16 vs 4) to amortise IPC.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

# Spawned workers import this module before running _worker_init.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
from tqdm import tqdm

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record


def _rasterize_mesh_fast(
    verts_cam: np.ndarray,
    tris: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Vectorised z-buffer rasteriser.  Returns a binary mask (H, W) uint8.

    All triangles are tested against a padded bounding-box grid in a single
    batched operation — no Python-level per-triangle loop.

    Parameters:
        tris      : (T, 3)  int32 face indices.
        width, height, fx, fy, cx, cy : intrinsics.
        verts_cam : (V, 3)  Camera-frame vertices, OpenGL convention (-Z forward).
    """
    W, H = width, height

    # Projection
    depth = -verts_cam[:, 2]
    valid = depth > 1e-6
    safe_depth = np.where(valid, depth, 1.0)
    x_px = fx * (verts_cam[:, 0] / safe_depth) + cx
    y_px = cy - fy * (verts_cam[:, 1] / safe_depth)
    v2 = np.stack([x_px, y_px], axis=1).astype(np.float32)

    # per-triangle culling
    tri_valid = valid[tris].all(axis=1)
    p0, p1, p2 = v2[tris[:, 0]], v2[tris[:, 1]], v2[tris[:, 2]]
    d0, d1, d2 = depth[tris[:, 0]], depth[tris[:, 1]], depth[tris[:, 2]]

    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    tri_valid &= np.abs(area2) > 1e-6

    xmin = np.clip(np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32), 0, W)
    xmax = np.clip(np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32), 0, W)
    ymin = np.clip(np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32), 0, H)
    ymax = np.clip(np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32), 0, H)
    tri_valid &= (xmax > xmin) & (ymax > ymin)

    idx = np.nonzero(tri_valid)[0]
    if len(idx) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Gather surviving triangles.
    p0, p1, p2 = p0[idx], p1[idx], p2[idx]
    d0, d1, d2 = d0[idx], d1[idx], d2[idx]
    area2 = area2[idx]
    xmin, xmax = xmin[idx], xmax[idx]
    ymin, ymax = ymin[idx], ymax[idx]

    # --- Sort front-to-back by minimum depth for better z-test rejection ---
    min_d = np.minimum(np.minimum(d0, d1), d2)
    order = np.argsort(min_d)
    p0, p1, p2 = p0[order], p1[order], p2[order]
    d0, d1, d2 = d0[order], d1[order], d2[order]
    area2 = area2[order]
    xmin, xmax = xmin[order], xmax[order]
    ymin, ymax = ymin[order], ymax[order]

    T = len(p0)

    # --- Batch into size-classes to limit padding waste ---
    # Group triangles by bounding-box area bracket, process each group as
    # one vectorised batch.  This avoids the worst-case where one huge
    # triangle forces max_h x max_w allocation for every triangle.
    bb_w = xmax - xmin
    bb_h = ymax - ymin
    bb_area = bb_w * bb_h

    depth_buf = np.full((H, W), np.inf, dtype=np.float32)

    # Size thresholds: process small, medium, and large triangles separately.
    # Small triangles are the vast majority for typical robot meshes.
    SMALL, MED = 64, 512
    for lo, hi in [(0, SMALL), (SMALL, MED), (MED, bb_area.max() + 1)]:
        sel = (bb_area >= lo) & (bb_area < hi)
        if not sel.any():
            continue
        si = np.nonzero(sel)[0]

        bw = bb_w[si]
        bh = bb_h[si]
        mw = int(bw.max())
        mh = int(bh.max())
        if mw == 0 or mh == 0:
            continue

        n = len(si)

        # Build per-triangle local grids: pixel centres relative to bbox origin.
        # xs[t, j] = xmin[t] + j + 0.5, etc. (padded beyond bbox width with -1).
        xs = np.arange(mw, dtype=np.float32)[None, :] + 0.5  # (1, mw)
        ys = np.arange(mh, dtype=np.float32)[None, :] + 0.5  # (1, mh)

        # Absolute pixel coordinates.
        Xabs = xs + xmin[si, None].astype(np.float32)  # (n, mw)
        Yabs = ys + ymin[si, None].astype(np.float32)  # (n, mh)

        # Expand to (n, mh, mw).
        X = np.broadcast_to(Xabs[:, None, :], (n, mh, mw))
        Y = np.broadcast_to(Yabs[:, :, None], (n, mh, mw))

        # Edge functions — (n, mh, mw).
        a, b, c = p0[si], p1[si], p2[si]
        # Reshape vertices to (n, 1, 1) for broadcasting.
        ax, ay = a[:, 0:1, None], a[:, 1:2, None]
        bx, by = b[:, 0:1, None], b[:, 1:2, None]
        cx, cy_ = c[:, 0:1, None], c[:, 1:2, None]

        w0 = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)
        w1 = (cx - bx) * (Y - by) - (cy_ - by) * (X - bx)
        w2 = (ax - cx) * (Y - cy_) - (ay - cy_) * (X - cx)

        # Winding-aware inside test.
        ccw = (area2[si] > 0)[:, None, None]
        inside = np.where(ccw, (w0 >= 0) & (w1 >= 0) & (w2 >= 0), (w0 <= 0) & (w1 <= 0) & (w2 <= 0))

        # Mask out padding (pixels beyond this triangle's actual bbox).
        col_valid = np.arange(mw)[None, None, :] < bw[:, None, None]
        row_valid = np.arange(mh)[None, :, None] < bh[:, None, None]
        inside &= col_valid & row_valid

        if not inside.any():
            continue

        # Barycentrics + perspective-correct depth.
        inv_area = (1.0 / area2[si])[:, None, None]
        l0 = ((bx - cx) * (Y - cy_) - (by - cy_) * (X - cx)) * inv_area
        l1 = ((cx - ax) * (Y - ay) - (cy_ - ay) * (X - ax)) * inv_area
        l2 = 1.0 - l0 - l1

        da = d0[si][:, None, None]
        db = d1[si][:, None, None]
        dc = d2[si][:, None, None]
        inv_z = l0 / da + l1 / db + l2 / dc
        z = 1.0 / np.maximum(inv_z, 1e-9)

        # --- Scatter into depth buffer, one triangle at a time ---
        # This inner loop is unavoidable (z-test is order-dependent) but the
        # heavy maths (edge tests, barycentrics) were done in batch above.
        for k in range(n):
            if not inside[k].any():
                continue
            ym, xm = int(ymin[si[k]]), int(xmin[si[k]])
            bh_k, bw_k = int(bb_h[si[k]]), int(bb_w[si[k]])
            local_inside = inside[k, :bh_k, :bw_k]
            local_z = z[k, :bh_k, :bw_k]

            existing = depth_buf[ym : ym + bh_k, xm : xm + bw_k]
            hit = local_inside & (local_z < existing)
            if hit.any():
                existing[hit] = local_z[hit]

    return (depth_buf < np.inf).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Worker: each process loads pyroki + URDF mesh once, then renders masks.
# ---------------------------------------------------------------------------

_WORKER_CTX: dict | None = None


def _worker_init(
    extr_dir: str,
    cams: tuple[str, ...],
    raw_h: int,
    raw_w: int,
    render_scale: float,
    decimate_ratio: float,
) -> None:
    """Load pyroki mesh + per-camera (K, w2c) once per worker."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    from crossformer.utils.rig import K_for_size, load_w2c
    from crossformer.utils.robot_model import fk_keypoints, get_robot_mesh

    robot = get_robot_mesh()

    # --- Mesh decimation ---
    # Robot CAD meshes are often 50k+ faces; silhouettes only need the hull.
    faces = robot.faces
    if decimate_ratio < 1.0:
        try:
            import trimesh

            mesh = trimesh.Trimesh(
                vertices=np.zeros((faces.max() + 1, 3)),  # placeholder
                faces=faces,
                process=False,
            )
            target = max(int(len(faces) * decimate_ratio), 100)
            mesh = mesh.simplify_quadric_decimation(target)
            faces = mesh.faces.astype(np.int32)
        except Exception:
            pass  # if trimesh unavailable, use original faces

    # Render resolution: smaller than raw for speed, upscaled later.
    rh = max(1, int(raw_h * render_scale))
    rw = max(1, int(raw_w * render_scale))
    K_raw = K_for_size(raw_h, raw_w)
    K_render = K_for_size(rh, rw)
    w2c_per_cam = {cam: load_w2c(cam, Path(extr_dir)) for cam in cams}

    # Warm pyroki FK + JIT with a dummy joint vector.
    dummy_joints = np.zeros(7, dtype=np.float64)
    _ = fk_keypoints(dummy_joints)
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    _ = robot.posed_verts(q)

    global _WORKER_CTX
    _WORKER_CTX = {
        "robot": robot,
        "faces": faces,
        "fk_keypoints": fk_keypoints,
        "K_raw": K_raw,
        "K_render": K_render,
        "w2c": w2c_per_cam,
        "raw_h": raw_h,
        "raw_w": raw_w,
        "render_h": rh,
        "render_w": rw,
        "cams": cams,
    }


def _project_kp(pts_3d: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D keypoints to pixel coords.  Returns (uv, z)."""
    pts_h = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=pts_3d.dtype)], axis=-1)
    cam = (w2c.astype(np.float64) @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pix = (K.astype(np.float64) @ cam.T).T
    uv = pix[:, :2] / z_safe[:, None]
    return uv.astype(np.float32), z.astype(np.float32)


def _process_record(args: tuple[int, np.ndarray, float]) -> tuple[int, dict]:
    """Render masks + kp2d for all configured cameras.

    FK is computed once and shared across cameras (was the main redundancy
    in the original code).
    """
    idx, joints, gripper_drive = args
    ctx = _WORKER_CTX
    assert ctx is not None

    robot = ctx["robot"]
    K_raw = ctx["K_raw"]
    K_render = ctx["K_render"]
    raw_h, raw_w = ctx["raw_h"], ctx["raw_w"]
    rh, rw = ctx["render_h"], ctx["render_w"]
    faces = ctx["faces"]

    # --- FK once for all cameras ---
    pts_3d = ctx["fk_keypoints"](joints.astype(np.float64))

    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints
    q[0, 7] = float(gripper_drive)
    verts_world = robot.posed_verts(q)[0]  # (V, 4) homogeneous

    masks: dict[str, np.ndarray] = {}
    kps: dict[str, np.ndarray] = {}

    for cam in ctx["cams"]:
        w2c = ctx["w2c"][cam]

        # --- 2D keypoints (always at full resolution) ---
        uv, z = _project_kp(pts_3d, w2c, K_raw)
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < raw_w) & (uv[:, 1] >= 0) & (uv[:, 1] < raw_h)
        vis = (in_frame & (z > 0)).astype(np.float32)
        kp2d = np.concatenate([uv, vis[:, None]], axis=-1)

        # --- Mask at render resolution ---
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        w2c_gl = flip @ w2c.astype(np.float64)
        verts_cam = (w2c_gl @ verts_world.astype(np.float64).T).T[:, :3]

        mask_small = _rasterize_mesh_fast(
            verts_cam,
            faces,
            rw,
            rh,
            fx=float(K_render[0, 0]),
            fy=float(K_render[1, 1]),
            cx=float(K_render[0, 2]),
            cy=float(K_render[1, 2]),
        )

        # Upscale to full resolution via nearest-neighbour (binary mask).
        if rh != raw_h or rw != raw_w:
            mask = np.kron(
                mask_small,
                np.ones((raw_h // rh, raw_w // rw), dtype=np.uint8),
            )
            # Handle non-integer scale factors: crop or pad to exact size.
            mask = mask[:raw_h, :raw_w]
        else:
            mask = mask_small

        masks[cam] = mask
        kps[cam] = kp2d

    return idx, {"mask": masks, "kp2d": kps}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _open_src(name: str, version: str, branch: str, root: Path):
    """Open the existing image+proprio writers as plain ArrayRecordDataSources."""
    builder = ArrayRecordBuilder(name=name, version=version, branch=branch, root=str(root))
    meta_path = builder.root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta: {meta_path}")
    import json

    meta = json.loads(meta_path.read_text())
    builder.writers = builder._normalize_writers(meta["writers"])
    builder.default_writer = "data" if "data" in builder.writers else next(iter(builder.writers))

    return builder, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="xgym_sweep_single")
    p.add_argument("--src-version", default="0.5.6")
    p.add_argument("--dst-version", default="0.6.0")
    p.add_argument("--branch", default="main")
    p.add_argument("--root", type=Path, default=Path.home() / ".cache/arrayrecords")
    p.add_argument("--cams", nargs="+", default=["low", "side"])
    p.add_argument("--extr-dir", type=Path, default=Path.home() / "data/extr/cam")
    p.add_argument("--raw-h", type=int, default=480)
    p.add_argument("--raw-w", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-preview", type=int, default=0, help="if > 0, only process this many records to a tmp branch")
    p.add_argument("--shard-size", type=int, default=1000)
    # --- New flags ---
    p.add_argument(
        "--render-scale",
        type=float,
        default=0.25,
        help="Render masks at this fraction of raw resolution, then upscale. "
        "0.25 = 120x160 for 480x640. Set 1.0 for full-res.",
    )
    p.add_argument(
        "--decimate-ratio",
        type=float,
        default=0.1,
        help="Keep this fraction of mesh faces (0.1 = 10%%). Set 1.0 to skip.",
    )
    p.add_argument("--chunksize", type=int, default=16, help="imap chunksize for worker pool dispatch.")
    args = p.parse_args()

    root = args.root.expanduser()
    cams = tuple(args.cams)
    print(f"reading {args.name} v{args.src_version} from {root}", flush=True)
    print(f"  render scale: {args.render_scale}  decimate: {args.decimate_ratio}", flush=True)
    src_builder, _ = _open_src(args.name, args.src_version, args.branch, root)

    if not {"image", "proprio"}.issubset(src_builder.writers):
        raise ValueError(f"expected image+proprio writers, got {list(src_builder.writers)}")

    img_src = src_builder.get_source("image")
    pro_src = src_builder.get_source("proprio")
    n = len(img_src)
    assert n == len(pro_src), f"image/proprio length mismatch: {len(img_src)} vs {len(pro_src)}"
    print(f"  records: {n}", flush=True)

    if args.n_preview > 0:
        n = min(n, args.n_preview)
        dst_branch = f"{args.branch}_preview"
        print(f"  preview mode: writing first {n} records to branch={dst_branch}", flush=True)
    else:
        dst_branch = args.branch

    dst_writers = {
        "image": (["image", "mask"], {"options": "group_size:1"}),
        "proprio": (["proprio", "info", "kp2d"], {"options": "group_size:1"}),
    }
    dst_builder = ArrayRecordBuilder(
        name=args.name,
        version=args.dst_version,
        branch=dst_branch,
        root=str(root),
        shard_size=args.shard_size,
        writers=dst_writers,
    )
    print(f"  writing to {dst_builder.root}", flush=True)

    def gen():
        ctx_args = (str(args.extr_dir), cams, args.raw_h, args.raw_w, args.render_scale, args.decimate_ratio)
        with mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=ctx_args,
        ) as pool:
            print(f"  pool of {args.workers} workers spawned + warmed", flush=True)

            def task_iter():
                for i in range(n):
                    pro = unpack_record(pro_src[i])
                    joints = np.asarray(pro["proprio"]["joints"], dtype=np.float32).reshape(7)
                    gripper = float(np.asarray(pro["proprio"]["gripper"]).reshape(-1)[0])
                    gripper_drive = gripper * 0.85
                    yield (i, joints, gripper_drive)

            t0 = time.perf_counter()
            for n_done, (idx, render_out) in enumerate(
                tqdm(
                    pool.imap(_process_record, task_iter(), chunksize=args.chunksize),
                    total=n,
                    desc="rendering",
                ),
                start=1,
            ):
                img_rec = unpack_record(img_src[idx])
                pro_rec = unpack_record(pro_src[idx])
                sample = {
                    "image": img_rec["image"],
                    "mask": render_out["mask"],
                    "proprio": pro_rec["proprio"],
                    "info": pro_rec.get("info", {}),
                    "kp2d": render_out["kp2d"],
                }
                yield sample
                if n_done % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = n_done / elapsed
                    eta = (n - n_done) / rate if rate > 0 else float("inf")
                    print(f"  [{n_done}/{n}] rate={rate:.1f}/s eta={eta / 60:.1f}min", flush=True)

    print("starting build...", flush=True)
    t = time.perf_counter()
    dst_builder.prepare(gen)
    print(f"done in {(time.perf_counter() - t) / 60:.1f} min → {dst_builder.root}", flush=True)


if __name__ == "__main__":
    main()
