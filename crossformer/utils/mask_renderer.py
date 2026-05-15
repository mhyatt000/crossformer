"""CPU z-buffer triangle rasterizer for robot silhouette masks.
TOOD: port inner look to cython for speed (?)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float  # horizontal focal length in pixels
    fy: float  # vertical focal length in pixels
    cx: float  # principal point x
    cy: float  # principal point y
    width: int  # img width
    height: int  # img height


def rasterize_mesh(
    verts_cam: np.ndarray,
    tris: np.ndarray,
    link_id: int,
    intr: Intrinsics,
    depth_buf: np.ndarray,
    inst_buf: np.ndarray,
) -> None:
    """Rasterize triangles into depth + instance buffers (in place).

    verts_cam: (V, 3) in camera frame, -Z forward (openGL)
    tris: (T, 3) int 32 triange indices
    """
    W, H = intr.width, intr.height

    # Perspective projection (OpenGL convention: camera looks down -Z)
    # Recover positive depth from the negative-Z camera frame.
    depth = -verts_cam[:, 2]
    valid = depth > 1e-6  # Reject verts behind or on camera plane

    # Guard against div by zero for invalid verts
    safe_depth = np.where(valid, depth, 1.0)

    # Project to pixel coords. cy - <...> flips Y from OpenGL (up) to image conv.
    x_px = intr.fx * (verts_cam[:, 0] / safe_depth) + intr.xc
    y_px = intr.cy - intr.fy * (verts_cam[:, 1] / safe_depth)

    # 2D proj. vertex positions; (V, 2)
    v2 = np.stack([x_px, y_px], axis=1).astype(np.float32)
    # --- Per-triangle validity and bounding-box cull ---
    # A triangle is valid only when all three vertices are in front of the camera.
    tri_valid = valid[tris].all(axis=1)

    # Gather per-triangle 2-D positions and depths.
    p0 = v2[tris[:, 0]]
    p1 = v2[tris[:, 1]]
    p2 = v2[tris[:, 2]]
    d0 = depth[tris[:, 0]]
    d1 = depth[tris[:, 1]]
    d2 = depth[tris[:, 2]]

    # Signed 2x triangle area via 2-D cross product — used for winding-order
    # tests and barycentric coordinate normalisation.
    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    tri_valid &= np.abs(area2) > 1e-6  # skip degenerate (zero-area) triangles

    # Integer bounding box per triangle, clipped to the image.
    xmin = np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32)
    xmax = np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32)
    ymin = np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32)
    ymax = np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32)

    xmin = np.clip(xmin, 0, W)
    xmax = np.clip(xmax, 0, W)
    ymin = np.clip(ymin, 0, H)
    ymax = np.clip(ymax, 0, H)

    # If the clipped bbox is empty the triangle is off-screen.
    tri_valid &= (xmax > xmin) & (ymax > ymin)

    # Rasterize each surviving triangle one at a time.
    idx = np.nonzero(tri_valid)[0]
    for i in idx:
        _rasterize_one(
            p0[i],
            p1[i],
            p2[i],
            d0[i],
            d1[i],
            d2[i],
            area2[i],
            int(xmin[i]),
            int(xmax[i]),
            int(ymin[i]),
            int(ymax[i]),
            link_id,
            depth_buf,
            inst_buf,
        )


def _rasterize_one(
    a,
    b,
    c,
    da,
    db,
    dc,
    area2,
    xmin,
    xmax,
    ymin,
    ymax,
    link_id,
    depth_buf,
    inst_buf,
) -> None:
    """Rasterize a single triangle into the depth and instance buffers.

    Uses a grid of pixel centres over the triangle's bounding box, computes
    edge-function point-in-triangle tests, then interpolates depth via
    perspective-correct barycentrics and writes pixels that pass the z-test.

    a, b, c:   2-D projected vertices (float32, shape (2,)).
    da, db, dc: positive depths at each vertex.
    area2:      signed 2x triangle area (determines winding).
    link_id:    robot link index written into inst_buf for hit pixels.
    """
    # Build a grid of pixel-centre sample points over the bounding box.
    xs = np.arange(xmin, xmax, dtype=np.float32) + 0.5
    ys = np.arange(ymin, ymax, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # Each w_i is the signed area of the sub-triangle formed by one edge and
    # the sample point.  A point is inside the triangle when all three have
    # the same sign (which depends on the winding order given by area2).
    w0 = (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])
    w1 = (c[0] - b[0]) * (Y - b[1]) - (c[1] - b[1]) * (X - b[0])
    w2 = (a[0] - c[0]) * (Y - c[1]) - (a[1] - c[1]) * (X - c[0])

    # CW vs CCW winding: flip the comparison direction.
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) if area2 > 0 else (w0 <= 0) & (w1 <= 0) & (w2 <= 0)

    if not inside.any():
        return

    # Barycentric coordinates from the edge functions (normalised by area).
    inv_area = 1.0 / area2
    l0 = ((b[0] - c[0]) * (Y - c[1]) - (b[1] - c[1]) * (X - c[0])) * inv_area
    l1 = ((c[0] - a[0]) * (Y - a[1]) - (c[1] - a[1]) * (X - a[0])) * inv_area
    l2 = 1.0 - l0 - l1

    # Interpolate in 1/z (the quantity that is linear in screen space) then
    # recover z.  This gives perspective-correct depth at each pixel.
    inv_z = l0 / da + l1 / db + l2 / dc
    z = 1.0 / np.maximum(inv_z, 1e-9)

    existing = depth_buf[ymin:ymax, xmin:xmax]
    hit = inside & (z < existing)  # closer than what's already stored

    if not hit.any():
        return

    # Write winning depths and stamp the link id into the instance mask.
    existing[hit] = z[hit]
    inst_buf[ymin:ymax, xmin:xmax][hit] = np.uint8(link_id)
