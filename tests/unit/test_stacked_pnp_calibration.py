from __future__ import annotations

import cv2
import numpy as np
import pytest

from crossformer.utils.spatial.calibration import project_points, solve_stacked_pnp


def _w2c_from_rvec_t(rvec, t):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = np.asarray(t, dtype=np.float64)
    return w2c


def test_solve_stacked_pnp_recovers_shared_camera():
    K = np.array([[620.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    w2c = _w2c_from_rvec_t([0.2, -0.1, 0.08], [0.1, -0.05, 1.2])
    base = np.array(
        [
            [-0.2, -0.1, 0.0],
            [-0.1, 0.1, 0.05],
            [0.0, -0.08, 0.02],
            [0.08, 0.12, 0.1],
            [0.18, -0.04, 0.08],
        ],
        dtype=np.float64,
    )
    shifts = np.array([[0.0, 0.0, 0.0], [0.15, 0.02, 0.08], [-0.1, 0.12, -0.03], [0.05, -0.14, 0.12]])
    pts = base[None] + shifts[:, None]
    uv = np.stack([project_points(w2c, frame, K) for frame in pts])

    res = solve_stacked_pnp(pts, uv, K, min_points=8, reproj_thresh_px=1.0)

    assert res.success
    assert res.n_points == 20
    assert res.reproj_px < 1e-6
    np.testing.assert_allclose(res.w2c, w2c, atol=1e-6)


def test_solve_stacked_pnp_requires_enough_points():
    K = np.eye(3, dtype=np.float64)
    pts = np.zeros((1, 3, 3), dtype=np.float64)
    uv = np.zeros((1, 3, 2), dtype=np.float64)

    res = solve_stacked_pnp(pts, uv, K, min_points=4)

    assert not res.success
    assert res.n_points == 3


@pytest.mark.parametrize(
    ("pts", "uv", "K", "match"),
    [
        (np.zeros((4, 3)), np.zeros((4, 2)), np.eye(4), "expected K shape"),
        (np.zeros((4, 2)), np.zeros((4, 2)), np.eye(3), "expected pts_3d"),
        (np.zeros((4, 3)), np.zeros((5, 2)), np.eye(3), "expected pts_3d"),
    ],
)
def test_solve_stacked_pnp_validates_shapes(pts, uv, K, match):
    with pytest.raises(ValueError, match=match):
        solve_stacked_pnp(pts, uv, K)
