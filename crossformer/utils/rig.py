"""Static rig calibration and CPU robot-silhouette mask rendering.

Provider per-camera (K, w2c) for xgym setup and thin wrapper around
in-tree CPU mask rendrer for use inside grain data workers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crossformer.utils.mask_renderer import Intrinsics, rasterize_mesh

"""Coordinate convention transform
XGYM extrinsics are stored in FLU (front-left-up) convention.
RDF (right-down-forward) is standard OpenCV convention.

This 4x4 rotation maps FLU basis vectors to RDF:
    FLU-x (front) -> RDF-z (forward)
    FLU-y (left) -> RDF-x (right, negated)
    FLU-z (up) -> RDF-y (down, negated)
"""
FLU2RDF = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)

DEFAULT_EXTR_DIR = Path.home() / "data/extr/cam"
DEFAULT_URDF = Path.home() / "crossformer/xarm7_standalone.urdf"
# xarm URDF paramaterizes finger spread as fingle drive_joint
# in [0, 0.85] radians. 0.85 = closed
GRIPPER_CLOSED_RAD = 0.85  # URDF drive_joint range [0, 0.85]; 0.85 = closed (fingers together)


def K_for_size(h: int, w: int, f: float = 515.0) -> np.ndarray:
    """Build 3x3 pinhole intrinsic matrix with square pixels.

    Assumes principal point is at the image center and both focal
    lengths equal *f* (no skew; no distortion)
    """
    return np.array(
        [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_w2c(cam: str, extr_dir: Path = DEFAULT_EXTR_DIR) -> np.ndarray:
    """Load a world-to-cam 4x4 matrix for cam.

    The on-disk file stores the hand-eye calibration as 4x4 homogeneous
    transform *HT* in FLU convention. We convert FLU -> RDF (Opencv)
    and invert to get world-to-cam
    """
    HT = np.load(extr_dir / cam / "HT.npz")["HT"].astype(np.float64)
    return np.linalg.inv(HT @ FLU2RDF).astype(np.float32)


def w2c_to_renderer(w2c: np.ndarray) -> np.ndarray:
    """Converts an OpenCV world-to-camera matrix to the renderer's format.

    The CPU rasterizer expects an OpenGL-style matrix (Y and Z axes flipped
    relative to OpenCV) stored as a row-vector transform
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # opencv -> opengl axis flip
    return (flip @ w2c.astype(np.float64)).T


def render_robot_mask(
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
    gripper_rad: float = GRIPPER_CLOSED_RAD,
) -> np.ndarray:
    """Render binary robot sihouette (h, w) uint8 in {0, 255}

    Uses pyroki FK for joint convention parity, then
    rasterizes via the in-tree CPU z-nuffer.
    """

    from crossformer.utils.callbacks.synth_viz import _get_robot_mesh

    robot = _get_robot_mesh()

    # Pack joint angles into a (1, N_actuaged) batch for pyroki FK,
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, 0:7] = joints_rad
    q[0, 7] = float(gripper_rad)

    # FK -> world-frame mesh vertices, (V, 4) homogeneous.
    verts_world = robot.posed_verts(q)[0]

    # Project into camera's openGL frame (-Z forward, +Y up)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c_gl = flip @ w2c.astype(np.float64)
    verts_cam = (w2c_gl @ verts_world.astype(np.float64).T).T[:, :3]

    # Build intrinsics from the 3x3 K matrix.
    intr = Intrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=int(w),
        height=int(h),
    )

    # Initialize z-buffer and instance mask
    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    inst_buf = np.zeros((h, w), dtype=np.uint8)

    # Rasterize the full robot mesh as a single instance
    rasterize_mesh(verts_cam, robot.faces, 1, intr, depth_buf, inst_buf)

    return (inst_buf > 0).astype(np.uint8) * 255
