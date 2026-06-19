from __future__ import annotations

from typing import Mapping

import numpy as np

# Matrices are ROS FLU-from-convention bases. Columns are the convention's
# x/y/z axes expressed in ROS body coords: x forward, y left, z up.
ros = np.eye(3, dtype=np.float32)
ocv = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)  # right, down, forward
ogl = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)  # right, up, back
uty = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)  # right, up, forward
p3d = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)  # left, up, forward
ros_optical = ocv

CONVENTIONS: Mapping[str, np.ndarray] = {
    "ocv": ocv,
    "opencv": ocv,
    "ogl": ogl,
    "opengl": ogl,
    "uty": uty,
    "unity": uty,
    "p3d": p3d,
    "pytorch3d": p3d,
    "ros": ros,
    "ros_optical": ros_optical,
}


def convert(src: str, dst: str) -> np.ndarray:
    """Return the 3x3 matrix that maps src coords to dst coords."""
    return CONVENTIONS[dst].T @ CONVENTIONS[src]


def hom(R: np.ndarray) -> np.ndarray:
    """Lift a 3x3 convention matrix to homogeneous 4x4."""
    H = np.eye(4, dtype=R.dtype)
    H[:3, :3] = R
    return H
