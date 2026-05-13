"""Coordinate frame transform utilities."""

import math
import numpy as np


def pose_2d_to_transform_matrix(x: float, y: float, theta: float) -> np.ndarray:
    """Create a 4x4 homogeneous transform from a 2D pose (x, y, theta).

    The resulting matrix transforms points from the body frame to the world frame.
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0, x],
        [s,  c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a 3D point.

    Args:
        T: 4x4 transformation matrix.
        point: 3D point as (3,) array.

    Returns:
        Transformed 3D point as (3,) array.
    """
    p_h = np.array([point[0], point[1], point[2], 1.0])
    return (T @ p_h)[:3]


def make_camera_transform(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build a 4x4 world-from-camera transform from position and MuJoCo quaternion [w,x,y,z]."""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T
