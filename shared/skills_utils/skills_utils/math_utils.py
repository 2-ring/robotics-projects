"""Math helpers for robotics computations."""

import math
import numpy as np


def quat_to_yaw_rad(quat: np.ndarray) -> float:
    """Convert MuJoCo quaternion [w, x, y, z] to yaw angle in radians."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle (radians) to MuJoCo quaternion [w, x, y, z]."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for angle theta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
