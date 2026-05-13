"""Define math utility functions."""

import math

import numpy as np
from geometry_msgs.msg import Quaternion
from numpy.typing import NDArray


def quat_to_yaw_rad(quat: NDArray) -> float:
    """Convert a quaternion stored as a NumPy array of shape (4,) into a yaw (radians)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)  # sin(yaw) * cos(pitch)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)  # cos(yaw) * cos(pitch)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_msg_to_yaw_rad(quat_msg: Quaternion) -> float:
    """Convert a geometry_msgs/Quaternion message into a planar yaw angle (radians)."""
    w = float(quat_msg.w)
    x = float(quat_msg.x)
    y = float(quat_msg.y)
    z = float(quat_msg.z)
    return quat_to_yaw_rad(np.array([w, x, y, z]))
