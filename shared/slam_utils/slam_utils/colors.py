"""Shared RGBA color definitions for RViz and MuJoCo."""

from typing import TypeAlias

RGB: TypeAlias = tuple[int, int, int]
RGBA255: TypeAlias = tuple[int, int, int, int]
RGBA01: TypeAlias = tuple[float, float, float, float]

ROBOT_BODY_RGB: RGB = (33, 107, 163)
OBSTACLE_RGB: RGB = (184, 26, 42)
LASER_RGB: RGB = (255, 51, 26)

ROBOT_GT_RGBA_255: RGBA255 = (*ROBOT_BODY_RGB, 235)
ROBOT_EST_RGBA_255: RGBA255 = (*ROBOT_BODY_RGB, 118)
LASER_RGBA_255: RGBA255 = (*LASER_RGB, 166)
OBSTACLE_RGBA_255: RGBA255 = (*OBSTACLE_RGB, 230)


def rgba_255_to_unit(rgba_255: RGBA255) -> RGBA01:
    """Convert an 8-bit RGBA color into a unit-range RGBA color."""
    r01 = rgba_255[0] / 255.0
    g01 = rgba_255[1] / 255.0
    b01 = rgba_255[2] / 255.0
    a01 = rgba_255[3] / 255.0
    return (r01, g01, b01, a01)


def rgba_255_to_mujoco_string(rgba_255: RGBA255) -> str:
    """Convert an 8-bit RGBA color into a MuJoCo-compatible string."""
    r, g, b, a = rgba_255_to_unit(rgba_255)
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"
