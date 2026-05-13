"""Procedural world generation and costmap utilities."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from slam_utils import GridInfo


@dataclass(frozen=True)
class CircularObstacle:
    """A circular obstacle represented in world coordinates."""

    x_m: float
    y_m: float
    radius_m: float


@dataclass(frozen=True)
class RectangularObstacle:
    """A rectangular obstacle represented in world coordinates."""

    x_m: float
    y_m: float
    half_width_m: float
    """Rectangle size (meters) along its x-axis."""
    half_height_m: float
    """Rectangle size (meters) along its y-axis."""
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class Environment2D:
    """An environment composed of 2D circular and rectangular obstacles."""

    circles: tuple[CircularObstacle, ...] = ()
    rectangles: tuple[RectangularObstacle, ...] = ()


def parse_scene_environment(scene_xml: Path) -> Environment2D:
    """Parse MuJoCo scene geometry into an Environment2D object.

    :param scene_xml: Filepath to the MuCoJo scene XML
    :return: Parsed environment containing circular and rectangular obstacles
    """
    xml_tree = ET.parse(scene_xml)
    xml_root = xml_tree.getroot()

    circles: list[CircularObstacle] = []
    rectangles: list[RectangularObstacle] = []

    for geom in xml_root.findall(".//worldbody/geom"):
        name = geom.get("name", "")

        # Only include obstacles and walls in the parsed 2D environment
        if not (name.startswith("obs_") or name.startswith("wall_")):
            continue

        geom_type = geom.get("type", "")
        position_tokens = geom.get("pos", "0 0 0").split()
        size_tokens = geom.get("size", "0 0 0").split()

        if len(position_tokens) < 2 or not size_tokens:
            print(f"Couldn't parse XML geometry: {geom}")
            continue

        center_x_m = float(position_tokens[0])
        center_y_m = float(position_tokens[1])

        if geom_type == "box" and len(size_tokens) >= 2:
            half_width_m = float(size_tokens[0])
            half_height_m = float(size_tokens[1])
            yaw_rad = 0.0

            euler_tokens = geom.get("euler", "").split()
            if len(euler_tokens) >= 3:
                yaw_rad = float(euler_tokens[2])

            rectangles.append(
                RectangularObstacle(
                    x_m=center_x_m,
                    y_m=center_y_m,
                    half_width_m=half_width_m,
                    half_height_m=half_height_m,
                    yaw_rad=yaw_rad,
                )
            )
        elif geom_type == "cylinder" and len(size_tokens) >= 1:
            circles.append(
                CircularObstacle(
                    x_m=center_x_m,
                    y_m=center_y_m,
                    radius_m=float(size_tokens[0]),
                )
            )

    return Environment2D(circles=tuple(circles), rectangles=tuple(rectangles))


def rasterize_environment(
    grid_info: GridInfo, environment: Environment2D
) -> NDArray[np.bool_]:
    """Rasterize an Environment2D into a Boolean occupancy map.

    :param grid_info: Occupancy grid configuration
    :param environment: Environment to rasterize
    :return: NumPy array of occupancy values (True = occupied)
    """
    grid_shape = (grid_info.height_cells, grid_info.width_cells)
    grid_data = np.zeros(grid_shape, dtype=bool)

    cell_x_coords = grid_info.column_x_coords[None, :]  # Shape (1, W)
    cell_y_coords = grid_info.row_y_coords[:, None]  # Shape (H, 1)

    for rect in environment.rectangles:
        dx_m = cell_x_coords - rect.x_m
        dy_m = cell_y_coords - rect.y_m
        cos_yaw = np.cos(rect.yaw_rad)
        sin_yaw = np.sin(rect.yaw_rad)

        # Rotate grid coordinates into the rectangle's local frame before bounds check
        local_x_m = cos_yaw * dx_m + sin_yaw * dy_m
        local_y_m = -sin_yaw * dx_m + cos_yaw * dy_m

        inside_mask = (np.abs(local_x_m) <= rect.half_width_m) & (
            np.abs(local_y_m) <= rect.half_height_m
        )
        grid_data[inside_mask] = True

    for circle in environment.circles:
        inside_mask = (
            np.hypot(cell_x_coords - circle.x_m, cell_y_coords - circle.y_m)
            <= circle.radius_m
        )
        grid_data[inside_mask] = True

    return grid_data


def build_gt_map(grid_info: GridInfo, scene_xml: Path) -> NDArray[np.bool_]:
    """Rasterize obstacles from a MuJoCo scene into a ground-truth occupancy grid.

    :param grid_info: Occupancy grid configuration
    :param scene_xml: Filepath to the MuCoJo scene XML
    :return: NumPy array of ground-truth occupancy values (True = occupied)
    """
    environment = parse_scene_environment(scene_xml=scene_xml)
    return rasterize_environment(grid_info=grid_info, environment=environment)
