"""Define an interface for 2D occupancy grids represented using log odds."""

import math
from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from numpy.typing import NDArray


class GridCell(NamedTuple):
    """A pair of (row, col) indices to a cell in a discrete grid."""

    row: int
    col: int


@dataclass(frozen=True)
class GridInfo:
    """Defines the structure of a 2D grid of cells on an x-y plane.

    Convention: Grid row indices increase as map-frame y-values increase (0 to height - 1)
                Grid column indices increase as map-frame x-values increase (0 to width - 1)

    For simplicity, we assume that the grid is axis-aligned with its parent frame ("map").
    """

    origin_x: float
    origin_y: float
    resolution_m: float
    height_cells: int
    width_cells: int
    parent_frame_id: str = "map"

    @property
    def column_x_coords(self) -> NDArray[np.floating]:
        """Return map-frame x-coordinates at the center of each grid column."""
        return (
            self.origin_x
            + (np.arange(self.width_cells, dtype=float) + 0.5) * self.resolution_m
        )

    @property
    def row_y_coords(self) -> NDArray[np.floating]:
        """Return map-frame y-coordinates at the center of each grid row."""
        return (
            self.origin_y
            + (np.arange(self.height_cells, dtype=float) + 0.5) * self.resolution_m
        )

    @overload
    def x_to_col(self, x: float) -> int: ...
    @overload
    def x_to_col(self, x: NDArray[np.floating]) -> NDArray[np.intp]: ...
    def x_to_col(self, x: float | NDArray[np.floating]) -> int | NDArray[np.intp]:
        """Convert map-frame x-coordinate(s) into the corresponding grid column index/indices."""
        local_x = x - self.origin_x
        return np.floor(local_x / self.resolution_m).astype(np.intp)

    @overload
    def y_to_row(self, y: float) -> int: ...
    @overload
    def y_to_row(self, y: NDArray[np.floating]) -> NDArray[np.intp]: ...
    def y_to_row(self, y: float | NDArray[np.floating]) -> int | NDArray[np.intp]:
        """Convert map-frame y-coordinate(s) into the corresponding grid row index/indices."""
        local_y = y - self.origin_y
        return np.floor(local_y / self.resolution_m).astype(np.intp)

    @overload
    def col_to_x(self, col: int) -> float: ...
    @overload
    def col_to_x(self, col: NDArray[np.intp]) -> NDArray[np.floating]: ...
    def col_to_x(self, col: int | NDArray[np.intp]) -> float | NDArray[np.floating]:
        """Convert column index/indices into world-frame x-coordinate(s) at cell center(s)."""
        local_x = (col + 0.5) * self.resolution_m
        return local_x + self.origin_x

    @overload
    def row_to_y(self, row: int) -> float: ...
    @overload
    def row_to_y(self, row: NDArray[np.intp]) -> NDArray[np.floating]: ...
    def row_to_y(self, row: int | NDArray[np.intp]) -> float | NDArray[np.floating]:
        """Convert row index/indices into world-frame y-coordinate(s) at cell center(s)."""
        local_y = (row + 0.5) * self.resolution_m
        return local_y + self.origin_y

    def coord_to_cell(self, xy: tuple[float, float]) -> GridCell:
        """Convert a map-frame (x,y) coordinate into the corresponding grid cell indices."""
        x, y = xy
        return GridCell(row=self.y_to_row(y), col=self.x_to_col(x))

    def is_valid_cell(self, cell: GridCell) -> bool:
        """Check whether the given cell coordinate is within the grid."""
        return 0 <= cell.row < self.height_cells and 0 <= cell.col < self.width_cells


def unpack_occupancy_grid_msg(
    msg: OccupancyGridMsg,
) -> tuple[NDArray[np.int8], GridInfo]:
    """Convert a nav_msgs/OccupancyGrid message into a NumPy array and grid metadata."""
    height_cells = int(msg.info.height)
    width_cells = int(msg.info.width)

    data_flat = np.asarray(msg.data, dtype=np.int16)
    if data_flat.size != height_cells * width_cells:
        raise ValueError(
            f"OccupancyGrid data size was {data_flat.size} but "
            f"expected {height_cells * width_cells} (H x W) cells."
        )

    grid_data = data_flat.reshape((height_cells, width_cells)).astype(np.int8)
    grid_info = GridInfo(
        origin_x=msg.info.origin.position.x,
        origin_y=msg.info.origin.position.y,
        resolution_m=msg.info.resolution,
        height_cells=height_cells,
        width_cells=width_cells,
        parent_frame_id=msg.header.frame_id,
    )

    return grid_data, grid_info


def inflate_costmap(
    occ_grid: NDArray[np.int16],
    grid_info: GridInfo,
    occupied_threshold: int,
    inflation_radius_m: float,
) -> NDArray[np.int16]:
    """Inflate obstacles in the costmap based on the given radius.

    :param occ_grid: Occupancy grid with values in [0, 100]
    :param grid_info: Grid metadata
    :param occupied_threshold: Integer threshold for occupied cells
    :param inflation_radius_m: Inflation radius in meters
    :return: Inflated occupancy costmap in [0, 100]
    """
    inflated = np.zeros_like(occ_grid, dtype=np.int16)
    occ_cells = np.argwhere(occ_grid >= occupied_threshold)
    if occ_cells.size == 0:
        return inflated

    # Calculate the maximum radius from occupied cells we need to check
    max_radius_cells = max(1, math.ceil(inflation_radius_m / grid_info.resolution_m))
    for r_occ, c_occ in occ_cells:
        for d_row in range(-max_radius_cells, max_radius_cells + 1):
            for d_col in range(-max_radius_cells, max_radius_cells + 1):
                row = int(r_occ + d_row)
                col = int(c_occ + d_col)
                if not grid_info.is_valid_cell(GridCell(row=row, col=col)):
                    continue

                dist_m = math.hypot(
                    d_row * grid_info.resolution_m,
                    d_col * grid_info.resolution_m,
                )
                if dist_m > inflation_radius_m:
                    continue

                cost = round(100.0 * max(0.0, 1.0 - dist_m / inflation_radius_m))

                inflated[row, col] = max(inflated[row, col], cost)

    inflated[occ_grid >= occupied_threshold] = 100
    return inflated


def bresenham_line(c0: GridCell, c1: GridCell) -> list[GridCell]:
    """Compute grid cells along a line using Bresenham's algorithm with integer arithmetic.

    Reference: https://zingl.github.io/bresenham.html ("Line" algorithm)

    :param c0: Start grid cell indices
    :param c1: End grid cell indices
    :return: List of (row, col) cell indices along the line
    """
    x0 = int(c0.col)
    y0 = int(c0.row)
    x1 = int(c1.col)
    y1 = int(c1.row)

    cells = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        cells.append(GridCell(y, x))

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx

        if e2 <= dx:
            err += dx
            y += sy

    return cells
