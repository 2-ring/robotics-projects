"""Define metrics to evaluate student solutions for the assignment."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class OccupancyMetrics(NamedTuple):
    """Organizes metrics for predicted occupancy grids."""

    precision: float
    recall: float
    f1: float


def occupancy_f1_score(
    predicted_grid: NDArray[np.int8],
    gt_occ_mask: NDArray[np.bool_],
    occupied_threshold: int,
) -> OccupancyMetrics:
    """Compute occupancy precision/recall/F1 scores using the given grids.

    :param predicted_grid: Student occupancy map with values in [-1, 100]
    :param gt_occ_mask: Ground-truth Boolean occupancy mask
    :param occupied_threshold: Threshold at which cells are considered occupied
    """
    if predicted_grid.shape != gt_occ_mask.shape:
        raise ValueError("Predicted and ground-truth grids must have the same shape.")

    predicted_occ_mask = predicted_grid >= occupied_threshold

    tp_count = int(np.sum(predicted_occ_mask & gt_occ_mask))
    fp_count = int(np.sum(predicted_occ_mask & ~gt_occ_mask))
    fn_count = int(np.sum(~predicted_occ_mask & gt_occ_mask))

    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1)

    return OccupancyMetrics(precision, recall, f1_score)
