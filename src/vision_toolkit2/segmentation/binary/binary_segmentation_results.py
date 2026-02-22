from ..base_segmentation_results import BaseSegmentationResults

from dataclasses import dataclass

import numpy as np


@dataclass
class BinarySegmentationResults(BaseSegmentationResults):
    is_labeled: np.ndarray[bool]
    fixation_intervals: np.ndarray[int]
    saccade_intervals: np.ndarray[int]
    fixation_centroids: np.ndarray[int]
