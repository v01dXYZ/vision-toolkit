from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class TernarySegmentationResults:
    is_saccade: np.ndarray[bool]
    saccade_intervals: np.ndarray[int]
    is_pursuit: np.ndarray[bool]
    pursuit_intervals: np.ndarray[int]
    is_fixation: np.ndarray[bool]
    fixation_intervals: np.ndarray[int]
