from vision_toolkit2.config import Config
from vision_toolkit2.oculomotor_series import Serie

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BinarySegmentationResults:
    is_labeled: np.ndarray[bool]
    fixation_intervals: np.ndarray[int]
    saccade_intervals: np.ndarray[int]
    fixation_centroids: np.ndarray[int]

    input: Serie
    config: Config
