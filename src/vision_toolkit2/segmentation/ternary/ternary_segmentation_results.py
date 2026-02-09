from dataclasses import dataclass
from vision_toolkit2.config import Config
from vision_toolkit2.oculomotor_series import Serie

import numpy as np
import numpy.typing as npt


@dataclass
class TernarySegmentationResults:
    is_fixation: npt.NDArray[np.bool_]
    fixation_intervals: npt.NDArray[np.int_]
    is_saccade: npt.NDArray[np.bool_]
    saccade_intervals: npt.NDArray[np.int_]
    is_pursuit: npt.NDArray[np.bool_]
    pursuit_intervals: npt.NDArray[np.int_]

    input: Serie
    config: Config
