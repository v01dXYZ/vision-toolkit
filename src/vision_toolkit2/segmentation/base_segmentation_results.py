from vision_toolkit2 import config as c
from vision_toolkit2.serie import Serie

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BaseSegmentationResults:
    input: Serie
    config: c.Segmentation

    fixation_intervals: npt.NDArray[np.int_]
    saccade_intervals: npt.NDArray[np.int_]
