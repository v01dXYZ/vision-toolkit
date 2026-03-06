from vision_toolkit2 import config as c
from vision_toolkit2.serie import Serie

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from typing import Self


@dataclass
class BaseSegmentationResults:
    input: Serie
    config: c.Segmentation

    fixation_intervals: npt.NDArray[np.int_]
    saccade_intervals: npt.NDArray[np.int_]

    filtered_from: Self | None = field(default=None, kw_only=True)

    def filter_events(self, filter=None):
        filter = filter or self.config.filter

        return self._filter_events(filter)

    def _filter_events(self, filter):
        return self
