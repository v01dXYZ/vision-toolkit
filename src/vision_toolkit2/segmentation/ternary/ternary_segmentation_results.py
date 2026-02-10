from dataclasses import dataclass
from vision_toolkit2.config import Config
from vision_toolkit2.oculomotor_series import Serie

from ..utils import interval_merging

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

    def filter_events_by_duration(
        self,
        fixation_duration_range,
        pursuit_duration_range,
    ):
        return self._filter_events_by_duration(
            self.config.nb_samples,
            self.config.sampling_frequency,
            self.fixation_intervals,
            self.pursuit_intervals,
            fixation_duration_range,
            pursuit_duration_range,
        )

    def _filter_events_by_duration(
        self,  # only used at the end to copy input/config
        nb_samples,
        sampling_frequency,
        fixation_intervals,
        pursuit_intervals,
        fixation_duration_range,
        pursuit_duration_range,
    ):
        min_fix_duration, max_fix_duration = fixation_duration_range
        min_pursuit_duration, max_pursuit_duration = pursuit_duration_range

        def _dur_samples(intv):
            return intv[1] - intv[0] + 1

        def _keep_by_duration(intervals, min_s, max_s, fs):
            min_n = int(np.ceil(min_s * fs))
            max_n = int(np.floor(max_s * fs))

            min_n = max(1, min_n)
            max_n = max(min_n, max_n)
            kept, rejected = [], []
            for itv in intervals:
                d = _dur_samples(itv)
                if (d >= min_n) and (d <= max_n):
                    kept.append(itv)
                else:
                    rejected.append(itv)
            return kept, rejected

        fs = float(sampling_frequency)

        fix_ints = fixation_intervals
        purs_ints = pursuit_intervals

        fix_kept, fix_bad = _keep_by_duration(
            fix_ints, min_fix_duration, max_fix_duration, fs
        )
        purs_kept, purs_bad = _keep_by_duration(
            purs_ints, min_pursuit_duration, max_pursuit_duration, fs
        )

        is_sac = np.ones(nb_samples, dtype=bool)
        is_fix = np.zeros(nb_samples, dtype=bool)
        is_purs = np.zeros(nb_samples, dtype=bool)

        for a, b in fix_kept:
            is_fix[a : b + 1] = True
            is_sac[a : b + 1] = False

        for a, b in purs_kept:
            is_purs[a : b + 1] = True
            is_sac[a : b + 1] = False

        # enforce exclusivity
        is_purs[is_fix] = False

        fix_out = interval_merging(np.where(is_fix)[0])
        purs_out = interval_merging(np.where(is_purs)[0])
        sac_out = interval_merging(np.where(is_sac)[0])

        return type(self)(
            is_saccade=is_sac,
            saccade_intervals=sac_out,
            is_pursuit=is_purs,
            pursuit_intervals=purs_out,
            is_fixation=is_fix,
            fixation_intervals=fix_out,
            input=self.input,
            config=self.config,
        )
