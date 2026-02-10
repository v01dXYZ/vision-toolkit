import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging

from ..ternary_segmentation_results import TernarySegmentationResults


def build_results_from_indicators(
    is_fixation,
    is_saccade,
    is_pursuit,
    config=None,
):
    def interval_merging_from_indicator(i):
        idx = np.where(i)[0]

        if config is not None:
            return interval_merging(
                idx,
                min_int_size=config.min_int_size,
            )

        return interval_merging(idx)

    return TernarySegmentationResults(
        is_fixation=is_fixation == 1,
        fixation_intervals=interval_merging_from_indicator(is_fixation),
        is_saccade=is_saccade == 1,
        saccade_intervals=interval_merging_from_indicator(is_saccade),
        is_pursuit=is_pursuit == 1,
        pursuit_intervals=interval_merging_from_indicator(is_pursuit),
        input=None,
        config=None,
    )


def build_results_from_ordinal(
    ordinal,
    fixation_ordinal=0,
    saccade_ordinal=1,
    pursuit_ordinal=2,
    config=None,
):
    return TernarySegmentationResults(
        is_fixation=ordinal == fixation_ordinal,
        fixation_intervals=interval_merging(np.where(ordinal == fixation_ordinal)[0]),
        is_saccade=ordinal == saccade_ordinal,
        saccade_intervals=interval_merging(np.where(ordinal == saccade_ordinal)[0]),
        is_pursuit=ordinal == pursuit_ordinal,
        pursuit_intervals=interval_merging(np.where(ordinal == pursuit_ordinal)[0]),
        input=None,
        config=None,
    )
