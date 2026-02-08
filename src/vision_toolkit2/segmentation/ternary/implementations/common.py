def build_results_from_indicators(
        is_fixation
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
        is_saccade=is_saccade == 1,
        saccade_intervals=interval_merging_from_indicator(is_saccade),
        is_pursuit=is_pursuit == 1,
        pursuit_intervals=interval_merging_from_indicator(is_pursuit),
        is_fixation=is_fixation == 1,
        fixation_intervals=interval_merging_from_indicator(is_fixation),
    )

def build_results_from_ordinal(
        ordinal,
        fixation_ordinal=0,
        saccade_ordinal=1,
        pursuit_ordinal=2,
        config=None,
):
    return build_results_from_ordinal(
        ordinal == fixation_ordinal,
        ordinal == saccade_ordinal,
        ordinal == pursuit_ordinal,
        config = config,
    )
