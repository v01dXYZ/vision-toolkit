# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging
from vision_toolkit2.config import Config

from ..ternary_segmentation_results import TernarySegmentationResults


def process_impl(s, config):
    """
    Adapted from Komogortsev & Karpov (2013).
    Modified I-VT algorithm, with a supplementary
    threshold to distinguish pursuits from fixations.
        - T_s = saccade velocity threshold.
        - T_p = saccade velocity threshold.
    """

    if config.verbose:
        print("Processing VVT Identification...")
        start_time = time.time()

    a_sp = s.absolute_speed

    T_s = config.IVVT_saccade_threshold
    T_p = config.IVVT_pursuit_threshold

    valid = np.isfinite(a_sp)

    is_saccade = (~valid) | (a_sp > T_s)
    is_pursuit = valid & (a_sp > T_p) & (a_sp <= T_s)
    is_fixation = valid & (a_sp <= T_p)

    saccade_intervals = interval_merging(np.where(is_saccade)[0])
    pursuit_intervals = interval_merging(np.where(is_pursuit)[0])
    fixation_intervals = interval_merging(np.where(is_fixation)[0])

    if config.verbose:
        print("\n...VVT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return TernarySegmentationResults(
        is_fixation=is_fixation,
        fixation_intervals=fixation_intervals,
        is_saccade=is_saccade,
        saccade_intervals=saccade_intervals,
        is_pursuit=is_pursuit,
        pursuit_intervals=pursuit_intervals,
        input=s,
        config=config,
    )


def default_config_impl(config, vf_diag):
    if config.distance_type == "euclidean":
        s_t = vf_diag * 0.5
        p_t = vf_diag * 0.15
        return Config(
            IVVT_saccade_threshold=s_t,
            IVVT_pursuit_threshold=p_t,
        )
    elif config.distance_type == "angular":
        return Config(
            IVVT_saccade_threshold=10,
            IVVT_pursuit_threshold=1,
        )
