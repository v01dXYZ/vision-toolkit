# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit2.segmentation.utils import interval_merging
from vision_toolkit2.config import Config
from vision_toolkit2.config import IVVT, Segmentation

from ..ternary_segmentation_results import TernarySegmentationResults


def process_impl(s, segmentation_config, distance_type, verbose):
    """
    Adapted from Komogortsev & Karpov (2013).
    Modified I-VT algorithm, with a supplementary
    threshold to distinguish pursuits from fixations.
        - T_s = saccade velocity threshold.
        - T_p = saccade velocity threshold.
    """

    if verbose:
        print("Processing VVT Identification...")
        start_time = time.time()

    a_sp = s.absolute_speed

    T_s = segmentation_config.ivvt.saccade_threshold
    T_p = segmentation_config.ivvt.pursuit_threshold

    valid = np.isfinite(a_sp)

    is_saccade = (~valid) | (a_sp > T_s)
    is_pursuit = valid & (a_sp > T_p) & (a_sp <= T_s)
    is_fixation = valid & (a_sp <= T_p)

    saccade_intervals = interval_merging(np.where(is_saccade)[0])
    pursuit_intervals = interval_merging(np.where(is_pursuit)[0])
    fixation_intervals = interval_merging(np.where(is_fixation)[0])

    if verbose:
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
        config=segmentation_config,
    )


def default_config_impl(config, vf_diag):
    if config.distance_type == "euclidean":
        s_t = vf_diag * 0.5
        p_t = vf_diag * 0.15
        ivvt_config = IVVT(
            saccade_threshold=s_t,
            pursuit_threshold=p_t,
        )
        return Config(
            segmentation=Segmentation(ivvt_config),
        )
    elif config.distance_type == "angular":
        ivvt_config = IVVT(
            saccade_threshold=10,
            pursuit_threshold=1,
        )
        return Config(
            segmentation=Segmentation(ivvt_config),
        )
