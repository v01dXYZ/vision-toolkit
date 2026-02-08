# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging

from ..ternary_segmentation_results import TernarySegmentationResults
from .common import build_results_from_indicators

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

    # Eye movement parameters
    a_sp = s.absolute_speed

    # Algorithm parameters
    T_s = config.IVVT_saccade_threshold
    T_p = config.IVVT_pursuit_threshold

    # Saccades are found like in the I-VT algorithm
    i_sac = np.where(a_sp > T_s, 1, 0)

    # An additional threshold is used for pursuits
    i_purs = np.where((a_sp > T_p) & (a_sp <= T_s), 1, 0)
    
    # The remaining points are fixations
    i_fix = np.where(a_sp <= T_p, 1, 0)
    print(a_sp)
    if config.verbose:
        print("\n...VVT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return build_results_from_indicators(
        i_fix,
        i_sac,
        i_purs,
    )
