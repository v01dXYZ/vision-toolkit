# -*- coding: utf-8 -*-

import time

from . import _optimized
from vision_toolkit2.config import Config
from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config):
    if config.verbose:
        print("Processing HMM Identification...")
        start_time = time.time()

    out = _optimized.process_IHMM(s, config)

    if config.verbose:
        print("\n...HMM Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return BinarySegmentationResults(
        is_labeled=out['is_labeled'],
        fixation_intervals=out['fixation_intervals'],
        saccade_intervals=out['saccade_intervals'],
        fixation_centroids=out['centroids'],
        input=s,
        config=config,
    )


def default_config_impl(config, vf_diag):
    i_l = 0.001 * vf_diag
    i_h = 10.0 * vf_diag
    i_v = 100 * vf_diag**2

    return Config(
        HMM_init_low_velocity=i_l,
        HMM_init_high_velocity=i_h,
        HMM_init_variance=i_v,
        HMM_nb_iters=10,
    )
