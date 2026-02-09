# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging
from vision_toolkit2.config import Config

from ..ternary_segmentation_results import TernarySegmentationResults


def process_impl(s, config):
    """
    Identifies saccades like the I-VT algorithm.
    Distinguishes pursuits from fixations using the movement
    pattern of the eye trace.
        - T_s = saccade velocity threshold.
        - n_w = temporal window size.
        – T_m = movement threshold.
    """

    if config.verbose:
        print("Processing VMP Identification...")
        start_time = time.time()

    a_sp = s.absolute_speed
    n_s = config.nb_samples
    s_f = config.sampling_frequency

    x_array = s.x
    y_array = s.y

    t_s = config.IVMP_saccade_threshold
    t_du = int(np.ceil(config.IVMP_window_duration * s_f))
    t_du = max(2, t_du)
    t_r = config.IVMP_rayleigh_threshold

    is_sac = a_sp > t_s
    is_fix = ~is_sac
    is_purs = np.zeros(n_s, dtype=bool)

    wi_intersac = np.where(~is_sac)[0]
    inter_ints = interval_merging(wi_intersac)

    dx = np.empty(n_s)
    dy = np.empty(n_s)
    dx[:-1] = x_array[1:] - x_array[:-1]
    dy[:-1] = y_array[1:] - y_array[:-1]
    dx[-1] = dx[-2]
    dy[-1] = dy[-2]

    suc_dir = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    for a, b in inter_ints:
        i = a
        while i <= b:
            j = min(i + t_du, b + 1)
            if (j - i) < 2:
                break

            pos_unitary_circle = np.array([np.cos(suc_dir[i:j]), np.sin(suc_dir[i:j])])
            rm_vec = np.sum(pos_unitary_circle, axis=1) / (j - i)
            z_score = np.linalg.norm(rm_vec) ** 2

            if z_score > t_r:
                is_purs[i:j] = True
                is_fix[i:j] = False

            i = j

    is_purs = is_purs & (~is_sac)
    is_fix = (~is_sac) & (~is_purs)

    saccade_intervals = interval_merging(np.where(is_sac)[0])
    pursuit_intervals = interval_merging(np.where(is_purs)[0])
    fixation_intervals = interval_merging(np.where(is_fix)[0])

    if config.verbose:
        print("\n...VMP Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return TernarySegmentationResults(
        is_fixation=is_fix,
        fixation_intervals=fixation_intervals,
        is_saccade=is_sac,
        saccade_intervals=saccade_intervals,
        is_pursuit=is_purs,
        pursuit_intervals=pursuit_intervals,
        input=s,
        config=config,
    )


def default_config_impl(config, vf_diag):
    if config.distance_type == "euclidean":
        s_t = vf_diag * 0.5
        return Config(
            IVMP_saccade_threshold=s_t,
            IVMP_rayleigh_threshold=0.50,
            IVMP_window_duration=0.050,
        )
    elif config.distance_type == "angular":
        return Config(
            IVMP_saccade_threshold=40,
            IVMP_rayleigh_threshold=0.50,
            IVMP_window_duration=0.050,
        )
