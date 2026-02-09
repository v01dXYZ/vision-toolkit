# -*- coding: utf-8 -*-

import time

import numpy as np

from ._optimized import baum_welch, Viterbi
from vision_toolkit2.config import Config
from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints
from ...binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config):
    if config.verbose:
        print("Processing HMM Identification...")
        start_time = time.time()

    a_s = s.absolute_speed

    i_low_vel = config.HMM_init_low_velocity
    i_high_vel = config.HMM_init_high_velocity
    i_var = config.HMM_init_variance
    n_iter = config.HMM_nb_iters
    s_f = config.sampling_frequency

    theta = baum_welch(a_s, 2, n_iter, i_low_vel, i_high_vel, i_var)

    s_s = Viterbi(theta[1], theta[3], theta[0])

    fix_s = int(np.argmin(theta[2]))

    wi_fix = np.where(s_s[:-1] == fix_s)[0]
    wi_fix = np.array(sorted(set(list(wi_fix) + list(wi_fix + 1))))

    i_fix = np.array([False] * config.nb_samples)
    i_fix[wi_fix] = True

    x_a = s.x
    y_a = s.y

    i_sac = (i_fix == False)
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config.min_sac_duration * s_f),
    )

    if config.verbose:
        print(
            "   Saccadic intervals identified with minimum duration: {s_du} sec".format(
                s_du=config.min_sac_duration
            )
        )

    i_fix = np.array([True] * config.nb_samples)
    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    fix_dur_t = int(np.ceil(config.min_fix_duration * s_f))
    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]
        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1] : s_int[0] + 1] = False

    if config.verbose:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=config.min_fix_duration
            )
        )

    wi_fix = np.where(i_fix == True)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config.min_fix_duration * s_f),
        max_int_size=np.ceil(config.max_fix_duration * s_f),
        status=s.status,
        proportion=config.status_threshold,
    )

    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    i_sac = (i_fix == False)
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config.min_sac_duration * s_f),
        status=s.status,
        proportion=config.status_threshold,
    )

    if config.verbose:
        print(
            "   Fixations ans saccades identified using availability status threshold: {s_th}".format(
                s_th=config.status_threshold
            )
        )

    assert len(f_ints) == len(ctrds), "Interval set and centroid set have different lengths"

    if config.verbose:
        print("\n...HMM Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    i_lab = np.array([False] * config.nb_samples)
    for f_int in f_ints:
        i_lab[f_int[0] : f_int[1] + 1] = True
    for s_int in s_ints:
        i_lab[s_int[0] : s_int[1] + 1] = True

    return BinarySegmentationResults(
        is_labeled=i_lab,
        fixation_intervals=f_ints,
        saccade_intervals=s_ints,
        fixation_centroids=ctrds,
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
