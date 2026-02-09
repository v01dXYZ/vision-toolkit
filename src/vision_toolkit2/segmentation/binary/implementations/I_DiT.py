# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit2.segmentation.utils import (
    centroids_from_ints,
    dispersion_metric,
    interval_merging,
)
from vision_toolkit2.config import Config
from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(
    s,
    config,
):
    if config.verbose:
        print("Processing DiT Identification...")
        start_time = time.time()

    if config.distance_type == "euclidean":
        x_a = s.x
        y_a = s.y

    if config.distance_type == "angular":
        theta_coord = s.theta_coord
        x_a = theta_coord[0, :]
        y_a = theta_coord[1, :]

    n_s = config.nb_samples
    s_f = config.sampling_frequency

    t_du = int(np.ceil(config.I_DiT_window_duration * s_f))
    t_di = config.I_DiT_dispersion_threshold

    i_fix = np.array([False] * config.nb_samples)

    i = 0
    while i + t_du < n_s:
        j = min(i + t_du, n_s)
        if j - i < 2:
            break
        d = dispersion_metric(x_a[i:j], y_a[i:j])

        if d < t_di:
            while d < t_di and j < n_s - 1:
                j += 1
                d = dispersion_metric(x_a[i:j], y_a[i:j])

            i_fix[i:j] = True
            i = j

        else:
            i += 1

    i_sac = ~i_fix
    wi_sac = np.where(i_sac)[0]

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

    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True] * config.nb_samples)

    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config.min_fix_duration * s_f))

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        gap = s_int[0] - o_s_int[1] - 1
        if 0 <= gap < fix_dur_t:
            i_fix[o_s_int[1] + 1 : s_int[0]] = False

    if config.verbose:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=config.min_fix_duration
            )
        )

    # Recompute fixation intervals
    wi_fix = np.where(i_fix)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config.min_fix_duration * s_f),
        max_int_size=np.ceil(config.max_fix_duration * s_f),
        status=s.status,
        proportion=config.status_threshold,
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    # Recompute saccadic intervals
    i_sac = ~i_fix
    wi_sac = np.where(i_sac)[0]

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

    assert len(f_ints) == len(ctrds), (
        "Interval set and centroid set have different lengths"
    )

    if config.verbose:
        print("\n...DiT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Keep track of index that were effectively labeled
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
    if config.distance_type == "euclidean":
        di_t = 0.01 * vf_diag

        return Config(
            I_DiT_window_duration=0.040,
            I_DiT_dispersion_threshold=di_t,
        )
    elif config.distance_type == "angular":
        return Config(
            I_DiT_window_duration=0.040,
            I_DiT_dispersion_threshold=0.3,
        )
