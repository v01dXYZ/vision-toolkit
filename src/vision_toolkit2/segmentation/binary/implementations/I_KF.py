# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints
from vision_toolkit2.velocity_distance_factory import process_speed_components
from vision_toolkit2.config import Config
from vision_toolkit2.config import IKF, Segmentation
from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config):
    assert config.distance_type == "euclidean", (
        "'Distance type' must be set to 'euclidean"
    )

    if config.verbose:
        print("Processing KF Identification...")
        start_time = time.time()

    n_s = config.serie_metadata.nb_samples
    s_f = config.serie_metadata.sampling_frequency

    d_t = 1 / s_f
    c_wn = config.segmentation.ikf.chi2_window

    x_a = s.x
    y_a = s.y

    pos = np.concatenate((x_a.reshape(1, n_s), y_a.reshape(1, n_s)), axis=0)

    sp = process_speed_components(s, config)[0:2, :]

    # predict velocities and positions from Kalman filter
    pred = process_Kalman_filter(
        pos, sp, d_t, config.segmentation.ikf.sigma_1, config.segmentation.ikf.sigma_2
    )

    # compute norms of predicted and true speed vectors
    p_sp = np.linalg.norm(
        np.concatenate(
            (pred["x"][1, :].reshape(1, n_s), pred["y"][1, :].reshape(1, n_s)), axis=0
        ),
        axis=0,
    )

    t_sp = np.linalg.norm(sp, axis=0)

    # compute chi2 statistics over sampling intervals of size c_wn
    chi2_a = compute_chi2(p_sp, t_sp, c_wn)

    wi_fix = np.where(chi2_a[:-1] <= config.segmentation.ikf.chi2_threshold)[0]

    # Add index + 1 to fixation since velocities are computed from two data points
    wi_fix = np.array(sorted(set(list(wi_fix) + list(wi_fix + 1))))

    i_fix = np.array([False] * config.serie_metadata.nb_samples)
    i_fix[wi_fix] = True

    i_sac = ~i_fix
    wi_sac = np.where(i_sac)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config.segmentation.filter.saccade_duration.min * s_f),
    )

    # i_sac events not retained as intervals are relabeled as fix events
    if config.verbose:
        print(
            "   Saccadic intervals identified with minimum duration: {s_du} sec".format(
                s_du=config.segmentation.filter.saccade_duration.min
            )
        )

    i_fix = np.array([True] * config.serie_metadata.nb_samples)

    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config.segmentation.filter.fixation_duration.min * s_f))

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1] : s_int[0] + 1] = False

    if config.verbose:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=config.segmentation.filter.fixation_duration.min
            )
        )

    wi_fix = np.where(i_fix)[0]

    # Recompute fixation intervals
    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config.segmentation.filter.fixation_duration.min * s_f),
        status=s.status,
        proportion=config.segmentation.filter.status_threshold,
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    i_sac = ~i_fix
    wi_sac = np.where(i_sac)[0]

    # Recompute saccadic intervals
    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config.segmentation.filter.saccade_duration.min * s_f),
        status=s.status,
        proportion=config.segmentation.filter.status_threshold,
    )

    if config.verbose:
        print(
            "   Fixations ans saccades identified using availability status threshold: {s_th}".format(
                s_th=config.segmentation.filter.status_threshold
            )
        )

    assert len(f_ints) == len(ctrds), (
        "Interval set and centroid set have different lengths"
    )

    if config.verbose:
        print("\n...KF Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    i_lab = np.array([False] * config.serie_metadata.nb_samples)

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


def compute_chi2(p_sp, t_sp, c_wn):
    chi2_a = np.zeros_like(p_sp)
    n_s = len(p_sp)

    i = 0

    while (i + c_wn) < n_s:
        j = i + c_wn

        c2_stat = np.sum((p_sp[i:j] - t_sp[i:j]) ** 2) / 5
        chi2_a[i:j] = c2_stat

        i = j

    return chi2_a


def process_Kalman_filter(pos, sp, d_t, sigma_1, sigma_2):
    n_s = np.shape(pos)[1]
    results = dict({})

    for k, _dir in enumerate(["x", "y"]):
        u_v = np.zeros((2, n_s))
        u_v_pl = np.zeros((2, n_s))

        u_v[:, 0] = u_v_pl[:, 0] = np.array([pos[k, 0], sp[k, 0]])

        pos_v = pos[k, :]

        a_m = np.array([[1, d_t], [0, 1]])
        h_m = np.array([[1, 0]])

        sigma_m_1 = np.diag(np.array([sigma_1] * 2) ** 2)
        sigma_m_2 = sigma_2**2

        p_m_pl = np.zeros((2, 2))

        for i in range(1, n_s):
            u_v[:, i] = a_m @ u_v_pl[:, i - 1]
            p_m = a_m @ p_m_pl @ a_m.T + sigma_m_1

            k_m = p_m @ h_m.T * (h_m @ p_m @ h_m.T + sigma_m_2) ** (-1)
            u_v_pl[:, i] = (
                u_v[:, i].reshape(2, 1)
                + k_m * (pos_v[i] - h_m @ u_v[:, i].reshape(2, 1))
            ).reshape(2)
            p_m_pl = (1 - k_m @ h_m) * p_m

        results.update({_dir: u_v})

    return results


def default_config_impl(config, vf_diag):
    si_1 = (4 * vf_diag) ** 2
    si_2 = (5 * vf_diag) ** 2
    c_t = (1 * vf_diag) ** 2

    ikf_config = IKF(
        chi2_threshold=c_t,
        chi2_window=10,
        chi2_sigma=1.0,
        sigma_1=si_1,
        sigma_2=si_2,
    )
    return Config(
        segmentation=Segmentation(ikf_config),
    )
