# -*- coding: utf-8 -*-
import time

import numpy as np

from vision_toolkit2.config import Config, IDeT, Segmentation
from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints
from ...binary_segmentation_results import BinarySegmentationResults
from ._optimized import vareps_neighborhood, expand_cluster


def process_impl(s, segmentation_config, distance_type, verbose):
    """
    I-DeT algorithm.
    """

    if verbose:
        print("Processing DeT Identification...")
        start_time = time.time()

    n_s = s.min_config.nb_samples
    s_f = s.min_config.sampling_frequency

    euclidean = distance_type == "euclidean"

    if euclidean:
        g_npts = np.concatenate(
            (s.x.reshape(1, n_s), s.y.reshape(1, n_s), s.z.reshape(1, n_s)),
            axis=0,
            dtype="float64",
        )
    else:
        g_npts = s.unitary_gaze_vectors.astype("float64")

    d_t = segmentation_config.i_det.density_threshold
    win_w = int(np.ceil(segmentation_config.i_det.duration_threshold * s_f))
    win_w = max(1, win_w)

    min_pts = int(np.ceil(segmentation_config.i_det.min_pts * s_f))
    min_pts = max(2, min_pts)

    assert min_pts <= win_w, (
        "Invalid I-DeT parameters: IDeT_min_pts (s) must not exceed "
        "IDeT_duration_threshold (s)"
    )

    C_clus = []
    avlb = {i: int(1) for i in range(0, n_s)}

    for i in range(n_s):
        if avlb[i]:
            neigh = vareps_neighborhood(g_npts, euclidean, n_s, i, d_t, win_w)

            if len(neigh) + 1 >= min_pts:
                avlb[i] = False
                l_C_clus, avlb = expand_cluster(
                    g_npts, euclidean, n_s, i, neigh, d_t, win_w, min_pts, avlb
                )

                if len(l_C_clus) >= min_pts:
                    C_clus.append(l_C_clus)

    i_fix = np.zeros(n_s, dtype=np.bool_)
    for clus in C_clus:
        for idx in clus:
            i_fix[idx] = True

    wi_fix = np.where(i_fix)[0]

    x_a = s.x
    y_a = s.y

    i_sac = ~i_fix
    wi_sac = np.where(i_sac)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(segmentation_config.filter.saccade_duration.min * s_f),
    )

    if verbose:
        print(
            "   Saccadic intervals identified with minimum duration: {s_du} sec".format(
                s_du=segmentation_config.filter.saccade_duration.min
            )
        )

    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True] * s.min_config.nb_samples)

    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = max(
        1, int(np.ceil(segmentation_config.filter.fixation_duration.min * s_f))
    )

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        gap = s_int[0] - o_s_int[1] - 1
        if 0 <= gap < fix_dur_t:
            i_fix[o_s_int[1] + 1 : s_int[0]] = False

    if verbose:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=segmentation_config.filter.fixation_duration.min
            )
        )

    # Recompute fixation intervals
    wi_fix = np.where(i_fix)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(segmentation_config.filter.fixation_duration.min * s_f),
        max_int_size=np.ceil(segmentation_config.filter.fixation_duration.max * s_f),
        status=s.status,
        proportion=segmentation_config.filter.status_threshold,
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    # Recompute saccadic intervals
    i_sac = ~i_fix
    wi_sac = np.where(i_sac)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(segmentation_config.filter.saccade_duration.min * s_f),
        status=s.status,
        proportion=segmentation_config.filter.status_threshold,
    )

    if verbose:
        print(
            "   Fixations ans saccades identified using availability status threshold: {s_th}".format(
                s_th=segmentation_config.filter.status_threshold
            )
        )

    assert len(f_ints) == len(ctrds), (
        "Interval set and centroid set have different lengths"
    )

    # Keep track of index that were effectively labeled
    i_lab = np.array([False] * s.min_config.nb_samples)

    for f_int in f_ints:
        i_lab[f_int[0] : f_int[1] + 1] = True

    for s_int in s_ints:
        i_lab[s_int[0] : s_int[1] + 1] = True

    if verbose:
        print("\n...DeT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return BinarySegmentationResults(
        is_labeled=i_lab,
        fixation_intervals=f_ints,
        saccade_intervals=s_ints,
        fixation_centroids=ctrds,
        input=s,
        config=segmentation_config,
    )


def default_config_impl(config, vf_diag):
    du_t = 5 / config.serie_metadata.sampling_frequency
    nb_t = 3 / config.serie_metadata.sampling_frequency
    if config.distance_type == "euclidean":
        de_t = vf_diag / config.serie_metadata.sampling_frequency
        return Config(
            segmentation=Segmentation(
                IDeT(
                    duration_threshold=du_t,
                    density_threshold=de_t,
                    min_pts=nb_t,
                )
            )
        )
    elif config.distance_type == "angular":
        de_t = 30 / config.serie_metadata.sampling_frequency
        return Config(
            segmentation=Segmentation(
                IDeT(
                    duration_threshold=du_t,
                    density_threshold=de_t,
                    min_pts=nb_t,
                )
            )
        )
