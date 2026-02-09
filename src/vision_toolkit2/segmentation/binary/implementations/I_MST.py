# -*- coding: utf-8 -*-

import time

import numpy as np
import networkx as nx
from networkx.algorithms import tree
from scipy.spatial.distance import cdist

from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints
from vision_toolkit2.config import Config
from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config):
    assert (
        config.distance_type == "euclidean"
    ), "'Distance type' must be set to 'euclidean"

    if config.verbose:
        print("Processing MST Identification...")
        start_time = time.time()

    x_a = s.x
    y_a = s.y

    n_s = config.nb_samples
    s_f = config.sampling_frequency

    g_p = np.concatenate((x_a.reshape(n_s, 1), y_a.reshape(n_s, 1)), axis=1)

    vareps = config.IMST_distance_threshold

    t_du = int(np.ceil(config.IMST_window_duration * s_f))

    i_fix = np.array([False] * n_s)

    i = 0
    while i + t_du < n_s:
        j = i + t_du

        w_gp = g_p[i:j]

        d_m = cdist(w_gp, w_gp, metric="euclidean")

        g = nx.from_numpy_array(d_m, create_using=nx.MultiGraph())
        mst = tree.minimum_spanning_tree(g, algorithm="prim")

        edgelist = mst.edges(data=True)

        for edge in edgelist:
            w_ = edge[2]["weight"]

            i_mst = edge[0]
            j_mst = edge[1]

            if w_ < vareps:
                i_fix[i + i_mst] = True
                i_fix[i + j_mst] = True

        i = j

    if config.verbose:
        print("Done")

    wi_fix = np.where(i_fix == True)[0]

    i_sac = i_fix == False
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
        status=s.status,
        proportion=config.status_threshold,
    )

    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    i_sac = i_fix == False
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

    assert len(f_ints) == len(
        ctrds
    ), "Interval set and centroid set have different lengths"

    if config.verbose:
        print("\n...MST Identification done\n")
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
    return Config(
        IMST_distance_threshold=0.5,
        IMST_window_duration=0.040,
    )
