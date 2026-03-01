# -*- coding: utf-8 -*-
import time

import numpy as np
import networkx as nx
from networkx.algorithms import tree
from scipy.spatial.distance import cdist

from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints
from vision_toolkit2.config import Config
from vision_toolkit2.config import IMST, Segmentation
from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config, segmentation_config, distance_type, verbose):
    """

    Parameters
    ----------
    s : Serie
        DESCRIPTION.
    config : Config
        DESCRIPTION.

    Returns
    -------
    BinarySegmentationResults
        DESCRIPTION.

    """
    assert distance_type == "euclidean", (
        "'distance_type' must be set to 'euclidean'"
    )

    if verbose:
        print("Processing MST Identification...")
        start_time = time.time()

    x_a = s.x
    y_a = s.y

    n_s = int(s.min_config.nb_samples)
    s_f = float(s.min_config.sampling_frequency)

    g_p = np.column_stack(
        (x_a.reshape(n_s), y_a.reshape(n_s))
    )  # (n_s, 2) array of gaze points

    vareps = float(segmentation_config.imst.distance_threshold)

    # Window length in samples
    t_du = int(np.ceil(float(segmentation_config.imst.window_duration) * s_f))
    t_du = max(2, t_du)  # need at least 2 points

    # Overlap / stride (B)
    # If user provides IMST_step_samples, use it; otherwise default to 50% overlap.
    step = segmentation_config.imst.step_samples
    if step is None:
        step = max(1, t_du // 2)
    else:
        step = max(1, int(step))

    # Minimum cluster size to accept as fixation-like (A)
    # Default: at least the minimum fixation duration in samples, capped by window length.
    min_pts = segmentation_config.imst.min_cluster_size
    if min_pts is None:
        min_pts = int(
            np.ceil(float(segmentation_config.filter.fixation_duration.min) * s_f)
        )
    min_pts = max(2, min(min_pts, t_du))

    # Build fixation mask via voting across overlapping windows
    fix_votes = np.zeros(n_s, dtype=np.int32)
    cover_votes = np.zeros(n_s, dtype=np.int32)

    i = 0
    while i < n_s:
        j = min(i + t_du, n_s)
        # If the remaining tail is too small, stop (nothing meaningful to MST)
        if (j - i) < 2:
            break

        w_gp = g_p[i:j]

        # Mark coverage for later normalisation
        cover_votes[i:j] += 1

        # Compute pairwise distances for this window
        d_m = cdist(w_gp, w_gp, metric="euclidean")

        # MST on dense graph
        g = nx.from_numpy_array(d_m, create_using=nx.Graph())
        mst = tree.minimum_spanning_tree(g, algorithm="prim")
        edgelist = mst.edges(data=True)

        # (A) Build thresholded graph from MST edges, then take connected components
        G_thr = nx.Graph()
        G_thr.add_nodes_from(range(j - i))
        for u, v, attr in edgelist:
            if attr["weight"] < vareps:
                G_thr.add_edge(u, v)

        # Mark nodes belonging to sufficiently large components as fixation-like
        for comp in nx.connected_components(G_thr):
            if len(comp) >= min_pts:
                idx = np.fromiter(comp, dtype=int)
                fix_votes[i + idx] += 1

        i += step

    # Convert votes to fixation mask:
    # - robust default: a sample is fixation if it was classified fixation-like
    #   in at least half of the windows that covered it.
    # If a sample was never covered (unlikely), it stays False.
    i_fix = np.zeros(n_s, dtype=bool)
    covered = cover_votes > 0
    i_fix[covered] = fix_votes[covered] >= np.ceil(0.5 * cover_votes[covered])

    if verbose:
        print("Done")

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
    fix_dur_t = int(np.ceil(segmentation_config.filter.fixation_duration.min * s_f))

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

    if verbose:
        print("\n...MST Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Keep track of index that were effectively labeled
    i_lab = np.array([False] * s.min_config.nb_samples)

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
    du_t = 40 / config.serie_metadata.sampling_frequency
    s_ = 0.001 * vf_diag

    imst_config = IMST(
        distance_threshold=s_,
        window_duration=du_t,
    )
    return Config(
        segmentation=Segmentation(imst_config),
    )
