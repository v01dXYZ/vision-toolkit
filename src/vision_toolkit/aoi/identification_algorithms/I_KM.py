# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IKM(values, config, ref_image=None):
    """
    

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.
    ref_image : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    pos_ = values[0:2]
    dur_ = values[2]

    n_st = config["AoI_IKM_cluster_number"]
    n_samples = pos_.shape[1]

    # ---- Safety for very small sequences ----
    if n_samples <= 1:
        seq_ = np.zeros(n_samples, dtype=int)
        centers_ = {"A": pos_[:, 0] if n_samples == 1 else np.array([0.0, 0.0])}
        clus_ = {"A": np.arange(n_samples, dtype=int)}
        seq_, seq_dur = compute_aoi_sequence(seq_, dur_, config)
        return {
            "AoI_sequence": seq_,
            "AoI_durations": seq_dur,
            "centers": centers_,
            "clustered_fixations": clus_,
        }

    if n_st == "search":
        k_min = int(config["AoI_IKM_min_clusters"])
        k_max = int(config["AoI_IKM_max_clusters"])

        # silhouette requires: 2 <= k <= n_samples-1
        k_min = max(2, k_min)
        k_max = min(k_max, n_samples - 1)

        # If search range is invalid, fall back to k=2 (or 1 if impossible)
        if k_max < k_min:
            n_st = 2 if n_samples >= 2 else 1
            kmeans = KMeans(n_clusters=n_st, random_state=0, n_init="auto").fit(pos_.T)
        else:
            best_k = None
            best_sc = -np.inf
            best_model = None

            for k in range(k_min, k_max + 1):  # include k_max
                model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(pos_.T)
                try:
                    sc = silhouette_score(pos_.T, model.labels_)
                except Exception:
                    continue

                if sc > best_sc:
                    best_sc = sc
                    best_k = k
                    best_model = model

            # Fallback if every silhouette computation failed
            if best_model is None:
                n_st = k_min
                kmeans = KMeans(n_clusters=n_st, random_state=0, n_init="auto").fit(pos_.T)
            else:
                n_st = best_k
                kmeans = best_model

    else:
        n_st = int(n_st)
        # Safety: k must be <= n_samples
        n_st = min(max(1, n_st), n_samples)
        kmeans = KMeans(n_clusters=n_st, random_state=0, n_init="auto").fit(pos_.T)

    seq_ = kmeans.labels_.astype(int)

    centers_ = {}
    clus_ = {}

    for i in range(int(n_st)):
        vals_ = np.where(seq_ == i)[0]
        key = chr(i + 65)
        clus_[key] = vals_
        # If a cluster is empty (rare, but safe), use kmeans center
        if vals_.size == 0:
            centers_[key] = kmeans.cluster_centers_[i]
        else:
            centers_[key] = np.mean(pos_[:, vals_], axis=1)

    seq_, seq_dur = compute_aoi_sequence(seq_, dur_, config)

    if config.get("display_AoI", False):
        if ref_image is None:
            display_aoi_identification(pos_, clus_, config)
        else:
            display_aoi_identification_reference_image(pos_, clus_, config, ref_image)

    return {
        "AoI_sequence": seq_,
        "AoI_durations": seq_dur,
        "centers": centers_,
        "clustered_fixations": clus_,
    }