# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import MeanShift

from vision_toolkit.utils.identification_utils import compute_aoi_sequence


def process_IMS(values, config, ref_image=None):
    """
    MeanShift-based AoI identification.

    config keys:
      - AoI_IMS_bandwidth
      - AoI_MS_cluster_all (bool, default True)

    Returns
    -------
    results : dict
    new_values : np.ndarray (3, n')
    """

    pos_ = values[0:2]
    dur_ = values[2]

    bandwidth = config["AoI_IMS_bandwidth"]
    cluster_all = bool(config.get("AoI_MS_cluster_all", True))
 
    ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)
    seq_all = ms.fit_predict(pos_.T).astype(int)
 
    if not cluster_all:
        keep = seq_all >= 0

        pos_f = pos_[:, keep]
        dur_f = dur_[keep]
        seq_f = seq_all[keep]

        # Fallback: everything unassigned
        if seq_f.size == 0:
            seq0 = np.zeros(pos_.shape[1], dtype=int)
            centers_ = {"A": np.mean(pos_, axis=1)}
            clus_ = {"A": np.arange(pos_.shape[1], dtype=int)}
            seq_letters, seq_dur = compute_aoi_sequence(seq0, dur_, config)
            new_values = np.vstack((pos_, dur_))
            return (
                {
                    "AoI_sequence": seq_letters,
                    "AoI_durations": seq_dur,
                    "centers": centers_,
                    "clustered_fixations": clus_,
                },
                new_values,
            )

        # Remap labels to 0..K-1
        uniq = np.unique(seq_f)
        remap = {lab: i for i, lab in enumerate(uniq)}
        seq_f = np.array([remap[l] for l in seq_f], dtype=int)

        centers_ = {}
        clus_ = {}
        K = len(uniq)

        for i in range(K):
            idxs = np.where(seq_f == i)[0]
            key = chr(i + 65)
            clus_[key] = idxs
            centers_[key] = np.mean(pos_f[:, idxs], axis=1)

        seq_letters, seq_dur = compute_aoi_sequence(seq_f, dur_f, config)
        new_values = np.vstack((pos_f, dur_f))

        return (
            {
                "AoI_sequence": seq_letters,
                "AoI_durations": seq_dur,
                "centers": centers_,
                "clustered_fixations": clus_,
            },
            new_values,
        )

    uniq = np.unique(seq_all)
    remap = {lab: i for i, lab in enumerate(uniq)}
    seq_ = np.array([remap[l] for l in seq_all], dtype=int)

    centers_ = {}
    clus_ = {}
    K = len(uniq)

    for i in range(K):
        idxs = np.where(seq_ == i)[0]
        key = chr(i + 65)
        clus_[key] = idxs
        centers_[key] = np.mean(pos_[:, idxs], axis=1)

    seq_letters, seq_dur = compute_aoi_sequence(seq_, dur_, config)
    new_values = np.vstack((pos_, dur_))

    return (
        {
            "AoI_sequence": seq_letters,
            "AoI_durations": seq_dur,
            "centers": centers_,
            "clustered_fixations": clus_,
        },
        new_values,
    )
