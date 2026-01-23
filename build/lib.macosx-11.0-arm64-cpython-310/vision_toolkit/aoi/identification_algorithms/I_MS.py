# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cluster import MeanShift

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IMS(values, config, ref_image=None):
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
    bandwidth = config["AoI_IMS_bandwidth"]

    ms = MeanShift(bandwidth=bandwidth, cluster_all=True).fit(pos_.T)
    seq_ = ms.labels_.astype(int)

    # Build clusters from unique labels (robust)
    labels = np.unique(seq_)

    # Identify singletons to relabel
    clus_tmp = {lab: np.where(seq_ == lab)[0] for lab in labels}
    multi_labels = [lab for lab, idxs in clus_tmp.items() if len(idxs) >= 2]
    single_idxs = np.concatenate(
        [idxs for lab, idxs in clus_tmp.items() if len(idxs) < 2],
        axis=0
    ) if any(len(idxs) < 2 for idxs in clus_tmp.values()) else np.array([], dtype=int)

    # Fallback: everything is singleton
    if len(multi_labels) == 0:
        seq_ = np.zeros(pos_.shape[1], dtype=int)
        clus_ = {"A": np.arange(pos_.shape[1], dtype=int)}
        centers_ = {"A": np.mean(pos_, axis=1)}
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

    # Compute centers for multi-point clusters
    centers_by_lab = {
        lab: np.mean(pos_[:, clus_tmp[lab]], axis=1) for lab in multi_labels
    }
    multi_centers = np.array([centers_by_lab[lab] for lab in multi_labels])

    # Relabel singleton points to nearest multi cluster
    if single_idxs.size > 0:
        for idx in single_idxs:
            p = pos_[:, idx]
            d = np.sum((multi_centers - p) ** 2, axis=1)
            nearest_lab = multi_labels[int(np.argmin(d))]
            seq_[idx] = nearest_lab

    # Remap labels to contiguous 0..K-1 and A,B,C...
    final_labels = np.unique(seq_)
    remap = {lab: i for i, lab in enumerate(final_labels)}
    seq_ = np.array([remap[lab] for lab in seq_], dtype=int)

    clus_ = {}
    centers_ = {}
    for i, lab in enumerate(final_labels):
        idxs = np.where(np.array([lab2 for lab2 in np.array(list(remap.keys()))]) == lab)[0]  # not used; ignore

    # Build final clusters by index
    for i in range(len(final_labels)):
        idxs = np.where(seq_ == i)[0]
        key = chr(i + 65)
        clus_[key] = idxs
        centers_[key] = np.mean(pos_[:, idxs], axis=1)

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