# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import DBSCAN

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IDT(values, config, ref_image=None):
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

    eps = config["AoI_IDT_density_threshold"]
    min_samples = config["AoI_IDT_min_samples"]

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_.T)
    labels = dbscan.labels_.copy()

    # Keep non-noise points only
    keep = labels >= 0

    # ---- Fallback: everything is noise ----
    if not np.any(keep):
        # One AoI containing all fixations (keeps pipeline alive)
        seq_ = np.zeros(pos_.shape[1], dtype=int)
        centers_ = {"A": np.mean(pos_, axis=1)}
        clus_ = {"A": np.arange(pos_.shape[1], dtype=int)}

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

    # Filter to clustered fixations
    pos_f = pos_[:, keep]
    dur_f = dur_[keep]
    lab_f = labels[keep]

    # Remap labels to contiguous 0..K-1
    uniq = np.unique(lab_f)
    remap = {lab: i for i, lab in enumerate(uniq)}
    seq_ = np.array([remap[lab] for lab in lab_f], dtype=int)

    # Build clusters + centers
    centers_ = {}
    clus_ = {}
    for i in range(len(uniq)):
        vals_ = np.where(seq_ == i)[0]
        key = chr(i + 65)
        clus_[key] = vals_
        centers_[key] = np.mean(pos_f[:, vals_], axis=1)

    # Final AoI sequence (collapse/binning inside if configured)
    seq_, seq_dur = compute_aoi_sequence(seq_, dur_f, config)

    if config.get("display_AoI", False):
        if ref_image is None:
            display_aoi_identification(pos_f, clus_, config)
        else:
            display_aoi_identification_reference_image(pos_f, clus_, config, ref_image)

    return {
        "AoI_sequence": seq_,
        "AoI_durations": seq_dur,
        "centers": centers_,
        "clustered_fixations": clus_,
    }