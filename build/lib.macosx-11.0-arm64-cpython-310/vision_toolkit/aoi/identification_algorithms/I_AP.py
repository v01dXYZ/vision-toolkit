# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cluster import AffinityPropagation

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IAP(values, config, ref_image=None):
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

    center_method = config["AoI_IAP_centers"]

    ap = AffinityPropagation(max_iter=1000, damping=0.8).fit(pos_.T)
    seq_ = ap.labels_.copy()

    centers_ = {}
    clus_ = {}
    to_relabel = []
    i_ = 0

    # Use unique labels  
    for lab in np.unique(seq_):
        vals_ = np.where(seq_ == lab)[0]

        if len(vals_) >= 2:
            key = chr(i_ + 65)
            clus_[key] = vals_
            seq_[vals_] = i_  # compact labels [0..K-1]

            # Centers
            if center_method in ("mean", "raw_IAP"):
                centers_[key] = np.mean(pos_[:, vals_], axis=1)
            else:
                centers_[key] = np.mean(pos_[:, vals_], axis=1)

            i_ += 1
        else:
            to_relabel.extend(vals_.tolist())

    # Fallback: if everything is singleton
    if len(centers_) == 0:
        clus_ = {"A": np.arange(pos_.shape[1], dtype=int)}
        centers_ = {"A": np.mean(pos_, axis=1)}
        seq_ = np.zeros(pos_.shape[1], dtype=int)
        to_relabel = []

    # Relabel singletons to nearest center
    if to_relabel:
        keys = sorted(centers_.keys())
        centers_array = np.array([centers_[k] for k in keys])  # (K, 2)

        for val in to_relabel:
            pos_l = pos_[:, val]  # (2,)
            d_ = np.sum((centers_array - pos_l) ** 2, axis=1)
            c_val = int(np.argmin(d_))  # 0..K-1

            seq_[val] = c_val
            k = chr(c_val + 65)
            clus_[k] = np.sort(np.append(clus_[k], val))

    # Convert to AoI sequence (collapse / binning handled inside)
    seq_, seq_dur = compute_aoi_sequence(seq_, dur_, config)

    # Display
    if config["display_AoI"]:
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