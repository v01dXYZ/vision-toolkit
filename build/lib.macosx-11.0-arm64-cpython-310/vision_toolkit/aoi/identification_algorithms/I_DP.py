# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gmean, norm

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IDP(values, config, ref_image=None):
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
    n_s = len(dur_)

    k_sd = float(config["AoI_IDP_gaussian_kernel_sd"])
    center_method = config["AoI_IDP_centers"]

    # Safety: avoid zero/near-zero kernel width
    k_sd = max(k_sd, 1e-9)

    dist_ = cdist(pos_.T, pos_.T)
    max_d = float(np.max(dist_))

    # rho: vectorized gaussian kernel density (excluding self-term)  
    # norm.pdf(dist_, 0, k_sd) gives (n,n). subtract the self pdf once per row.
    pdf_mat = norm.pdf(dist_, loc=0.0, scale=k_sd)
    self_pdf = norm.pdf(0.0, loc=0.0, scale=k_sd)
    rho = np.sum(pdf_mat, axis=1) - self_pdf

    max_r = float(np.max(rho))

    # delta: distance to nearest point of higher density 
    delta = np.empty(n_s, dtype=float)
    for i in range(n_s):
        if rho[i] < max_r:
            idxs = np.where(rho > rho[i])[0]
            # idxs is non-empty because max_r exists
            delta[i] = float(np.min(dist_[i, idxs]))
        else:
            delta[i] = max_d

    gamma = rho * delta

    thresh = compute_threshold(gamma)

    # ---- centers: points with gamma above threshold ----
    center_idx = np.where(gamma > thresh)[0]

    # Fallback if no centers selected:
    # pick at least one center = argmax(gamma)
    if center_idx.size == 0:
        center_idx = np.array([int(np.argmax(gamma))], dtype=int)

    # Assign each point to nearest center
    dist_f = dist_[:, center_idx]  # (n, k)
    seq_ = np.argmin(dist_f, axis=1).astype(int)

    centers_ = {}
    clus_ = {}

    for i in range(len(center_idx)):
        vals_ = np.where(seq_ == i)[0]
        clus_[chr(i + 65)] = vals_

    if center_method == "mean":
        for i in range(len(center_idx)):
            key = chr(i + 65)
            # handle any empty cluster just in case
            if clus_[key].size == 0:
                centers_[key] = pos_[:, center_idx[i]]
            else:
                centers_[key] = np.mean(pos_[:, clus_[key]], axis=1)

    elif center_method == "raw_IDP":
        for i in range(len(center_idx)):
            centers_[chr(i + 65)] = pos_[:, center_idx[i]]
    else:
        # default to mean
        for i in range(len(center_idx)):
            key = chr(i + 65)
            centers_[key] = np.mean(pos_[:, clus_[key]], axis=1)

    # Final AoI sequence
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


def compute_threshold(gamma):
    gamma = np.asarray(gamma, dtype=float)

    # gmean requires positive values
    eps = 1e-12
    gamma = np.maximum(gamma, eps)

    gamma_sorted = np.sort(gamma)[::-1]
    n_s = len(gamma_sorted)

    # Use log2 for the intended discrete weighting behavior
    weights = []
    for i in range(n_s):
        alpha = 2 ** (np.floor(np.log2(n_s)) - np.ceil(np.log2(i + 1)) + 1) - 1
        weights.append(alpha)

    thresh = gmean(gamma_sorted, weights=np.array(weights, dtype=float))

    # Safety: if numerical issues occur, fall back to a percentile
    if not np.isfinite(thresh):
        thresh = float(np.percentile(gamma_sorted, 90))

    return float(thresh)