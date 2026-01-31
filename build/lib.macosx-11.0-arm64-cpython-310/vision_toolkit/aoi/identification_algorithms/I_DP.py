# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gmean

from vision_toolkit.utils.identification_utils import compute_aoi_sequence


def process_IDP(values, config, ref_image=None):
    pos_ = values[0:2]
    dur_ = values[2]
    n_s = len(dur_)

    k_sd = float(config["AoI_IDP_gaussian_kernel_sd"])
    center_method = config["AoI_IDP_centers"]

    k_sd = max(k_sd, 1e-9)

    # distances
    dist_ = cdist(pos_.T, pos_.T)
    max_d = float(np.max(dist_))

    # rho (gaussian kernel density, excluding self-term)
    pdf_mat = np.exp(-0.5 * (dist_ / k_sd) ** 2)
    rho = np.sum(pdf_mat, axis=1) - 1.0
    max_r = float(np.max(rho))

    # delta: distance to nearest point with higher density
    delta = np.empty(n_s, dtype=float)
    for i in range(n_s):
        if rho[i] < max_r:
            idxs = np.where(rho > rho[i])[0]
            delta[i] = float(np.min(dist_[i, idxs]))  # idxs non-empty
        else:
            delta[i] = max_d

    gamma = rho * delta
    thresh = compute_threshold(gamma)

    center_idx = np.where(gamma > thresh)[0]
    if center_idx.size == 0:
        center_idx = np.array([int(np.argmax(gamma))], dtype=int)

    # assign each point to nearest center (label = 0..K-1)
    dist_f = dist_[:, center_idx]              # (n, K)
    seq_ = np.argmin(dist_f, axis=1).astype(int)

    # clustered_fixations (letters -> indices)
    clus_ = {}
    K = int(center_idx.size)
    for i in range(K):
        key = chr(65 + i)
        clus_[key] = np.where(seq_ == i)[0]

    # centers (letters -> (x,y))
    centers_ = {}
    if center_method == "raw_IDP":
        for i in range(K):
            centers_[chr(65 + i)] = pos_[:, center_idx[i]]
    else:
        # default/mean
        for i in range(K):
            key = chr(65 + i)
            idxs = clus_[key]
            if idxs.size == 0:
                centers_[key] = pos_[:, center_idx[i]]
            else:
                centers_[key] = np.mean(pos_[:, idxs], axis=1)

    # convert to AoI sequence (letters + temporal binning/collapse handled here)
    seq_letters, seq_dur = compute_aoi_sequence(seq_, dur_, config)

    return {
        "AoI_sequence": seq_letters,
        "AoI_durations": seq_dur,
        "centers": centers_,
        "clustered_fixations": clus_,
    }


def compute_threshold(gamma):
    
    gamma = np.asarray(gamma, dtype=float)

    eps = 1e-12
    gamma = np.maximum(gamma, eps)

    gamma_sorted = np.sort(gamma)[::-1]
    n_s = len(gamma_sorted)

    weights = []
    for i in range(n_s):
        alpha = 2 ** (np.floor(np.log2(n_s)) - np.ceil(np.log2(i + 1)) + 1) - 1
        weights.append(alpha)

    w = np.array(weights, dtype=float)
    sw = float(np.sum(w))
    if sw > 0:
        w /= sw

    thresh = gmean(gamma_sorted, weights=w)

    if not np.isfinite(thresh):
        thresh = float(np.percentile(gamma_sorted, 90))

    return float(thresh)
