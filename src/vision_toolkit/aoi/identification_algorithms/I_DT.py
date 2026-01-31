import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from vision_toolkit.utils.identification_utils import compute_aoi_sequence


def process_IDT(values, config, ref_image=None):
    """
    IDT via DBSCAN with optional noise reassignment.

    config keys:
      - AoI_IDT_density_threshold (eps)
      - AoI_IDT_min_samples
      - AoI_IDT_reassign_noise (bool, default False)

    Returns
    -------
    results : dict
      {
        "AoI_sequence": list[str] (or collapsed/binned depending config),
        "AoI_durations": np.ndarray,
        "centers": dict[str -> np.ndarray(2,)],
        "clustered_fixations": dict[str -> np.ndarray(indices)],
      }
    new_values : np.ndarray shape (3, n')
      - if reassign_noise == False : n' = number of non-noise fixations (filtered)
      - if reassign_noise == True  : n' = original number of fixations (kept)
    """

    # --- Get inputs and parameters ---
    pos_ = values[0:2]
    dur_ = values[2]

    eps = config["AoI_IDT_density_threshold"]
    min_samples = config["AoI_IDT_min_samples"]
    reassign_noise = bool(config.get("AoI_IDT_reassign_noise", False))

    # --- Density based clustering ---
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_.T)
    seq_all = np.asarray(dbscan.labels_, dtype=int)  # -1 = noise
    n = seq_all.size
 
    if not reassign_noise:
        t_k = seq_all >= 0

        pos_f = pos_[:, t_k]
        dur_f = dur_[t_k]
        seq_f = seq_all[t_k]

        centers_ = {}
        clus_ = {}

        if seq_f.size == 0:
            # (rare) everything noise -> keep pipeline alive like other algos
            centers_ = {"A": np.mean(pos_, axis=1)}
            clus_ = {"A": np.arange(pos_.shape[1], dtype=int)}
            seq0 = np.zeros(pos_.shape[1], dtype=int)
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
 
        K = int(np.max(seq_f)) + 1
        for i in range(K):
            vals_ = np.argwhere(seq_f == i).T[0]
            clus_[chr(i + 65)] = vals_
            centers_[chr(i + 65)] = np.mean(pos_f[:, vals_], axis=1)

        seq_letters, seq_dur = compute_aoi_sequence(seq_f, dur_f, config)
        new_values = np.vstack((pos_f, dur_f))

        results = {
            "AoI_sequence": seq_letters,
            "AoI_durations": seq_dur,
            "centers": centers_,
            "clustered_fixations": clus_,
        }
        return results, new_values
 
    non_noise = np.where(seq_all >= 0)[0]
    noise = np.where(seq_all < 0)[0]

    if non_noise.size == 0:
        # all noise -> fallback single AoI
        centers_ = {"A": np.mean(pos_, axis=1)}
        clus_ = {"A": np.arange(n, dtype=int)}
        seq0 = np.zeros(n, dtype=int)
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

    # DBSCAN labels among non-noise can be non-contiguous in theory; we remap to 0..K-1
    uniq = np.unique(seq_all[non_noise])
    remap = {lab: i for i, lab in enumerate(uniq)}
    seq_remap = np.full(n, -1, dtype=int)
    for idx in non_noise:
        seq_remap[idx] = remap[seq_all[idx]]

    K = len(uniq)

    # compute centers from non-noise first
    centers_arr = np.zeros((K, 2), dtype=float)
    for k in range(K):
        idxs = np.where(seq_remap == k)[0]
        centers_arr[k] = np.mean(pos_[:, idxs], axis=1)

    # assign each noise point to nearest center
    if noise.size > 0:
        d = cdist(pos_[:, noise].T, centers_arr, metric="euclidean")  # (n_noise, K)
        nearest = np.argmin(d, axis=1)
        seq_remap[noise] = nearest

    # build clusters in ORIGINAL space
    centers_ = {}
    clus_ = {}
    for k in range(K):
        idxs = np.where(seq_remap == k)[0]
        clus_[chr(k + 65)] = idxs
        centers_[chr(k + 65)] = np.mean(pos_[:, idxs], axis=1)

    # compute final AoI sequence on ALL points
    seq_letters, seq_dur = compute_aoi_sequence(seq_remap, dur_, config)
    new_values = np.vstack((pos_, dur_))

    results = {
        "AoI_sequence": seq_letters,
        "AoI_durations": seq_dur,
        "centers": centers_,
        "clustered_fixations": clus_,
    }
    return results, new_values
