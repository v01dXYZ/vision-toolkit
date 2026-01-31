# -*- coding: utf-8 -*-

import itertools
from itertools import groupby

import numpy as np


def compute_aoi_sequence(seq_, dur_, config):
    """


    Parameters
    ----------
    seq_ : TYPE
        DESCRIPTION.
    dur_ : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    seq_ : TYPE
        DESCRIPTION.

    """

    n_s = len(seq_)
    aoi_temporal_binning = config["AoI_temporal_binning"]

    if aoi_temporal_binning == False:
        ## Convert to capital letters
        seq_ = [chr(seq_[i] + 65) for i in range(n_s)]
        seq_dur = [dur_[i] for i in range((n_s))]

    elif aoi_temporal_binning == True:
        ## Convert to capital letters with repetitions depending on fixation duration
        temp_bin = config["AoI_temporal_binning_length"]

        seq__ = []
        for i in range(n_s):
            [
                seq__.append(chr(seq_[i] + 65))
                for _ in range(int(np.ceil(dur_[i] / temp_bin)))
            ]
        seq_ = seq__

        seq_dur = []
        for i in range(n_s):
            if dur_[i] // temp_bin > 0:
                [seq_dur.append(temp_bin) for _ in range(int(dur_[i] // temp_bin))]
            if dur_[i] % temp_bin > 0:
                seq_dur.append(dur_[i] % temp_bin)

    elif aoi_temporal_binning == "collapse":
        ## Convert to capital letters removing elements that have consecutive duplicates
        seq_ = [chr(seq_[i] + 65) for i in range(n_s)]

        seq_dur = []
        dur__ = list(dur_)
        for key, _group in groupby(seq_):
            g_ = list(_group)
            l_ = sum(1 for x in g_)
            seq_dur.append(np.sum(np.array(dur__[:l_])))
            dur__ = dur__[l_:]

        seq_ = [key for key, _group in groupby(seq_)]

    else:
        raise ValueError(
            "'AoI_temporal_binning' must be set to True, or False, or 'collapse'"
        )

    assert len(seq_) == len(
        seq_dur
    ), "AoI sequences and duration sequences must have same length"

    return seq_, np.array(seq_dur)

 

def merge_small_aois(sequence,
                     durations,
                     centers,
                     clustered_fixations,
                     min_fixations,
                     relabel=True):
    """
    Merge small AoIs when AoI labels are LETTERS only.

    Parameters
    ----------
    sequence : array-like of str
        AoI sequence like ["A","A","G", ...] (already converted to letters).
        Can be the output of compute_aoi_sequence (binning/collapse).
    durations : array-like
        Sequence durations aligned with `sequence` (same length).
        Returned unchanged (still aligned with the merged/relabeled sequence).
    centers : dict[str, np.ndarray]
        {label: np.array([x, y])}
    clustered_fixations : dict[str, np.ndarray]
        {label: fixation_indices_in_original_fixations}
    min_fixations : int
        AoIs with < min_fixations in clustered_fixations are merged.
    relabel : bool
        If True, relabel remaining AoIs to "A","B","C"... in sorted order.

    Returns
    -------
    new_sequence : np.ndarray of str
    new_durations : np.ndarray
    new_centers : dict[str, np.ndarray]
    new_clustered_fixations : dict[str, np.ndarray]
    """
    seq = np.asarray(sequence, dtype=str).copy()
    dur = np.asarray(durations).copy()

    if min_fixations is None or min_fixations <= 0:
        return seq, dur, centers, clustered_fixations

    if centers is None or len(centers) <= 1:
        return seq, dur, centers, clustered_fixations

    if clustered_fixations is None or len(clustered_fixations) == 0:
        # fallback: estimate "fixations per AoI" from occurrences in sequence
        # (less faithful if temporal binning=True, but avoids crashing)
        labels, counts = np.unique(seq, return_counts=True)
        clustered_fixations = {lab: np.arange(c, dtype=int) for lab, c in zip(labels, counts)}

    # Only consider labels existing in centers
    labels = sorted([lab for lab in clustered_fixations.keys() if lab in centers])
    if len(labels) <= 1:
        return seq, dur, centers, clustered_fixations

    # Small/valid based on number of fixations (clustered_fixations)
    small = [lab for lab in labels if len(clustered_fixations.get(lab, [])) < min_fixations]
    valid = [lab for lab in labels if len(clustered_fixations.get(lab, [])) >= min_fixations]

    # Nothing to merge or everything small
    if not small or not valid:
        return seq, dur, centers, clustered_fixations

    valid_centers = np.array([centers[lab] for lab in valid], dtype=float)

    # Build merge mapping: small -> nearest valid
    merge_into = {}
    for lab in small:
        src_center = np.asarray(centers[lab], dtype=float)
        dists = np.sum((valid_centers - src_center) ** 2, axis=1)
        tgt = valid[int(np.argmin(dists))]
        merge_into[lab] = tgt

    # --- update sequence labels ---
    for src, tgt in merge_into.items():
        seq[seq == src] = tgt

    # --- update clustered_fixations ---
    # start from valid labels
    new_clus = {lab: np.asarray(clustered_fixations.get(lab, []), dtype=int).copy() for lab in valid}

    # add merged fixations into target
    for src, tgt in merge_into.items():
        src_idx = np.asarray(clustered_fixations.get(src, []), dtype=int)
        if src_idx.size == 0:
            continue
        new_clus[tgt] = np.unique(np.concatenate([new_clus.get(tgt, np.array([], dtype=int)), src_idx])).astype(int)

    # keep only labels still present after merge
    kept = sorted(set(seq.tolist()))
    new_centers = {lab: centers[lab] for lab in kept if lab in centers}
    new_clus = {lab: new_clus.get(lab, np.array([], dtype=int)) for lab in kept}

    if not relabel:
        return seq, dur, new_centers, new_clus

    # --- relabel to A,B,C,... (compact) ---
    kept_sorted = sorted(new_centers.keys())
    mapping = {old: chr(65 + i) for i, old in enumerate(kept_sorted)}

    new_seq = np.array([mapping[lab] for lab in seq], dtype=str)
    relabeled_centers = {mapping[old]: new_centers[old] for old in kept_sorted}
    relabeled_clus = {mapping[old]: new_clus[old] for old in kept_sorted}

    return new_seq, dur, relabeled_centers, relabeled_clus



def temporal_binning_AoI(seq_, dur_, config):
    n_s = len(seq_)

    assert (
        dur_ is not None
    ), "AoI_durations must be provided to perform temporal binning"

    temp_bin = config["AoI_temporal_binning_length"]

    seq__ = []
    for i in range(n_s):
        [seq__.append(seq_[i]) for _ in range(int(np.ceil(dur_[i] / temp_bin)))]
    seq_ = seq__

    seq_dur = []
    for i in range(n_s):
        if dur_[i] // temp_bin > 0:
            [seq_dur.append(temp_bin) for _ in range(int(dur_[i] // temp_bin))]
        if dur_[i] % temp_bin > 0:
            seq_dur.append(dur_[i] % temp_bin)

    return seq_, np.array(seq_dur)


def collapse_AoI(seq_, dur_):
    if dur_ is not None:
        seq_dur = []
        dur__ = list(dur_)
        for key, _group in groupby(seq_):
            g_ = list(_group)
            l_ = sum(1 for x in g_)
            seq_dur.append(np.sum(np.array(dur__[:l_])))
            dur__ = dur__[l_:]

    else:
        seq_dur = None

    seq_ = [key for key, _group in groupby(seq_)]

    return seq_, np.array(seq_dur)

