# -*- coding: utf-8 -*-
"""Shared utilities for segmentation algorithms."""

import numpy as np


def interval_merging(
    x,
    min_int_size=0,
    max_int_size=np.inf,
    p_o=0,
    s_o=0,
    status=None,
    proportion=0.95,
):
    """
    Merge contiguous indices into intervals.

    Parameters
    ----------
    x : array-like
        Sequence of indices
    min_int_size : int
        Minimum interval size
    max_int_size : int
        Maximum interval size
    p_o : int
        Padding offset
    s_o : int
        Suffix offset
    status : array-like, optional
        Status vector for filtering
    proportion : float
        Minimum proportion of valid status in interval

    Returns
    -------
    array
        Valid intervals
    """
    if len(x) == 0:
        return np.zeros((0, 2), dtype=int)

    is_break = (np.diff(x) >= 2).flatten()
    idx_break, = is_break.nonzero()

    idx_intervals = np.zeros(
        (len(idx_break) + 1, 2),
        dtype=int,
    )

    idx_intervals[0, 0] = 0
    idx_intervals[-1, -1] = len(x) - 1

    idx_intervals[:-1, 1] = idx_break
    idx_intervals[1:, 0] = idx_break + 1

    x_idx_intervals = np.take(x, idx_intervals)
    x_size_intervals = x_idx_intervals[:, 1] - x_idx_intervals[:, 0] + s_o - p_o

    is_valid_interval = (min_int_size <= x_size_intervals) & (
        x_size_intervals < max_int_size
    )

    if status is not None:
        status_proportion = np.array(
            [
                np.mean(status[start - p_o : end + s_o + 1])
                for start, end in x_idx_intervals
            ]
        )
        is_with_enough_status = status_proportion > proportion

        is_valid_interval &= is_with_enough_status

    return x_idx_intervals[is_valid_interval]


def centroids_from_ints(f_ints, x_coords, y_coords):
    """
    Compute centroids for each interval.

    Parameters
    ----------
    f_ints : array-like
        List of intervals
    x_coords : array-like
        X coordinates
    y_coords : array-like
        Y coordinates

    Returns
    -------
    list
        List of [x_centroid, y_centroid] for each interval
    """
    return [
        [x_coords[start : end + 1].mean(), y_coords[start : end + 1].mean()]
        for start, end in f_ints
    ]


def dispersion_metric(x_coords, y_coords):
    """
    Calculate dispersion metric for coordinates.

    Parameters
    ----------
    x_coords : array-like
        X coordinates
    y_coords : array-like
        Y coordinates

    Returns
    -------
    float
        l1-spread of data points
    """

    return x_coords.max() - x_coords.min() + y_coords.max() - y_coords.min()
