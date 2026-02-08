import numpy as np

# from vision_toolkit.utils.segmentation_utils import (
#     centroids_from_ints,
#     interval_merging)

import math

from vision_toolkit2.oculomotor_series import AugmentedSerie
from vision_toolkit2.config import Config

from ..binary_segmentation_results import BinarySegmentationResults

def centroids_from_ints(f_ints, x_coords, y_coords):
    """

    """
    return [
        comp_centroid(
            x_coords[start : end + 1],
            y_coords[start : end + 1],
        )
        for start, end in f_ints        
    ]

def comp_centroid(x_coords, y_coords):
    """

    """

    return [x_coords.mean(), y_coords.mean()]

def interval_merging(
    x,
    min_int_size=0,
    max_int_size=np.inf,
    p_o = 0,
    s_o = 0,
    status = None,
    proportion = 0.95,
):
    """
    This function takes a sequence of indexes and outputs a sequence of intervals corresponding to the indexes that are contiguous.

    For instance, [0, 3, 4, 6, 8] outputs something like [(0, 0), (3, 6), (8, 8)].

    Furthermore, it filters out intervals that are deemed not satisfying enough such as:

    - not the right size: too small or too large
    - status is a integrity vector, if an interval do not contain enough values with integrity it is rejected.    
    """
    is_break = (np.diff(x) >= 2).flatten()
    idx_break, = is_break.nonzero()

    # we build a 2 column array where the first column is the first
    # index of the interval and the second is the last index of the
    # interval
    idx_intervals = np.zeros(
        (len(idx_break) + 1, 2),
        dtype=int,
    )

    # we initialize as the first interval starts at 0 and the last one
    # ends at the last element
    idx_intervals[0, 0] = 0
    idx_intervals[-1, -1] = len(x) - 1

    idx_intervals[:-1, 1] = idx_break
    idx_intervals[1:, 0] = idx_break + 1

    # Now we go back to the values stored in x
    x_idx_intervals = np.take(x, idx_intervals)
    x_size_intervals = x_idx_intervals[:, 1] - x_idx_intervals[:, 0] + s_o - p_o

    # Now we filter out the intervals that are not deemed worthy
    is_valid_interval = (min_int_size <= x_size_intervals)  & (x_size_intervals < max_int_size)

    if status is not None:
        status_proportion = np.array([
            np.mean(status[start - p_o:end + s_o + 1]) 
            for start, end in x_idx_intervals
        ])
        is_with_enough_status = status_proportion > proportion

        is_valid_interval &= is_with_enough_status

    return x_idx_intervals[is_valid_interval]

def process_impl(s, config):
    idx_velocity_lower_than_threshold, = np.where(
        s.absolute_speed <= config.IVT_velocity_threshold
    )

    is_fix = np.full(config.nb_samples, False)

    is_fix[idx_velocity_lower_than_threshold] = True
    is_fix[(idx_velocity_lower_than_threshold+1).clip(max=config.nb_samples-1)] = True

    is_sac = ~is_fix
    idx_fix, = is_fix.nonzero()
    idx_sac, = (~is_fix).nonzero()

    s_ints = interval_merging(
        idx_sac,
        min_int_size=math.ceil(
            config.min_sac_duration *
            config.sampling_frequency,
        ),
    )

    is_fix = np.full(config.nb_samples, True)

    for s_start, s_end in s_ints:
        is_fix[s_start : s_end + 1] = False

    idx_fix, = is_fix.nonzero()

    fix_dur_t = math.ceil(config.min_fix_duration * config.sampling_frequency)

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        if s_int[0] - o_s_int[-1] < fix_dur_t:
            is_fix[o_s_int[-1] : s_int[0] + 1] = False

    f_ints = interval_merging(
        idx_fix,
        min_int_size=math.ceil(config.min_fix_duration * config.sampling_frequency),
        max_int_size=math.ceil(config.max_fix_duration * config.sampling_frequency),
        status=s.status,
        proportion=config.status_threshold,
    )

    ctrds = centroids_from_ints(f_ints, s.x, s.y)

    is_sac = ~is_fix

    idx_sac, = is_sac.nonzero()

    s_ints = interval_merging(
        idx_sac,
        min_int_size=math.ceil(config.min_sac_duration * config.sampling_frequency),
        status=s.status,
        proportion=config.status_threshold,
    )

    i_lab = np.full(config.nb_samples, False)

    for start, end in f_ints + s_int:
        i_lab[start : end + 1] = True

    return BinarySegmentationResults(
        is_labeled= i_lab,
        fixation_intervals= f_ints,
        saccade_intervals= s_ints,
        fixation_centroids= ctrds,
        input = s,
        config = config,
    )

def default_config_impl(config, vf_diag):    
    if config.distance_type == "euclidean":
        v_t = vf_diag * 0.2
        return Config(
            IVT_velocity_threshold = v_t,
        )
    elif config.distance_type == "angular":
        return Config(
            IVT_velocity_threshold = 50,
        )
