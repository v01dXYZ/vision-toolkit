import numpy as np
import math

from vision_toolkit2.config import Config
from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints

from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config):
    (idx_velocity_lower_than_threshold,) = np.where(
        s.absolute_speed <= config.IVT_velocity_threshold
    )

    is_fix = np.full(config.nb_samples, False)

    is_fix[idx_velocity_lower_than_threshold] = True
    is_fix[(idx_velocity_lower_than_threshold + 1).clip(max=config.nb_samples - 1)] = (
        True
    )

    is_sac = ~is_fix
    (idx_fix,) = is_fix.nonzero()
    (idx_sac,) = (~is_fix).nonzero()

    s_ints = interval_merging(
        idx_sac,
        min_int_size=math.ceil(
            config.min_sac_duration * config.sampling_frequency,
        ),
    )

    is_fix = np.full(config.nb_samples, True)

    for s_start, s_end in s_ints:
        is_fix[s_start : s_end + 1] = False

    (idx_fix,) = is_fix.nonzero()

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

    (idx_sac,) = is_sac.nonzero()

    s_ints = interval_merging(
        idx_sac,
        min_int_size=math.ceil(config.min_sac_duration * config.sampling_frequency),
        status=s.status,
        proportion=config.status_threshold,
    )

    i_lab = np.full(config.nb_samples, False)

    for ints in (f_ints, s_ints):
        for start, end in ints:
            i_lab[start : end + 1] = True

    return BinarySegmentationResults(
        is_labeled=i_lab,
        fixation_intervals=f_ints,
        saccade_intervals=s_ints,
        fixation_centroids=ctrds,
        input=s,
        config=config,
    )


def default_config_impl(config, vf_diag):
    if config.distance_type == "euclidean":
        v_t = vf_diag * 0.2
        return Config(
            IVT_velocity_threshold=v_t,
        )
    elif config.distance_type == "angular":
        return Config(
            IVT_velocity_threshold=50,
        )
