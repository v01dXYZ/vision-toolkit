import numpy as np
import math

from vision_toolkit2.config import Config
from vision_toolkit2.config import IVT, Segmentation
from vision_toolkit2.segmentation.utils import interval_merging, centroids_from_ints

from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, segmentation_config, distance_type, verbose):
    # Find indices where velocity is below threshold
    (idx_velocity_lower_than_threshold,) = np.where(
        s.absolute_speed <= segmentation_config.ivt.velocity_threshold
    )

    is_fix = np.full(s.min_config.nb_samples, False)

    # Add index + 1 to fixation since velocities are computed from two data points
    is_fix[idx_velocity_lower_than_threshold] = True
    is_fix[
        (idx_velocity_lower_than_threshold + 1).clip(
            max=s.min_config.nb_samples - 1
        )
    ] = True

    # Compute saccadic intervals
    is_sac = ~is_fix
    (idx_fix,) = is_fix.nonzero()
    (idx_sac,) = (~is_fix).nonzero()

    s_ints = interval_merging(
        idx_sac,
        min_int_size=math.ceil(
            segmentation_config.filter.saccade_duration.min
            * s.min_config.sampling_frequency,
        ),
    )

    is_fix = np.full(s.min_config.nb_samples, True)

    for s_start, s_end in s_ints:
        is_fix[s_start : s_end + 1] = False

    (idx_fix,) = is_fix.nonzero()

    fix_dur_t = math.ceil(
        segmentation_config.filter.fixation_duration.min
        * s.min_config.sampling_frequency
    )

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        if s_int[0] - o_s_int[-1] <= fix_dur_t:
            is_fix[o_s_int[-1] : s_int[0] + 1] = False

    # Recompute fixation intervals
    f_ints = interval_merging(
        idx_fix,
        min_int_size=math.ceil(
            segmentation_config.filter.fixation_duration.min
            * s.min_config.sampling_frequency
        ),
        max_int_size=math.ceil(
            segmentation_config.filter.fixation_duration.max
            * s.min_config.sampling_frequency
        ),
        status=s.status,
        proportion=segmentation_config.filter.status_threshold,
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, s.x, s.y)

    is_sac = ~is_fix

    (idx_sac,) = is_sac.nonzero()

    # Recompute saccadic intervals
    s_ints = interval_merging(
        idx_sac,
        min_int_size=math.ceil(
            segmentation_config.filter.saccade_duration.min
            * s.min_config.sampling_frequency
        ),
        status=s.status,
        proportion=segmentation_config.filter.status_threshold,
    )

    # Keep track of index that were effectively labeled
    i_lab = np.full(s.min_config.nb_samples, False)

    for ints in (f_ints, s_ints):
        for start, end in ints:
            i_lab[start : end + 1] = True

    return BinarySegmentationResults(
        is_labeled=i_lab,
        fixation_intervals=f_ints,
        saccade_intervals=s_ints,
        fixation_centroids=ctrds,
        input=s,
        config=segmentation_config,
    )


def default_config_impl(config, vf_diag):
    if config.distance_type == "euclidean":
        v_t = vf_diag * 0.2
        ivt_config = IVT(velocity_threshold=v_t)
        return Config(
            segmentation=Segmentation(ivt_config),
        )
    elif config.distance_type == "angular":
        ivt_config = IVT(velocity_threshold=50)
        return Config(
            segmentation=Segmentation(ivt_config),
        )
