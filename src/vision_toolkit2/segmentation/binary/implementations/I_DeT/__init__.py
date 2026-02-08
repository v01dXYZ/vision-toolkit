# -*- coding: utf-8 -*-
import time

from . import _optimized
from vision_toolkit2.config import Config
from ..binary_segmentation_results import BinarySegmentationResults


def process_impl(s, config):
    if config.verbose:
        print("Processing DeT Identification...")
        start_time = time.time()

    data_set = {
        'x_array': s.x,
        'y_array': s.y,
        'z_array': s.z,
        'status': s.status,
    }

    old_config = {
        'nb_samples': config.nb_samples,
        'sampling_frequency': config.sampling_frequency,
        'distance_type': config.distance_type,
        'IDeT_density_threshold': config.IDeT_density_threshold,
        'IDeT_duration_threshold': config.IDeT_duration_threshold,
        'IDeT_min_pts': config.IDeT_min_pts,
        'min_sac_duration': config.min_sac_duration,
        'min_fix_duration': config.min_fix_duration,
        'max_fix_duration': config.max_fix_duration,
        'status_threshold': config.status_threshold,
        'verbose': config.verbose,
    }

    out = _optimized.process_IDeT(data_set, old_config)

    if config.verbose:
        print("\n...DeT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return BinarySegmentationResults(
        is_labeled=out['is_labeled'],
        fixation_intervals=out['fixation_intervals'],
        saccade_intervals=out['saccade_intervals'],
        fixation_centroids=out['centroids'],
        input=s,
        config=config,
    )


def default_config_impl(config, vf_diag):
    ## To accelerate computation, duration threshold should be equal to 5 time stamps
    du_t = 5 / config.sampling_frequency
    if config.distance_type == "euclidean":
        ## The default density threshold is thus defined from the sampling frequency
        de_t = vf_diag / config.sampling_frequency
        return Config(
            IDeT_duration_threshold=du_t,
            IDeT_density_threshold=de_t,
            IDeT_min_pts=3,
        )
    elif config.distance_type == "angular":
        ## The default density threshold is thus defined from the sampling frequency
        de_t = 30 / config.sampling_frequency
        return Config(
            IDeT_duration_threshold=du_t,
            IDeT_density_threshold=de_t,
            IDeT_min_pts=3,
        )
