from vision_toolkit2.config import Config, StackedConfig
from vision_toolkit2.oculomotor_series import AugmentedSerie
from vision_toolkit2.velocity_distance_factory import (
    absolute_angular_distance,
    absolute_euclidian_distance,
)

from  .binary import implementations as binary_implementations
from .ternary import implementations as ternary_implementations

from .ternary.ternary_segmentation_results import TernarySegmentationResults

import numpy as np


IMPLEMENTATIONS = {
    **binary_implementations.IMPLEMENTATIONS,
    **ternary_implementations.IMPLEMENTATIONS,
}

class DefaultConfigBuilder:
    DEFAULT_CONFIG = Config(
        segmentation_method="I_HMM",
        distance_type="angular",
        min_fix_duration=7e-2,
        max_fix_duration=2.0,
        min_sac_duration=1.5e-2,
        min_pursuit_duration=1e-1,
        max_pursuit_duration=2.0,
        status_threshold=0.5,
        display_segmentation=False,
        display_results=True,
        verbose=True,
    )

    @classmethod
    def update(cls, input_, config):
        config = StackedConfig([cls.DEFAULT_CONFIG, config])
        config += cls.for_smoothing(config)

        _, default_config_impl = IMPLEMENTATIONS[config.segmentation_method]
        vf_diag = np.linalg.norm(np.array([config.size_plan_x, config.size_plan_y]))
        config += default_config_impl(config, vf_diag)

        return config

    @staticmethod
    def for_smoothing(config):
        if config.smoothing in (
            "moving_average",
            "speed_moving_average",
        ):
            return Config(moving_average_window=5)
        elif config.smoothing == "savgol":
            return Config(
                savgol_window_length=31,
                savgol_polyorder=3,
            )
        return Config()

class Segmentation:
    DISTANCES = {
        "euclidean": absolute_euclidian_distance,
        "angular": absolute_angular_distance,
    }


    def __init__(
        self,
        input_: AugmentedSerie,
        config: Config,
    ):
        self.input_ = input_
        self.config = DefaultConfigBuilder.update(input_, config)
        self.config += config

    def process(self):
        process_impl, _ = IMPLEMENTATIONS[self.config.segmentation_method]

        results = process_impl(self.input_, self.config)

        if isinstance(results, TernarySegmentationResults):
            conf = self.config
            results = results.filter_events_by_duration(
                fixation_duration_range=(
                    conf.min_fix_duration, 
                    conf.max_fix_duration,
                ),
                pursuit_duration_range=(
                    conf.min_pursuit_duration,
                    conf.max_pursuit_duration,
                ),
            )

        self.config.print()

        return results
