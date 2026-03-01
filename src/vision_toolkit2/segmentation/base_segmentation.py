from vision_toolkit2.config import Config, StackedConfig
from vision_toolkit2.serie import Serie
from vision_toolkit2.velocity_distance_factory import (
    absolute_angular_distance,
    absolute_euclidean_distance,
)
from vision_toolkit2 import config

from .binary import implementations as binary_implementations
from .ternary import implementations as ternary_implementations

from .ternary.ternary_segmentation_results import TernarySegmentationResults

import numpy as np


IMPLEMENTATIONS = {
    **binary_implementations.IMPLEMENTATIONS,
    **ternary_implementations.IMPLEMENTATIONS,
}

DEFAULT_SEGMENTATION_METHOD = "I_HMM"


class DefaultConfigBuilder:
    DEFAULT_CONFIG = config.Config(
        distance_type="angular",
        segmentation=config.Segmentation(
            DEFAULT_SEGMENTATION_METHOD,
            filter=config.SegmentationFilter(
                fixation_duration=config.FilterRange[float](
                    min=7e-2,
                    max=2.0,
                ),
                saccade_duration=config.FilterRange[float](
                    min=1.5e-2,
                    max=None,
                ),
                pursuit_duration=config.FilterRange[float](
                    min=1e-1,
                    max=2.0,
                ),
                status_threshold=0.5,
            ),
        ),
        display=config.Display(
            segmentation=False,
            results=True,
        ),
        verbose=True,
    )

    @classmethod
    def update(cls, input_, config, segmentation_method=None):
        stack = [
            cls.DEFAULT_CONFIG,
            input_.config,
        ]
        if config is not None:
            stack.append(config)

        if segmentation_method is not None:
            stack.append(Config(segmentation_method=segmentation_method))

        config = StackedConfig(stack)

        segmentation_method = config.segmentation.method
        _, default_config_impl = IMPLEMENTATIONS[segmentation_method]
        vf_diag = np.linalg.norm(
            np.array([config.screen_dimensions.x, config.screen_dimensions.y])
        )
        config += default_config_impl(config, vf_diag)

        return config


class Segmentation:
    DISTANCES = {
        "euclidean": absolute_euclidean_distance,
        "angular": absolute_angular_distance,
    }

    def __init__(
        self,
        input_: Serie,
        *,
        config: Config = None,
        segmentation_method=None,
    ):
        self.input_ = input_
        self.config = DefaultConfigBuilder.update(
            input_,
            config,
            segmentation_method=segmentation_method,
        )
        self.min_config = self.config.segmentation

    def process(self):
        process_impl, _ = IMPLEMENTATIONS[self.config.segmentation.method]
        results = process_impl(self.input_, self.config, self.min_config, self.config.distance_type, self.config.verbose)

        if isinstance(results, TernarySegmentationResults):
            conf = self.config
            results = results.filter_events_by_duration(
                fixation_duration_range=(
                    conf.segmentation.filter.fixation_duration.min,
                    conf.segmentation.filter.fixation_duration.max,
                ),
                pursuit_duration_range=(
                    conf.segmentation.filter.pursuit_duration.min,
                    conf.segmentation.filter.pursuit_duration.max,
                ),
            )

        # self.config.print()

        return results
