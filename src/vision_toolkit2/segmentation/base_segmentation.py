from vision_toolkit2.config import Config, StackedConfig
from vision_toolkit2.serie import Serie
from vision_toolkit2.velocity_distance_factory import (
    absolute_angular_distance,
    absolute_euclidean_distance,
)
from vision_toolkit2 import config as c

from .binary import implementations as binary_implementations
from .ternary import implementations as ternary_implementations

from .ternary.ternary_segmentation_results import TernarySegmentationResults

import numpy as np


IMPLEMENTATIONS = {
    **binary_implementations.IMPLEMENTATIONS,
    **ternary_implementations.IMPLEMENTATIONS,
}

class DefaultConfigBuilder:
    DEFAULT_CONFIG = c.Config(
        distance_type="angular",
        segmentation=c.Segmentation(
            filter=c.SegmentationFilter(
                fixation_duration=c.FilterRange[float](
                    min=7e-2,
                    max=2.0,
                ),
                saccade_duration=c.FilterRange[float](
                    min=1.5e-2,
                    max=None,
                ),
                pursuit_duration=c.FilterRange[float](
                    min=1e-1,
                    max=2.0,
                ),
                status_threshold=0.5,
            ),
            pursuit=c.Pursuit(
                start_idx=0,
            ),
        ),
        display=c.Display(
            segmentation=False,
            results=True,
        ),
        verbose=False,
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
            stack.append(Config(segmentation=c.Segmentation(segmentation_method)))

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
    DEFAULT_SEGMENTATION_METHOD = None
    ALLOWED_SEGMENTATION_METHODS = None

    def __init__(
        self,
        input_: Serie,
        *,
        segmentation_method=None,
        config: Config = None,
    ):

        if segmentation_method is None:
            try:
                segmentation_method = config.segmentation.method
            except AttributeError:
                pass

            segmentation_method = segmentation_method or self.DEFAULT_SEGMENTATION_METHOD

        if segmentation_method is None:
            raise ValueError("There is no segmentation method specified either as arg, in config nor in class default")

        if (
                self.ALLOWED_SEGMENTATION_METHODS is not None
                and segmentation_method not in self.ALLOWED_SEGMENTATION_METHODS
        ):
            raise ValueError(f"Segmentation method should be among: {self.ALLOWED_SEGMENTATION_METHODS}")

        self.input_ = input_
        self.config = DefaultConfigBuilder.update(
            input_,
            config,
            segmentation_method=segmentation_method,
        )
        self.min_config = self.config.segmentation

    def process(self):
        process_impl, _ = IMPLEMENTATIONS[self.config.segmentation.method]
        results = process_impl(
            self.input_,
            self.min_config,
            self.config.distance_type,
            self.config.verbose,
        )
        results = results.filter_events()

        return results

class BinarySegmentation(Segmentation):
    DEFAULT_SEGMENTATION_METHOD = "I_HMM"
    ALLOWED_SEGMENTATION_METHODS = binary_implementations.IMPLEMENTATIONS.keys()

class TernarySegmentation(Segmentation):
    DEFAULT_SEGMENTATION_METHOD = "I_VMP"
    ALLOWED_SEGMENTATION_METHODS = ternary_implementations.IMPLEMENTATIONS.keys()
