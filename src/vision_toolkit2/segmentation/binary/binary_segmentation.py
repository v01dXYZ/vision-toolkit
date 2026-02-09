from vision_toolkit2.config import Config


from vision_toolkit2.velocity_distance_factory import (
    absolute_angular_distance,
    absolute_euclidian_distance,
)

from .implementations import IMPLEMENTATIONS
from .default_config_builder import BinaryDefaultConfigBuilder
from vision_toolkit2.oculomotor_series import AugmentedSerie

# @dataclass
# class BinarySegmentationResults:
#     config: Config

#     is_labeled: None # array[bool, length=N]
#     fixation_intervals: None # array[
#     saccade_intervals: None
#     fixation_centroids: None


class BinarySegmentation:
    DISTANCES = {
        "euclidean": absolute_euclidian_distance,
        "angular": absolute_angular_distance,
    }

    # {
    #     "I_VT": I_VT.process_impl,
    #     # # "I_VT": process_IVT,
    #     # "I_DiT": process_IDiT,
    #     # "I_DeT": process_IDeT,
    #     # "I_KF": process_IKF,
    #     # "I_MST": process_IMST,
    #     # "I_HMM": process_IHMM,
    #     # "I_2MC": process_I2MC,
    #     # "I_RF": process_IRF
    # }

    def __init__(
        self,
        input_: AugmentedSerie,
        config: Config,
    ):
        self.input_ = input_
        self.config = BinaryDefaultConfigBuilder.update(input_, config)
        self.config += config

    def process(self):
        process_impl, _ = IMPLEMENTATIONS[self.config.segmentation_method]

        results = process_impl(self.input_, self.config)

        self.config.print()

        return results
