from .implementations import IMPLEMENTATIONS
from ..default_config_builder import DefaultConfigBuilder

import numpy as np


class BinaryDefaultConfigBuilder(DefaultConfigBuilder):
    @classmethod
    def update(cls, input_, config):
        config = super().update(input_, config)

        _, default_config_impl = IMPLEMENTATIONS[config.segmentation_method]
        vf_diag = np.linalg.norm(np.array([config.size_plan_x, config.size_plan_y]))
        config += default_config_impl(config, vf_diag)
        return config

    # @staticmethod
    # def for_I_VT(config, vf_diag):
    #     if config.distance_type == "euclidean":
    #         v_t = vf_diag * 0.2
    #         return Config(
    #             IVT_velocity_threshold = v_t,
    #         )
    #     elif config.distance_type == "angular":
    #         return Config(
    #             IVT_velocity_threshold = 50,
    #         )

    # @staticmethod
    # def for_I_DiT(config, vf_diag):
    #     if config.distance_type == "euclidean":
    #         di_t = 0.01 * vf_diag

    #         return Config(
    #             IDiT_window_duration = 0.040,
    #             IDiT_dispersion_threshold = di_t,
    #         )
    #     elif config.distance_type == "angular":
    #         return Config(
    #             IDiT_window_duration =  0.040,
    #             IDiT_dispersion_threshold = 0.3,
    #         )

    # @staticmethod
    # def for_I_DeT(config, vf_diag):
    #     ## To accelerate computation, duration threshold should be equal to 5 time stamps
    #     du_t = 5 / sampling_frequency
    #     if config.distance_type == "euclidean":
    #         ## The default density threshold is thus defined from the sampling frequency
    #         de_t = vf_diag / sampling_frequency
    #         return Config(
    #             IDeT_duration_threshold =  du_t,
    #             IDeT_density_threshold = de_t,
    #         )
    #     elif config.distance_type == "angular":
    #         ## The default density threshold is thus defined from the sampling frequency
    #         de_t = 30 / sampling_frequency
    #         return Config(
    #             IDeT_duration_threshold = du_t,
    #             IDeT_density_threshold = de_t,
    #         )

    # def for_I_KF(config, vf_diag):
    #     pass

    # def for_I_MST(config, vf_diag):
    #     pass

    # def for_I_HMM(config, vf_diag):
    #     i_l = 0.001 * vf_diag
    #     i_h = 10.0 * vf_diag
    #     i_v = 100 * vf_diag**2

    #     return Config(
    #         HMM_init_low_velocity = i_l,
    #         HMM_init_high_velocity = i_h,
    #         HMM_init_variance = i_v,
    #         HMM_nb_iters = 10,
    #     )

    # def for_I_2MC(config, vf_diag):
    #     pass

    # def for_I_RF(config, vf_diag):
    #     pass
