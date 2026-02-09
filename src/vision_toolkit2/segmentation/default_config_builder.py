from vision_toolkit2.config import Config, StackedConfig


class DefaultConfigBuilder:
    DEFAULT_CONFIG = Config(
        segmentation_method="I_HMM",
        distance_type="angular",
        min_fix_duration=7e-2,
        max_fix_duration=2.0,
        min_sac_duration=1.5e-2,
        status_threshold=0.5,
        display_segmentation=False,
        display_results=True,
        verbose=True,
    )

    @classmethod
    def update(cls, input_, config):
        config = StackedConfig([cls.DEFAULT_CONFIG, config])
        config += cls.for_smoothing(config)

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
