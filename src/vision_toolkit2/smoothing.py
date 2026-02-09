from vision_toolkit2 import StackedConfig

from scipy.signal import savgol_filter


class Smoothing:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def process(self):
        dataset_with_old_config = self.process_dataset()

        return dataset_with_old_config.update_config(
            StackedConfig(
                [
                    dataset_with_old_config.config,
                    self.config,
                ]
            )
        )

    @classmethod
    def create(self, dataset, config):
        kw_to_cls = {
            None: NoSmoothing,
            "no": NoSmoothing,
            "moving_average": MovingAverage,
            "speed_moving_average": SpeedMovingAverage,
            "savgol": SavgolFiltering,
        }

        return kw_to_cls[config.smoothing](dataset, config)

    @classmethod
    def create_and_process(cls, dataset, config):
        return cls.create(dataset, config).process()


class NoSmoothing(Smoothing):
    pass


class MovingAverage(Smoothing):
    def process_dataset(self):
        _len = self.config.nb_samples
        _w = self.config.moving_average_window

        assert _w % 2 == 1, "Moving average window must be odd"

        for _dir in ["x", "y", "z"]:
            conv = np.convolve(getattr(self.dataset, _dir), np.ones(_w)) / _w
            smoothed_array = conv[:_len]

            for i in range(_w):
                smoothed_array[i] = smoothed_array[_w]

            setattr(self.dataset, _dir, smoothed_array)

        return self.dataset


class SpeedMovingAverage(Smoothing):
    def process_dataset(self):
        _delta_t = 1 / self.config.sampling_frequency
        _len = self.config.nb_samples
        _w = self.config.moving_average_window

        assert _w % 2 == 1, "Moving average window must be odd"

        for _dir in ["x", "y", "z"]:
            smoothed_array = copy.deepcopy(getattr(self.dataset, _dir))
            smoothed_speed_vector = self.generate_smoothed_speed_vector(
                self.dataset[_dir], _len, _w, _delta_t
            )

            for i in range(1, self.config.nb_samples):
                smoothed_array[i] = (
                    smoothed_array[i - 1] + smoothed_speed_vector[i - 1] * _delta_t
                )

            setattr(self.dataset, _dir, smoothed_array)

        return self.dataset


class SavgolFiltering(Smoothing):
    def process_dataset(self):
        _w = self.config.savgol_window_length
        _order = self.config.savgol_polyorder

        assert _w % 2 == 1, "Savgol window must be odd"

        for _dir in ["x", "y", "z"]:
            smoothed_array = savgol_filter(getattr(self.dataset, _dir), _w, _order)
            setattr(self.dataset, _dir, smoothed_array)

        return self.dataset
