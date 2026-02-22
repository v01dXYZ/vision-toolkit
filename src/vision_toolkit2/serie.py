from dataclasses import dataclass
import dataclasses

import pandas as pd
import numpy as np


from .config import Config, StackedConfig

from .velocity_distance_factory import (
    process_angular_absolute_speeds,
    process_angular_coord,
    process_euclidean_absolute_speeds,
    process_unitary_gaze_vectors,
)

import copy

from scipy.signal import savgol_filter
import numpy as np

class Smoothing:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def process(self):
        dataset_with_old_config = self.process_serie(self.dataset)

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

    def process_serie(self, serie):
        return SmoothedSerie(
            x=self.process_coordinate(serie.x),
            y=self.process_coordinate(serie.y),
            z=self.process_coordinate(serie.z),
            status=serie.status,
            config=serie.config,
            smoothing_config=self.config,
        )

class NoSmoothing(Smoothing):
    def process_serie(self, serie):
        return SmoothedSerie(
            x=serie.x,
            y=serie.y,
            z=serie.z,
            status=serie.status,
            config=serie.config,
            smoothing_config=None,
        )

class MovingAverage(Smoothing):
    def process_coordinate(self, coordinate):
        _len = self.config.nb_samples
        _w = self.config.moving_average_window
        assert _w % 2 == 1, "Moving average window must be odd"

        conv = np.convolve(coordinate, np.ones(_w)) / _w
        smoothed_array = conv[:_len]

        for i in range(_w):
            smoothed_array[i] = smoothed_array[_w]

        return smoothed_array

class SpeedMovingAverage(Smoothing):
    def process_coordinate(self, coordinate):
        _delta_t = 1 / self.config.sampling_frequency
        _len = self.config.nb_samples
        _w = self.config.moving_average_window
        assert _w % 2 == 1, "Moving average window must be odd"

        smoothed_array = copy.deepcopy(coordinate)
        smoothed_speed_vector = self.generate_smoothed_speed_vector(
            self.dataset[_dir], _len, _w, _delta_t
        )

        for i in range(1, self.config.nb_samples):
            smoothed_array[i] = (
                smoothed_array[i - 1] + smoothed_speed_vector[i - 1] * _delta_t
            )

        return smoothed_array

class SavgolFiltering(Smoothing):
    def process_coordinate(self, coordinate):
        _w = self.config.savgol_window_length
        assert _w % 2 == 1, "Savgol window must be odd"
        _order = self.config.savgol_polyorder

        smoothed_array = savgol_filter(coordinate, _w, _order)

        return smoothed_array

EPSILON = 1e-3
DEFAULT_DISTANCE_PROJECTION = 1_000

# @dataclass
# class OcculomotorConfig:
#     distance_projection: int
#     size_plan_x: float
#     size_plan_y: float

#     nb_samples: int
@dataclass
class RawSerie:
    x: np.array
    y: np.array
    z: np.array
    status: np.array

    config: Config

    def update_config(self, config):
        if self.config == config:
            return self

        return dataclasses.replace(
            self,
            config=StackedConfig([
                self.config,
                config,
            ]),
        )

    def __init__(self, x, y, z, status, config):
        self.x = x.astype("float64")
        self.y = y.astype("float64")
        if z is not None:
            z = z.astype("float64")
        else:
            distance_projection = config.distance_project or DEFAULT_DISTANCE_PROJECTION
            z = np.full_like(x, distance_projection, dtype="float64")
        self.z = z

        self.status = status

        self.config = config

    @classmethod
    def read_csv(
        cls,
        csv_path,
        *args,
        **kwargs,
    ):
        df = pd.read_csv(csv_path)

        return cls.from_df(
            df,
            *args,
            **kwargs,
        )

    @classmethod
    def from_df(
        cls,
        df,
        *,
        size_plan_x,
        size_plan_y,
        distance_projection=None,
        sampling_frequency=None,
        distance_type=None,
        **kwargs,
    ):
        distance_projection = distance_projection or 1000

        # Populate data (add columns if missing)
        x = df["gazeX"].values
        y = df["gazeY"].values
        z = df.get("gazeZ")
        if z is None:
            z = np.full_like(x, distance_projection)

        status = df.get("status")
        if status is None:
            status = np.ones_like(x)

        # generate the config to keep track of it
        if size_plan_x is None:
            size_plan_x = np.max(x) + EPSILON
        if size_plan_y is None:
            size_plan_y = np.max(y) + EPSILON

        config = Config(
            distance_projection=distance_projection,
            size_plan_x=size_plan_x,
            size_plan_y=size_plan_y,
            nb_samples=len(x),
            sampling_frequency=sampling_frequency,
            distance_type=distance_type,
        )

        raw_serie = RawSerie(
            x=x,
            y=y,
            z=z,
            status=status,
            config=config,
        )

        return cls.from_raw_serie(raw_serie, **kwargs)

    @classmethod
    def from_raw_serie(cls, raw_serie, **kwargs):
        return raw_serie

    ###############################
    # MOCK
    ###############################
    def get_data_set(self):
        return {
            "x_array": self.x,
            "y_array": self.y,
            "z_array": self.z,
            "status": self.status,
        }


BASE_SMOOTHING_CONFIG = Config(
    smoothing="savgol",
    savgol_window_length=31,
    savgol_polyorder=3,
)
@dataclass
class SmoothedSerie(RawSerie):
    smoothing_config: Config

    @classmethod
    def from_raw_serie(
            cls,
            raw_serie,
            *,
            smoothing_config=None,
            **kwargs,
    ):
        if smoothing_config is None:
            smoothing_config = BASE_SMOOTHING_CONFIG
        else:
            smoothing_config = StackedConfig([BASE_SMOOTHING_CONFIG, smoothing_config])

        raw_serie = super().from_raw_serie(raw_serie, **kwargs)
        return Smoothing.create_and_process(raw_serie, smoothing_config)

@dataclass
class Serie(SmoothedSerie):
    absolute_speed: np.array

    @classmethod
    def augment_serie_with_data(
        cls,
        serie,
        *,
        absolute_speed,
        **kwargs,
    ):
        return cls(
            x=serie.x,
            y=serie.y,
            z=serie.z,
            config=serie.config,
            status=serie.status,
            absolute_speed=absolute_speed,
            smoothing_config=serie.smoothing_config,
            **kwargs,
        )

    @classmethod
    def from_raw_serie(cls, serie, **kwargs):
        """

        Parameters
        ----------
        data_set : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        serie = super().from_raw_serie(serie, **kwargs)

        if serie.config.distance_type == "euclidean":
            return EuclideanAugmentedSerie.augment_serie_with_data(
                serie,
                absolute_speed=process_euclidean_absolute_speeds(
                    serie,
                    serie.config,
                ),
                **kwargs,
            )
        elif serie.config.distance_type == "angular":
            unitary_gaze_vectors = process_unitary_gaze_vectors(
                serie,
                serie.config,
            )
            return AngularAugmentedSerie.augment_serie_with_data(
                serie,
                absolute_speed=process_angular_absolute_speeds(
                    serie,
                    serie.config,
                    unitary_gaze_vectors=unitary_gaze_vectors,
                ),
                theta_coord=process_angular_coord(
                    serie,
                    serie.config,
                ),
                unitary_gaze_vectors=unitary_gaze_vectors,
                **kwargs,
            )

        raise ValueError(
            f"distance_type {serie.config.distance_type!r} should be either euclidean or angular"
        )


class EuclideanAugmentedSerie(Serie):
    pass

@dataclass
class AngularAugmentedSerie(Serie):
    unitary_gaze_vectors: str
    theta_coord: str

