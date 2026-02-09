import pandas as pd
import numpy as np

from dataclasses import dataclass

from .config import Config

from .velocity_distance_factory import (
    process_angular_absolute_speeds, process_angular_coord,
    process_euclidian_absolute_speeds, process_unitary_gaze_vectors)


EPSILON = 1e-3

# @dataclass
# class OcculomotorConfig:
#     distance_projection: int
#     size_plan_x: float
#     size_plan_y: float

#     nb_samples: int

class Serie:
    x: np.array
    y: np.array
    z: np.array
    status: np.array

    config: Config

    def update_config(self, config):
        return type(self)(
            x = self.x,
            y = self.y,
            z = self.z,
            status = self.status,
            config = config,
        )

    def __init__(self, x, y, z, status, config):
        self.x = x.astype("float64")
        self.y = y.astype("float64")
        if z is not None:
            z = z.astype("float64")
        else:
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
            size_plan_x,
            size_plan_y,
            distance_projection = None,
            sampling_frequency = None,
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
            distance_projection = distance_projection,
            size_plan_x = size_plan_x,
            size_plan_y = size_plan_y,
            nb_samples = len(x),
            sampling_frequency = sampling_frequency,
        )

        return cls(
            x = x,
            y = y,
            z = z,
            status = status,
            config = config,
        )

    ###############################
    # MOCK
    ###############################
    def get_data_set(self):
        return {
            "x_array": self.x,
            "y_array": self.y,
            "z_array": self.z,
            "status": self.status
        }


class AugmentedSerie(Serie):
    absolute_speed: np.array

    def __init__(self, absolute_speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.absolute_speed = absolute_speed

    @classmethod
    def augment_serie_with_data(
            cls,
            serie,
            *,
            absolute_speed,
            **kwargs,
    ):
        return cls(
            x = serie.x,
            y = serie.y,
            z = serie.z,
            config = serie.config,
            status = serie.status,
            absolute_speed = absolute_speed, 
            **kwargs,
        )


    @classmethod
    def augment_serie(cls, serie):
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

        # Add it later
        # smoothing = smg.Smoothing(data_set, config)
        # data_set = smoothing.process()
 
        if serie.config.distance_type == "euclidean":
            return EuclideanAugmentedSerie.augment_serie_with_data(
                serie,
                absolute_speed = process_euclidian_absolute_speeds(
                    serie,
                    serie.config,
                ),
            )
        elif serie.config.distance_type == "angular":
            return AngularAugmentedSerie.augment_serie_with_data(
                serie,
                absolute_speed = process_angular_absolute_speeds(
                    serie,
                    serie.config,
                ),
                theta_coord = process_angular_coord(
                    serie,
                    serie.config,
                ),
                unitary_gaze_vectors =  process_unitary_gaze_vectors(
                    serie, 
                    serie.config,
                ),
            )

        assert False

class EuclideanAugmentedSerie(AugmentedSerie):
    pass

class AngularAugmentedSerie(AugmentedSerie):

    def __init__(self, unitary_gaze_vectors, theta_coord,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unitary_gaze_vectors: np.array
        self.theta_coord: np.array

