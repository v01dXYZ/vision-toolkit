import numpy as np

from .base_analysis import BaseBinarySegmentationAnalysis, passthrough_attr, results_delegation, input_delegation, config_delegation, EasyAccessFunction
from .pursuit import PursuitAnalysis

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class PursuitTaskRecalibratedData:
    x_pursuit: np.ndarray
    y_pursuit: np.ndarray
    x_theo_pursuit: np.ndarray
    y_theo_pursuit: np.ndarray
    recomputed_intervals: np.ndarray

    @classmethod
    def create_from_results_and_theorical_coords(
        cls,
        df_theo: pd.DataFrame,
        ternary_segmentation_results,
        config,
    ):
        x_theo = np.asarray(df_theo.iloc[:, 0], dtype=np.float64)
        y_theo = np.asarray(df_theo.iloc[:, 1], dtype=np.float64)
        n_theo = int(len(x_theo))

        start_idx = int(config.pursuit_start_idx)
        start_idx = max(start_idx, 0)

        x_view = ternary_segmentation_results.input.x
        y_view = ternary_segmentation_results.input.y
        nb_samples = int(config.nb_samples)
        end_idx = min(start_idx + n_theo, nb_samples)

        n_win = max(0, end_idx - start_idx)
        x_theo = x_theo[:n_win]
        y_theo = y_theo[:n_win]

        end_idx = min(start_idx + n_win, len(x_view), len(y_view))
        n_valid = end_idx - start_idx

        if n_valid <= 0:
            x_win = np.array([], dtype=np.float64)
            y_win = np.array([], dtype=np.float64)
            x_theo_win = np.array([], dtype=np.float64)
            y_theo_win = np.array([], dtype=np.float64)
        else:
            x_win = x_view[start_idx:end_idx]
            y_win = y_view[start_idx:end_idx]
            x_theo_win = np.asarray(x_theo[:n_valid], dtype=np.float64)
            y_theo_win = np.asarray(y_theo[:n_valid], dtype=np.float64)

        return cls(
            x_pursuit=x_win,
            y_pursuit=y_win,
            x_theo_pursuit=x_theo_win,
            y_theo_pursuit=y_theo_win,
            recomputed_intervals=cls._reproject_intervals_to_window(
                ternary_segmentation_results.pursuit_intervals,
                start_idx,
                end_idx,
            ),
        )

    @staticmethod
    def _reproject_intervals_to_window(intervals, start_idx, end_idx):
        recomputed_intervals = []

        for a, b in intervals:
            a = int(a)
            b = int(b)

            aa = max(a, start_idx)
            bb = min(b, end_idx - 1)

            if bb >= aa:
                recomputed_intervals.append([aa - start_idx, bb - start_idx])

        return np.asarray(recomputed_intervals, dtype=np.int64)

recalibrated_data_delegation = passthrough_attr("recalibrated_data")


class PursuitTaskAnalysis(PursuitAnalysis):
    df_theo: pd.DataFrame    
    recalibrated_data: PursuitTaskRecalibratedData = field(init=False)
    
    _intervals = recalibrated_data_delegation("recomputed_intervals")
    _x_pursuit = recalibrated_data_delegation("x_pursuit")
    _y_pursuit = recalibrated_data_delegation("y_pursuit")
    _x_theo_pursuit = recalibrated_data_delegation("x_theo_pursuit")
    _y_theo_pursuit = recalibrated_data_delegation("y_theo_pursuit")
    _nb_samples_pursuit = config_delegation("nb_samples_pursuit")

    def ap_entropy(self, diff_vec, w_s, t_eps):
        diff_vec = np.asarray(diff_vec, dtype=np.float64)
        n_s = len(diff_vec)
        w_s = int(w_s)
        t_eps = float(t_eps)

        if n_s <= w_s + 1 or w_s < 1:
            return np.nan

        x_m = np.zeros((n_s - w_s + 1, w_s), dtype=np.float64)
        x_mp = np.zeros((n_s - w_s, w_s + 1), dtype=np.float64)

        for i in range(n_s - w_s + 1):
            x_m[i] = diff_vec[i : i + w_s]
            if i < n_s - w_s:
                x_mp[i] = diff_vec[i : i + w_s + 1]

        C_m = np.zeros(n_s - w_s + 1, dtype=np.float64)
        C_mp = np.zeros(n_s - w_s, dtype=np.float64)

        for i in range(n_s - w_s + 1):
            d = np.abs(x_m - x_m[i])
            d_m = np.sum(np.max(d, axis=1) < t_eps)
            C_m[i] = d_m / (n_s - w_s + 1)

        for i in range(n_s - w_s):
            d = np.abs(x_mp - x_mp[i])
            d_mp = np.sum(np.max(d, axis=1) < t_eps)
            C_mp[i] = d_mp / (n_s - w_s)

        C_m = np.clip(C_m, 1e-12, None)
        C_mp = np.clip(C_mp, 1e-12, None)

        entropy = np.sum(np.log(C_m)) / len(C_m) - np.sum(np.log(C_mp)) / len(C_mp)

        return float(entropy)

    def entropy(self, pursuit_entropy_window=10, pursuit_entropy_tolerance=0.1, get_raw=True):
        w_s = int(pursuit_entropy_window)
        t_eps = float(pursuit_entropy_tolerance)

        x_e = np.asarray(self._x_pursuit(), dtype=np.float64)
        y_e = np.asarray(self._y_pursuit(), dtype=np.float64)
        x_t = np.asarray(self._x_theo_pursuit(), dtype=np.float64)
        y_t = np.asarray(self._y_theo_pursuit(), dtype=np.float64)

        n = min(x_e.size, y_e.size, x_t.size, y_t.size)
        if n < 3:
            return {"x": np.nan, "y": np.nan}

        x_e = x_e[:n]
        y_e = y_e[:n]
        x_t = x_t[:n]
        y_t = y_t[:n]

        s_f = self._sampling_frequency()

        sp_e_x = np.zeros(n, dtype=np.float64)
        sp_e_y = np.zeros(n, dtype=np.float64)
        sp_t_x = np.zeros(n, dtype=np.float64)
        sp_t_y = np.zeros(n, dtype=np.float64)

        sp_e_x[:-1] = (x_e[1:] - x_e[:-1]) * s_f
        sp_e_y[:-1] = (y_e[1:] - y_e[:-1]) * s_f
        sp_t_x[:-1] = (x_t[1:] - x_t[:-1]) * s_f
        sp_t_y[:-1] = (y_t[1:] - y_t[:-1]) * s_f

        diff_x = sp_e_x - sp_t_x
        diff_y = sp_e_y - sp_t_y

        return {
            "x": self.ap_entropy(diff_x, w_s, t_eps),
            "y": self.ap_entropy(diff_y, w_s, t_eps),
        }


easy_access_function = EasyAccessFunction(
    cls=PursuitTaskAnalysis,
    common_default_kwargs={
        "get_raw": True,
    },
)

count = easy_access_function(PursuitTaskAnalysis.count)
frequency = easy_access_function(PursuitTaskAnalysis.frequency)
durations = easy_access_function(PursuitTaskAnalysis.durations)
proportion = easy_access_function(PursuitTaskAnalysis.proportion)
velocity = easy_access_function(PursuitTaskAnalysis.velocity)
velocity_means = easy_access_function(PursuitTaskAnalysis.velocity_means)
peak_velocity = easy_access_function(PursuitTaskAnalysis.peak_velocity)
amplitude = easy_access_function(PursuitTaskAnalysis.amplitude)
distance = easy_access_function(PursuitTaskAnalysis.distance)
efficiency = easy_access_function(PursuitTaskAnalysis.efficiency)
slope_ratios = easy_access_function(PursuitTaskAnalysis.slope_ratios)
slope_gain = easy_access_function(PursuitTaskAnalysis.slope_gain)
crossing_time = easy_access_function(PursuitTaskAnalysis.crossing_time)
overall_gain = easy_access_function(PursuitTaskAnalysis.overall_gain)
overall_gain_x = easy_access_function(PursuitTaskAnalysis.overall_gain_x)
overall_gain_y = easy_access_function(PursuitTaskAnalysis.overall_gain_y)
sinusoidal_phase = easy_access_function(PursuitTaskAnalysis.sinusoidal_phase)
accuracy = easy_access_function(PursuitTaskAnalysis.accuracy)
entropy = easy_access_function(PursuitTaskAnalysis.entropy)
