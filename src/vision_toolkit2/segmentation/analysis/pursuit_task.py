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
    def create_from_results_and_theo_coords(
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

        return np.asarray(recomputed_intervals, dtype=np.int64).reshape(-1, 2)

recalibrated_data_delegation = passthrough_attr("recalibrated_data")

@dataclass
class PursuitTaskAnalysis(PursuitAnalysis):
    df_theo: pd.DataFrame
    recalibrated_data: PursuitTaskRecalibratedData = field(init=False)

    def __post_init__(self):
        self.recalibrated_data = PursuitTaskRecalibratedData.create_from_results_and_theo_coords(
            self.df_theo,
            self.binary_segmentation_results,
            self.binary_segmentation_results.config,
        )
    
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

        return {"entropy": {
            "x": self.ap_entropy(diff_x, w_s, t_eps),
            "y": self.ap_entropy(diff_y, w_s, t_eps),
        }}

    def slope_ratios(self, get_raw=True):
        _ints = self._intervals()
        d_t = 1.0 / self._sampling_frequency()

        pos = {"x": self._x_pursuit(), "y": self._y_pursuit()}
        theo = {"x": self._x_theo_pursuit(), "y": self._y_theo_pursuit()}

        s_r = {"x": [], "y": []}

        for a, b in _ints:
            theo_start = max(0, int(a))
            theo_end = min(len(theo["x"]), int(b) + 1)

            for _dir in ["x", "y"]:
                l_p_e = np.asarray(pos[_dir][int(a) : int(b) + 1], dtype=np.float64)
                l_p_t = np.asarray(theo[_dir][theo_start:theo_end], dtype=np.float64)

                min_len = min(len(l_p_e), len(l_p_t))
                if min_len < 2:
                    s_r[_dir].append(np.nan)
                    continue

                l_p_e = l_p_e[:min_len]
                l_p_t = l_p_t[:min_len]
                l_x = np.arange(min_len, dtype=np.float64) * d_t

                try:
                    slope_e = np.polyfit(l_x, l_p_e, deg=1)[0]
                    slope_t = np.polyfit(l_x, l_p_t, deg=1)[0]
                    s_r[_dir].append(slope_e / slope_t if slope_t != 0 else np.nan)
                except Exception:
                    s_r[_dir].append(np.nan)

        for _dir in ["x", "y"]:
            s_r[_dir] = np.asarray(s_r[_dir], dtype=np.float64)

        return {"slope ratios": s_r}

    def slope_gain(self, _type="weighted"):
        if not self._intervals().size:
            return {"slope gain": {"gain_x": 0.0, "gain_y": 0.0}}

        slope_ratios = self.slope_ratios(get_raw=True)["slope ratios"]

        intervals = np.asarray(self._intervals(), dtype=np.int64)
        durations = (intervals[:, 1] - intervals[:, 0] + 1).astype(np.float64)
        valid_dur = durations > 0

        gains = {}
        for direction in ["x", "y"]:
            ratios = np.asarray(slope_ratios.get(direction, np.array([], dtype=np.float64)), dtype=np.float64)

            if ratios.size != len(self._intervals()):
                gains["gain_" + direction] = 0.0
                continue

            valid = valid_dur & np.isfinite(ratios)
            if not np.any(valid):
                gains["gain_" + direction] = 0.0
                continue

            if _type == "mean":
                gains["gain_" + direction] = float(np.nanmean(ratios[valid]))
            else:
                w = durations[valid]
                r = ratios[valid]
                denom = float(np.sum(w))
                gains["gain_" + direction] = float(np.sum(w * r) / denom) if denom > 0 else 0.0

        return {"slope gain": gains}

    def overall_gain(self, get_raw=True):
        d_t = 1.0 / self._sampling_frequency()

        pos = {"x": self._x_pursuit(), "y": self._y_pursuit()}
        theo = {"x": self._x_theo_pursuit(), "y": self._y_theo_pursuit()}

        gains = []

        for a, b in self._intervals():
            a = int(a)
            b = int(b)
            if b <= a:
                continue

            xe = pos["x"][a:b+1]
            ye = pos["y"][a:b+1]
            xt = theo["x"][a:b+1]
            yt = theo["y"][a:b+1]

            min_len = min(len(xe), len(ye), len(xt), len(yt))
            if min_len < 2:
                continue

            xe = xe[:min_len]
            ye = ye[:min_len]
            xt = xt[:min_len]
            yt = yt[:min_len]

            se_x = xe[1:] - xe[:-1]
            se_y = ye[1:] - ye[:-1]
            st_x = xt[1:] - xt[:-1]
            st_y = yt[1:] - yt[:-1]

            e_vel = np.sqrt((se_x / d_t) ** 2 + (se_y / d_t) ** 2)
            t_vel = np.sqrt((st_x / d_t) ** 2 + (st_y / d_t) ** 2)

            with np.errstate(divide="ignore", invalid="ignore"):
                l_g = e_vel / t_vel
            gains += list(l_g[np.isfinite(l_g)])

        gains = np.asarray(gains, dtype=np.float64)

        return {"overall_gain": float(np.nanmean(gains)) if gains.size else np.nan}

    def overall_gain_x(self, get_raw=True):
        d_t = 1.0 / self._sampling_frequency()

        xe_all = self._x_pursuit()
        xt_all = self._x_theo_pursuit()

        gains = []

        for a, b in self._intervals():
            a = int(a)
            b = int(b)
            if b <= a:
                continue

            xe = xe_all[a:b+1]
            xt = xt_all[a:b+1]

            min_len = min(len(xe), len(xt))
            if min_len < 2:
                continue

            xe = xe[:min_len]
            xt = xt[:min_len]

            se = xe[1:] - xe[:-1]
            st = xt[1:] - xt[:-1]

            e_vel = se / d_t
            t_vel = st / d_t

            with np.errstate(divide="ignore", invalid="ignore"):
                l_g = e_vel / t_vel
            gains += list(l_g[np.isfinite(l_g)])

        gains = np.asarray(gains, dtype=np.float64)

        return {"overall_gain_x": float(np.nanmean(gains)) if gains.size else np.nan}

    def overall_gain_y(self, get_raw=True):
        d_t = 1.0 / self._sampling_frequency()

        ye_all = self._y_pursuit()
        yt_all = self._y_theo_pursuit()

        gains = []

        for a, b in self._intervals():
            a = int(a)
            b = int(b)
            if b <= a:
                continue

            ye = ye_all[a:b+1]
            yt = yt_all[a:b+1]

            min_len = min(len(ye), len(yt))
            if min_len < 2:
                continue

            ye = ye[:min_len]
            yt = yt[:min_len]

            se = ye[1:] - ye[:-1]
            st = yt[1:] - yt[:-1]

            e_vel = se / d_t
            t_vel = st / d_t

            with np.errstate(divide="ignore", invalid="ignore"):
                l_g = e_vel / t_vel
            gains += list(l_g[np.isfinite(l_g)])

        gains = np.asarray(gains, dtype=np.float64)

        return {"overall_gain_y": float(np.nanmean(gains)) if gains.size else np.nan}

    def sinusoidal_phase(self):
        import scipy.optimize

        def fit_sin(tt, yy):
            tt = np.asarray(tt, dtype=np.float64)
            yy = np.asarray(yy, dtype=np.float64)

            if tt.size < 3 or yy.size < 3:
                raise ValueError("Not enough samples")

            ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))
            Fyy = np.abs(np.fft.fft(yy))
            guess_freq = np.abs(ff[np.argmax(Fyy[1:]) + 1]) if len(Fyy) > 1 else 1.0
            guess_amp = np.std(yy) * np.sqrt(2.0)
            guess_offset = np.mean(yy)
            guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset], dtype=np.float64)

            def sinfunc(t, A, w, p, c):
                return A * np.sin(w * t + p) + c

            popt, _ = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=20000)
            A, w, p, c = popt
            f = w / (2.0 * np.pi)

            return {"phase": p, "freq": f}

        s_f = self._sampling_frequency()
        d_t = 1.0 / s_f

        x_t = self._x_theo_pursuit()
        x_e = self._x_pursuit()

        n = min(x_t.size, x_e.size)
        if n < 3:
            return {"phase_difference": np.nan}

        x_t = x_t[:n]
        x_e = x_e[:n]

        tt = np.linspace(0.0, (n - 1) * d_t, n, dtype=np.float64)

        try:
            res1 = fit_sin(tt, x_t)
            res2 = fit_sin(tt, x_e)
        except Exception:
            return {"phase_difference": np.nan}

        return {"phase_difference": float(res1["phase"] - res2["phase"])}

    def crossing_time(self, tolerance=1.0):
        s_f = self._sampling_frequency()
        n = self._nb_samples_pursuit()
        time = np.arange(n, dtype=np.float64) / s_f

        crossings = {"x": None, "y": None}

        for direction in ["x", "y"]:
            eye_pos = np.asarray(self._x_pursuit() if direction == "x" else self._y_pursuit(), dtype=np.float64)
            target_pos = np.asarray(self._x_theo_pursuit() if direction == "x" else self._y_theo_pursuit(), dtype=np.float64)

            min_len = min(len(eye_pos), len(target_pos), len(time))
            if min_len <= 0:
                continue

            eye_pos = eye_pos[:min_len]
            target_pos = target_pos[:min_len]
            tt = time[:min_len]

            error = np.abs(eye_pos - target_pos)
            idx = np.where(error < float(tolerance))[0]
            if idx.size > 0:
                crossings[direction] = float(tt[idx[0]])

        return {"crossing_time": crossings}

    def accuracy(self, pursuit_accuracy_tolerance=0.15, _type="mean"):
        if not self._intervals().size:
            return {"x": 0.0, "y": 0.0}

        intervals = np.asarray(self._intervals(), dtype=np.int64)
        durations = (intervals[:, 1] - intervals[:, 0] + 1).astype(np.float64)
        valid_dur = durations > 0

        s_r = self.slope_ratios(get_raw=True)["slope ratios"]
        ac_t = float(pursuit_accuracy_tolerance)

        out = {}
        for _dir in ["x", "y"]:
            ratios = np.asarray(s_r.get(_dir, np.array([], dtype=np.float64)), dtype=np.float64)
            if ratios.size != len(self._intervals()):
                out[_dir] = 0.0
                continue

            within = np.where((ratios < 1.0 + ac_t) & (ratios > 1.0 - ac_t), 1.0, 0.0)
            within = np.where(np.isfinite(ratios), within, 0.0)

            if _type == "weighted":
                mask = valid_dur & np.isfinite(ratios)
                denom = float(np.sum(durations[mask])) if np.any(mask) else 0.0
                out[_dir] = float(np.sum(within[mask] * durations[mask]) / denom) if denom > 0 else 0.0
            else:
                out[_dir] = float(np.mean(within)) if within.size else 0.0

        return {"accuracy": out}


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
