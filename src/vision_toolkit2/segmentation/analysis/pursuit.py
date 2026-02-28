import numpy as np

from .base_analysis import (
    BaseAnalysis,
    results_delegation,
    EasyAccessFunction,
)
from ..base_segmentation import Segmentation


class PursuitAnalysis(BaseAnalysis):
    """
    For a pursuit [start,end]:
        * positions:  start .. end        (n_samples = end-start+1)
        * speeds:     start .. end-1      (n_vel = n_samples-1) => slice a_sp[start:end]
    """

    _intervals = results_delegation("pursuit_intervals")

    def proportion(self):
        a_i = np.asarray(self._intervals(), dtype=np.int64)
        if a_i.size == 0:
            return 0.0
        n_p = float(np.sum(a_i[:, 1] - a_i[:, 0] + 1))
        denom = float(self._nb_samples())

        return float(n_p / denom) if denom > 0 else np.nan

    def velocity(self, get_raw=True):
        l_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)  # start..end-1
            if seg.size:
                l_sp.append(seg)

        all_sp = np.concatenate(l_sp) if len(l_sp) else np.array([], dtype=np.float64)

        results = {
            "velocity_mean": float(np.nanmean(all_sp)) if all_sp.size else np.nan,
            "velocity_sd": self._safe_sd(all_sp) if all_sp.size else np.nan,
            "raw": all_sp,
        }
        if not get_raw:
            del results["raw"]
        return results

    def velocity_means(self, get_raw=True):
        m_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)  # start..end-1
            m_sp.append(float(np.nanmean(seg)) if seg.size else np.nan)

        m_sp = np.asarray(m_sp, dtype=np.float64)

        results = {
            "velocity_mean_mean": float(np.nanmean(m_sp)),
            "velocity_mean_sd": self._safe_sd(m_sp),
            "raw": m_sp,
        }
        if not get_raw:
            del results["raw"]
        return results

    def peak_velocity(self, get_raw=True):
        p_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)  # start..end-1
            p_sp.append(float(np.nanmax(seg)) if seg.size else np.nan)

        p_sp = np.asarray(p_sp, dtype=np.float64)

        results = {
            "velocity_peak_mean": float(np.nanmean(p_sp)),
            "velocity_peak_sd": self._safe_sd(p_sp),
            "raw": p_sp,
        }
        if not get_raw:
            del results["raw"]
        return results

    def amplitude(self, get_raw=True):
        x_a = self._x()
        y_a = self._y()
        z_a = self._z()
        dist_ = Segmentation.DISTANCES[self._distance_type()]

        dsp = []
        for start, end in self._intervals():
            s_p = np.array([x_a[start], y_a[start], z_a[start]])
            e_p = np.array([x_a[end], y_a[end], z_a[end]])
            dsp.append(dist_(s_p, e_p))

        dsp = np.asarray(dsp, dtype=np.float64)

        results = {
            "pursuit_amplitude_mean": float(np.nanmean(dsp)),
            "pursuit_amplitude_sd": self._safe_sd(dsp),
            "raw": dsp,
        }
        if not get_raw:
            del results["raw"]
        return results

    def distance(self, get_raw=True):
        x_a = self._x()
        y_a = self._y()
        z_a = self._z()
        dist_ = Segmentation.DISTANCES[self._distance_type()]

        t_cum = []
        for start, end in self._intervals():
            if end <= start:
                t_cum.append(np.nan)
                continue

            l_cum = 0.0
            for k in range(start, end):
                s_p = np.array([x_a[k], y_a[k], z_a[k]])
                e_p = np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]])
                l_cum += dist_(s_p, e_p)

            t_cum.append(l_cum)

        t_cum = np.asarray(t_cum, dtype=np.float64)

        results = {
            "pursuit_cumul_distance_mean": float(np.nanmean(t_cum)),
            "pursuit_cumul_distance_sd": self._safe_sd(t_cum),
            "raw": t_cum,
        }
        if not get_raw:
            del results["raw"]
        return results

    def efficiency(self, get_raw=True):
        x_a = self._x()
        y_a = self._y()
        z_a = self._z()
        dist_ = Segmentation.DISTANCES[self._distance_type()]

        eff = []
        for start, end in self._intervals():
            if end <= start:
                eff.append(np.nan)
                continue

            s_p = np.array([x_a[start], y_a[start], z_a[start]])
            e_p = np.array([x_a[end], y_a[end], z_a[end]])
            amp = dist_(s_p, e_p)

            l_cum = 0.0
            for k in range(start, end):
                p0 = np.array([x_a[k], y_a[k], z_a[k]])
                p1 = np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]])
                l_cum += dist_(p0, p1)

            eff.append(amp / l_cum if l_cum > 0 else np.nan)

        eff = np.asarray(eff, dtype=np.float64)

        results = {
            "pursuit_efficiency_mean": float(np.nanmean(eff)),
            "pursuit_efficiency_sd": self._safe_sd(eff),
            "raw": eff,
        }
        if not get_raw:
            del results["raw"]
        return results


easy_access_function = EasyAccessFunction(
    cls=PursuitAnalysis,
    common_default_kwargs={
        "get_raw": True,
    },
)


count = easy_access_function(PursuitAnalysis.count)
frequency = easy_access_function(PursuitAnalysis.frequency)
durations = easy_access_function(PursuitAnalysis.durations)
proportion = easy_access_function(PursuitAnalysis.proportion)
velocity = easy_access_function(PursuitAnalysis.velocity)
velocity_means = easy_access_function(PursuitAnalysis.velocity_means)
peak_velocity = easy_access_function(PursuitAnalysis.peak_velocity)
amplitude = easy_access_function(PursuitAnalysis.amplitude)
distance = easy_access_function(PursuitAnalysis.distance)
efficiency = easy_access_function(PursuitAnalysis.efficiency)
