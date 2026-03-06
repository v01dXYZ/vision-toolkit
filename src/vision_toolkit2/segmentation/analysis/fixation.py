import numpy as np
from numpy.linalg import norm

from .base_analysis import (
    BaseBinarySegmentationAnalysis,
    results_delegation,
    EasyAccessFunction,
)
from ..base_segmentation import Segmentation
from vision_toolkit2.config import Config
import vision_toolkit2.config as c


class FixationAnalysis(BaseBinarySegmentationAnalysis):
    """
    For a fixation [start,end]:
        * positions:      start .. end        (n_samples = end-start+1)
        * speeds:       start .. end-1      (n_vel = n_samples-1)  => slice a_sp[start:end]
    """

    _intervals = results_delegation("fixation_intervals")
    _centroids = results_delegation("fixation_centroids")

    def centroids(self):
        return {
            "centroids": np.asarray(self._centroids()),
        }

    def drift_displacements(self, get_raw=True):
        x_a = self._x()
        y_a = self._y()
        z_a = self._z()

        dist_ = self.SEGMENTATION_CLS.DISTANCES[self._distance_type()]

        dsp = []
        for start, end in self._intervals():
            dsp.append(
                dist_(
                    np.array([x_a[start], y_a[start], z_a[start]]),
                    np.array([x_a[end], y_a[end], z_a[end]]),
                )
            )

        dsp = np.asarray(dsp, dtype=np.float64)
        results = {
            "drift_displacement_mean": float(np.nanmean(dsp)),
            "drift_displacement_sd": self._safe_sd(dsp),
        }

        results["raw"] = dsp

        return results

    def drift_distances(self, get_raw=True):
        x_a = self._x()
        y_a = self._y()
        z_a = self._z()
        n_s = len(x_a)

        dist_ = self.SEGMENTATION_CLS.DISTANCES[self._distance_type()]

        t_cum = []
        stack_ = np.concatenate(
            (
                x_a.reshape((1, n_s)),
                y_a.reshape((1, n_s)),
                z_a.reshape((1, n_s)),
            ),
            axis=0,
        )

        for start, end in self._intervals():
            if end <= start:
                t_cum.append(np.nan)
                continue

            if self._distance_type() == "euclidean":
                l_a = stack_[:, start : end + 1]
                l_c = float(np.nansum(norm(l_a[:, 1:] - l_a[:, :-1], axis=0)))
            else:
                l_c = float(
                    np.nansum(
                        [
                            dist_(
                                np.array([x_a[k], y_a[k], z_a[k]]),
                                np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]]),
                            )
                            for k in range(start, end)
                        ]
                    )
                )

            t_cum.append(l_c)

        t_cum = np.asarray(t_cum, dtype=np.float64)
        results = {
            "drift_cumul_distance_mean": float(np.nanmean(t_cum)),
            "drift_cumul_distance_sd": self._safe_sd(t_cum),
            "raw": t_cum,
        }

        if not get_raw:
            del results["raw"]

        return results

    def drift_velocities(self, get_raw=True, duration_mode="diffs"):
        d_d = np.asarray(
            self.drift_displacements(get_raw=True)["raw"], dtype=np.float64
        )
        n_samples = self._n_samples_per_interval(self._intervals())

        if duration_mode == "samples":
            dur_s = n_samples / self._sampling_frequency()
        else:
            dur_s = np.maximum(n_samples - 1.0, 0.0) / self._sampling_frequency()

        d_vel = np.full_like(d_d, np.nan, dtype=np.float64)
        mask = np.isfinite(d_d) & np.isfinite(dur_s) & (dur_s > 0)
        d_vel[mask] = d_d[mask] / dur_s[mask]

        results = {
            "drift_velocity_mean": float(np.nanmean(d_vel)),
            "drift_velocity_sd": self._safe_sd(d_vel),
            "raw": d_vel,
        }

        if not get_raw:
            del results["raw"]

        return results

    def BCEA(self, BCEA_probability=0.68, get_raw=True):
        def pearson_corr_(x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size < 2:
                return 0.0

            xm = x - np.mean(x)
            ym = y - np.mean(y)

            den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
            if den <= 0:
                return 0.0

            r = float(np.sum(xm * ym) / den)

            return max(min(r, 1.0), -1.0)

        if BCEA_probability is None:
            BCEA_probability = (
                self.segmentation_results.config.fixation.BCEA_probability
            )

        p = float(BCEA_probability)
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        k = -np.log(1.0 - p)

        x_a = np.asarray(self._x(), dtype=np.float64)
        y_a = np.asarray(self._y(), dtype=np.float64)

        bcea_s = []
        for start, end in self._intervals():
            l_x = x_a[start : end + 1]
            l_y = y_a[start : end + 1]

            r = pearson_corr_(l_x, l_y)
            sd_x = (
                float(np.nanstd(l_x, ddof=1)) if np.sum(np.isfinite(l_x)) >= 2 else 0.0
            )
            sd_y = (
                float(np.nanstd(l_y, ddof=1)) if np.sum(np.isfinite(l_y)) >= 2 else 0.0
            )

            inside = max(0.0, 1.0 - r**2)
            l_bcea = 2.0 * np.pi * k * sd_x * sd_y * np.sqrt(inside)
            bcea_s.append(l_bcea)

        bcea_s = np.asarray(bcea_s, dtype=np.float64)
        results = {
            "average_BCEA": float(np.nanmean(bcea_s)),
            "raw": bcea_s,
        }

        if not get_raw:
            del results["raw"]

        return results


easy_access_function = EasyAccessFunction(
    cls=FixationAnalysis,
    common_default_kwargs={
        "get_raw": True,
    },
)


count = easy_access_function(FixationAnalysis.count)
frequency = easy_access_function(FixationAnalysis.frequency)
frequency_wrt_labels = easy_access_function(FixationAnalysis.frequency_wrt_labels)
durations = easy_access_function(FixationAnalysis.durations)
centroids = easy_access_function(FixationAnalysis.centroids)
mean_velocities = easy_access_function(FixationAnalysis.mean_velocities)
average_velocity_means = easy_access_function(
    FixationAnalysis.average_velocity_means,
    config=Config(segmentation=c.Segmentation(fixation=c.Fixation(weighted_average_velocity_means=False))),
)
average_velocity_deviations = easy_access_function(
    FixationAnalysis.average_velocity_deviations
)
drift_displacements = easy_access_function(FixationAnalysis.drift_displacements)
drift_distances = easy_access_function(FixationAnalysis.drift_distances)
drift_velocities = easy_access_function(FixationAnalysis.drift_velocities)
BCEA = easy_access_function(
    FixationAnalysis.BCEA,
    default_kwargs={"BCEA_probability": 0.68},
    config=Config(segmentation=c.Segmentation(fixation=c.Fixation(BCEA_probability=0.68))),
)
