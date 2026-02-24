import numpy as np

from dataclasses import dataclass
from ..base_segmentation import Segmentation
from ..binary.binary_segmentation_results import BinarySegmentationResults
from vision_toolkit2.config import StackedConfig
import inspect

def passthrough_attr(*passthrough_attr):
    def delegate(attr):
        def f(self):
            passthrough = self
            for intermediate_attr in passthrough_attr:
                passthrough = getattr(passthrough, intermediate_attr)

            value = getattr(passthrough, attr)

            return value

        return f

    return delegate

results_delegation = passthrough_attr("segmentation_results")
config_delegation = passthrough_attr("segmentation_results", "config")
input_delegation = passthrough_attr("segmentation_results", "input")

@dataclass
class BaseBinarySegmentationAnalysis:
    segmentation_results: BinarySegmentationResults

    _absolute_speed = input_delegation("absolute_speed")

    _x = input_delegation("x")
    _y = input_delegation("y")
    _z = input_delegation("z")
    _is_labeled = results_delegation("is_labeled")

    _nb_samples = config_delegation("nb_samples")
    _distance_type = config_delegation("distance_type")
    _sampling_frequency = config_delegation("sampling_frequency")

    @staticmethod
    def _n_samples_per_interval(intervals):
        a_i = np.asarray(intervals, dtype=np.int64)
        return (a_i[:, 1] - a_i[:, 0] + 1).astype(np.float64)

    def _speed_segment(self, start, end):
        if end <= start:
            return np.array([], dtype=np.float64)

        a_sp = self._absolute_speed()
        return a_sp[start:end]

    def _acc_segment(self, start, end):
        v = self._speed_segment(start, end)
        if v.size < 2:
            return np.array([], dtype=np.float64)
        return np.abs(np.diff(v)) * self._sampling_frequency()

    @staticmethod
    def _safe_sd(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) >= 2 else 0.0

    def count(self):
        return {
            "count": int(len(self._intervals())),
        }

    def frequency(self):
        ct = len(self._intervals())
        denom = self._nb_samples() / self._sampling_frequency()
        f = ct / denom if denom > 0 else np.nan

        return {
            "frequency": float(f),
        }

    def frequency_wrt_labels(self):
        ct = len(self._intervals())
        labeled = float(np.sum(self._is_labeled()))
        denom = labeled / self._sampling_frequency()

        f = ct / denom if denom > 0 else np.nan

        return {
            "frequency": float(f),
        }

    def durations(self, get_raw=True):
        a_i = np.asarray(self._intervals(), dtype=np.int64)
        a_d = (a_i[:, 1] - a_i[:, 0] + 1) / self._sampling_frequency()

        results = {
            "duration_mean": float(np.nanmean(a_d)),
            "duration_sd": self._safe_sd(a_d),
        }
        if get_raw:
            results["raw"] = a_d

        return results

    def mean_velocities(self):
        m_sp, sd_sp = [], []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)
            m_sp.append(float(np.nanmean(seg)) if seg.size else np.nan)
            sd_sp.append(self._safe_sd(seg) if seg.size else np.nan)

        return {
            "velocity_means": np.asarray(m_sp, dtype=np.float64),
            "velocity_sd": np.asarray(sd_sp, dtype=np.float64),
        }

    def average_velocity_means(self, weighted=None, get_raw=True, weight_mode="diffs"):
        m_sp = self.mean_velocities()["velocity_means"]

        if not weighted:
            results = {"average_velocity_means": float(np.nanmean(m_sp)), "raw": m_sp}

            if not get_raw:
                del results["raw"]

            return results

        n_samples = self._n_samples_per_interval(self._intervals())
        if weight_mode == "samples":
            w = np.maximum(n_samples, 0.0)
        else:
            w = np.maximum(n_samples - 1.0, 0.0)

        denom = np.nansum(w)
        w_v = float(np.nansum(w * m_sp) / denom) if denom > 0 else np.nan

        results = {"weighted_average_velocity_means": w_v, "raw": m_sp}

        if not get_raw:
            del results["raw"]

        return results

    def average_velocity_deviations(self, get_raw=True, weight_mode="diffs"):
        sd_sp = self.mean_velocities()["velocity_sd"]

        n_samples = self._n_samples_per_interval(self._intervals())
        if weight_mode == "samples":
            w = np.maximum(n_samples, 0.0)
        else:
            w = np.maximum(n_samples - 1.0, 0.0)

        denom = np.nansum(w)
        a_sd = (
            float(np.sqrt(np.nansum(w * (sd_sp**2)) / denom)) if denom > 0 else np.nan
        )

        results = {"average_velocity_sd": a_sd, "raw": sd_sp}

        if not get_raw:
            del results["raw"]

        return results


class EasyAccessFunction:
    def __init__(self, cls, common_default_kwargs):
        self.cls = cls
        self.common_default_kwargs = common_default_kwargs

    def __call__(
        self,
        method,
        default_kwargs=None,
        config=None,
    ):
        if default_kwargs:
            default_kwargs = self.common_default_kwargs | default_kwargs
        else:
            default_kwargs = self.common_default_kwargs

        method_signature = inspect.signature(method)

        keys = iter(method_signature.parameters.keys())
        next(keys)  # skip self (or first arg)

        default_kwargs = {k: default_kwargs[k] for k in keys if k in default_kwargs}

        base_config = config

        def f(input, *args, config=None, **kwargs):
            kwargs = default_kwargs | kwargs

            if config is None:
                config = base_config
            elif base_config is not None:
                config = StackedConfig([base_config, config])

            if not isinstance(input, self.cls):
                segmentation = Segmentation(input, config=config)
                segmentation_results = segmentation.process()

                input = self.cls(segmentation_results, *args)

            results = method(input, **kwargs)

            return results

        return f


BinarySegmentationAnalysis = BaseBinarySegmentationAnalysis
