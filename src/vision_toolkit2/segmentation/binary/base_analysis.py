import numpy as np

from dataclasses import dataclass
from ..base_segmentation import Segmentation

from .binary_segmentation_results import BinarySegmentationResults
import inspect

def results_delegation(attr_name):
    def f(self):
        return getattr(self.binary_segmentation_results, attr_name)

    return f

def config_delegation(attr_name):
    def f(self):
        return getattr(self.binary_segmentation_results.config, attr_name)

    return f

def input_delegation(attr_name):
    def f(self):
        return getattr(self.binary_segmentation_results.input, attr_name)

    return f

@dataclass
class BinarySegmentationAnalysis:
    binary_segmentation_results: BinarySegmentationResults

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
        return a_p[start:end]

    @staticmethod
    def _safe_sd(x):

        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) >= 2 else 0.0

    def _acc_segment(self, start, end):
       
        v = self._speed_segment(start, end)
        if v.size < 2:
            return np.array([], dtype=np.float64)
        return np.abs(np.diff(v)) * self._sampling_frequency()


class EasyAccessFunction:
    def __init__(self, cls, common_default_kwargs):
        self.cls = cls
        self.common_default_kwargs = common_default_kwargs

    def __call__(self,
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
        next(keys) # skip self (or first arg)

        default_kwargs = {k: default_kwargs[k] for k in keys if k in default_kwargs}

        def f(input, *args, **kwargs):
            kwargs = default_kwargs | kwargs 

            if not isinstance(input, self.cls):
                segmentation = Segmentation(input, config)
                segmentation_results = segmentation.process()

                input = self.cls(segmentation_results)

            results = method(input, *args, **kwargs)

            return results

        return f
