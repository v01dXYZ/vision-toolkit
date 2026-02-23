#!/bin/env python

from vision_toolkit2.segmentation.analysis import (
    fixation as fa2,
    saccade as sa2,
    pursuit as pu2,
)
from vision_toolkit2 import Serie, Config

# from vision_toolkit import FixationAnalysis, SaccadeAnalysis, PursuitAnalysis
from vision_toolkit.oculomotor.segmentation_based import (
    fixation as fa1,
    saccade as sa1,
    pursuit as pu1,
)

data_file = "../documentation/Documentaion_VT/dataset/DS_Hollywood2/gaze_s21.csv"
method = "I_VT"
ternary_method = "I_VVT"
sampling_frequency = 1000

KWARGS = {
    "size_plan_x": 512,
    "size_plan_y": 512,
    "distance_type": "euclidean",
    "sampling_frequency": sampling_frequency,
}

serie2 = Serie.read_csv(
    data_file,
    **KWARGS,
)

METHODS_PER_EVENT = {
    "fixation": [
        "count",
        "frequency",
        "frequency",
        "durations",
        "centroids",
        "mean_velocities",
        "average_velocity_means",
        "average_velocity_deviations",
        "drift_displacements",
        "drift_distances",
        "drift_velocities",
        "BCEA",
    ],
    "saccade": [
        "count",
        "frequency",
        "frequency_wrt_labels",
        "durations",
        "amplitudes",
        "travel_distances",
        "efficiencies",
        "directions",
        "horizontal_deviations",
        "successive_deviations",
        "initial_directions",
        "initial_deviations",
        "max_curvatures",
        "area_curvatures",
        "mean_velocities",
        "average_velocity_means",
        "average_velocity_deviations",
        "peak_velocities",
        "mean_acceleration_profiles",
        "mean_accelerations",
        "mean_decelerations",
        "average_acceleration_profiles",
        "average_acceleration_means",
        "average_deceleration_means",
        "peak_accelerations",
        "peak_decelerations",
        "skewness_exponents",
        "gamma_skewness_exponents",
        "amplitude_duration_ratios",
        "peak_velocity_amplitude_ratios",
        "peak_velocity_duration_ratios",
        "peak_velocity_velocity_ratios",
        "acceleration_deceleration_ratios",
        "main_sequence",
    ],
    "pursuit": [
        "count",
        "frequency",
        "durations",
        "proportion",
        "velocity",
        "velocity_means",
        "peak_velocity",
        "amplitude",
        "distance",
        "efficiency",
    ],
}

MODULE_PER_EVENT = {
    "fixation": (fa1, fa2),
    "saccade": (sa1, sa2),
    "pursuit": (pu1, pu2),
}

SEGMENTATION_METHOD_PER_EVENT = {
    "fixation": "I_VT",
    "saccade": "I_VT",
    "pursuit": "I_VVT",
}
RES = {}
for event_name, methods in METHODS_PER_EVENT.items():
    (mod1, mod2) = MODULE_PER_EVENT[event_name]
    segmentation_method = SEGMENTATION_METHOD_PER_EVENT[event_name]

    RES[event_name] = {}
    for method in methods:
        attr1 = f"{event_name}_{method}"
        attr2 = method

        fun1 = getattr(mod1, attr1)
        fun2 = getattr(mod2, attr2)

        res1 = fun1(data_file, **KWARGS, segmentation_method=segmentation_method)
        res2 = fun2(serie2, config=Config(segmentation_method=segmentation_method))

        RES[event_name][method] = (res1, res2)
