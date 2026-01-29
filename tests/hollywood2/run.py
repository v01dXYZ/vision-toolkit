#!/bin/env python3
import pathlib

from sp_tool.arff_helper import ArffHelper
from sp_tool.evaluate import evaluate
import pandas as pd
import numpy as np

from scipy.io import arff as a

import vision_toolkit as v

NOISE = 0
FIX = 1
SACCADE = 2
SP = 3

NOISE_STR = "NOISE"
FIX_STR = "FIX"
SACCADE_STR = "SACCADE"
SP_STR = "SP"


LABELS_ORDINALS = [
    ("NOISE", NOISE),
    ("FIX", FIX),
    ("SACCADE", SACCADE),
    ("SP", SP),
]

LABELS_STR_TO_ORDINAL = dict(LABELS_ORDINALS)
LABELS_ORDINAL_TO_STR = dict(reversed(t) for t in LABELS_ORDINALS)

X_NAME = "gazeX"
Y_NAME = "gazeY"


def get_ground_truth_df(fp):
    with open(fp) as f:
        arff_data = ArffHelper.load(f)

    metadata = arff_data["metadata"]

    width_px = metadata["width_px"]
    height_px = metadata["height_px"]
    width_mm = metadata["width_mm"]
    height_mm = metadata["height_mm"]
    distance_mm = metadata["distance_mm"]

    df = pd.DataFrame(arff_data["data"])
    return (
        df[["time", "x", "y", "handlabeller_final"]],
        {
            "width_px": width_px,
            "height_px": height_px,
            "width_mm": width_mm,
            "height_mm": height_mm,
        },
    )


def convert_spt_to_vstk(
    df,
    width_px,
    height_px,
    width_mm,
    height_mm,
):
    res = pd.DataFrame(
        {
            X_NAME: df["x"] * width_mm / width_px,
            Y_NAME: df["y"] * height_mm / height_px,
            "event_label": df["handlabeller_final"].replace(LABELS_STR_TO_ORDINAL),
        }
    )

    is_out_of_bounds = (
        (res[X_NAME] <= 0)
        | (res[X_NAME] > width_mm)
        | (res[Y_NAME] <= 0)
        | (res[Y_NAME] > height_mm)
    )

    res.loc[is_out_of_bounds, "event_label"] = NOISE
    res.loc[res["event_label"] == NOISE, [X_NAME, Y_NAME]] = np.nan

    return res[[X_NAME, Y_NAME]].interpolate(), res["event_label"], df["time"]


def convert_vstk_to_spt(
    coords,
    labels,
    time,
    width_px,
    height_px,
    width_mm,
    height_mm,
):
    return pd.DataFrame(
        {
            "time": time,
            "x": coords[X_NAME] * width_px / width_mm,
            "y": coords[Y_NAME] * height_px / height_mm,
            "handlabeller_final": labels.replace(LABELS_ORDINAL_TO_STR),
        }
    )


def build_predictions_from_results(r, gt_df):
    predictions = np.full(r.config["nb_samples"], NOISE_STR, dtype="<U8")
    for k, val in [
        ("fixation_intervals", FIX_STR),
        ("saccade_intervals", SACCADE_STR),
        ("pursuit_intervals", SP),
    ]:
        intervals = r.segmentation_results.get(k)

        if intervals is None:
            continue

        for start, end in intervals:
            predictions[start : end + 1] = val

    predictions_df = gt_df.drop(columns=["handlabeller_final"])
    predictions_df["EYE_MOVEMENT_TYPE"] = predictions

    return predictions_df


METHODS_CONFIG = {
    "BINARY": {
        "I_VT": {},
        "I_DeT": {},
        "I_HMM": {},
        "I_2MC": {},
        "I_DeT": {},
        "I_DiT": {},
        "I_HMM": {},
        "I_KF": {},
        "I_MST": {},
        "I_VT": {},
    },
    "TERNARY": {
        "I_BDT": {},
        "I_VDT": {},
        "I_VMP": {},
    },
}


def run_segmentation(fp, nary, method):
    gt_df, dimensions = get_ground_truth_df(fp)

    ds, _, _ = convert_spt_to_vstk(
        gt_df,
        **dimensions,
    )

    if ds["gazeX"].isna().sum() >= 1:
        return None

    if nary == "BINARY":
        Segmentation = v.BinarySegmentation
    elif nary == "TERNARY":
        Segmentation = v.TernarySegmentation
    else:
        raise RuntimeError()

    r = Segmentation(
        ds,
        sampling_frequency=500,
        segmentation_method=method,
        distance_type="euclidean",
        display_segmentation=True,
        size_plan_x=dimensions["width_mm"],
        size_plan_y=dimensions["height_mm"],
        smoothing="savgol",
        savgol_window_length=31,
        savgol_polyorder=3,
        verbose=False,
    )

    r.process()

    predictions_df = build_predictions_from_results(r, gt_df)

    return gt_df, predictions_df


def as_arff_data(x):
    ret = {
        "data": x.to_records(
            index=False,
            column_dtypes={
                k: np.dtype("<U8")
                for k, v in x.dtypes.items()
                if v == np.dtypes.ObjectDType
            },
        ),
    }

    return ret


def main():
    paths = [
        p for p in (pathlib.Path(__file__).parent / "data" / "test").glob(
            "**/*.arff"
        )
    ][:5]

    report = {}
    for nary, methods in METHODS_CONFIG.items():
        report[nary] = {}
        for method_name, config in methods.items():
            gt = []
            pred = []

            for i, p in enumerate(paths):
                g_p = run_segmentation(p, nary, method_name)
                if g_p is None:
                    continue

                g, p = g_p

                pred.append(as_arff_data(p))
                gt.append(as_arff_data(g))

            res_stats = {}
            for positive_label in sorted([FIX_STR, SACCADE_STR, SP_STR]) + [None]:
                res_stats[positive_label or "all"] = evaluate(
                    gt,
                    pred,
                    experts=["handlabeller_final"],
                    positive_label=positive_label,
                )

            report[nary][method_name] = res_stats

    return report

R = main()
R_dict = { method_name: r["all"]["F1"] for method_name, r in {**R["BINARY"], **R["TERNARY"]}.items() }
R_df = pd.Series(R_dict)
    
    


