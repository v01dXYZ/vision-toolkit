#!/bin/env python3
import argparse
import pathlib

from sp_tool.arff_helper import ArffHelper
from sp_tool.evaluate import evaluate
import pandas as pd
import numpy as np

from scipy.io import arff as a

import vision_toolkit as v
import vision_toolkit_test as vt

HANDLABELLER_FINAL = "handlabeller_final"

NOISE_STR = "NOISE"
FIX_STR = "FIX"
SACCADE_STR = "SACCADE"
SP_STR = "SP"


LABELS_ORDINALS = [
    ("NOISE", vt.NOISE),
    ("FIX", vt.FIX),
    ("SACCADE", vt.SACCADE),
    ("SP", vt.SP),
]

LABELS_STR_TO_ORDINAL = dict(LABELS_ORDINALS)
LABELS_ORDINAL_TO_STR = dict(reversed(t) for t in LABELS_ORDINALS)


EYE_MOVEMENT_TYPE = "EYE_MOVEMENT_TYPE"



# def convert_vstk_to_spt(
#     coords,
#     labels,
#     time,
#     width_px,
#     height_px,
#     width_mm,
#     height_mm,
# ):
#     return pd.DataFrame(
#         {
#             "time": time,
#             "x": coords[vt.GAZE_X] * width_px / width_mm,
#             "y": coords[vt.GAZE_Y] * height_px / height_mm,
#             "handlabeller_final": labels.replace(LABELS_ORDINAL_TO_STR),
#         }
#     )

def as_arff_data(x):
    column_dtypes = {
        k: np.dtype("<U8")
        for k, v in x.dtypes.items()
        if isinstance(v, (np.dtypes.ObjectDType, pd.StringDtype))
    }
    ret = {
        "data": x.to_records(
            index=False,
            column_dtypes=column_dtypes,
        ),
    }
    return ret

SORTED_LABELS = sorted([FIX_STR, SACCADE_STR, SP_STR])
    

# def main(cutoff, report_name):
#     P = sorted((pathlib.Path(__file__).parent / "data" / "test").glob(
#         "**/*.arff"
#     ))


#     P_cutoff = [P[i] for i in range(0, len(P), len(P) // cutoff)]
#     print("P_cutoff", P_cutoff)
#     gt_dim_list = [
#         get_ground_truth_df(p) for p in P_cutoff
#     ]

#     report = Hollywood2ReportForEachMethod.evaluate(
#         gt_dim_list=gt_dim_list, # not great but KISS
#     )

#     s = pd.Series({
#         method_name: r["all"]["F1"]
#         for method_name, r in {**report["BINARY"], **report["TERNARY"]}.items()
#     })

#     report_path = pathlib.Path(report_name)

#     s.to_json(report_path.with_suffix(".json"))
#     s.to_markdown(report_path.with_suffix(".md"))
#     s.plot.bar().figure.savefig(report_path.with_suffix(".png"))



class Hollywood2ReportForEachMethod(vt.ReportForEachMethod):
    @classmethod
    def evaluate_predictions(cls, gt_list, pred_list):
        res_stats = {}

        for positive_label in SORTED_LABELS + [None]:
            res_stats[positive_label or "all"] = evaluate(
                gt_list,
                pred_list,
                experts=[HANDLABELLER_FINAL],
                positive_label=positive_label,
            )


        return res_stats

    @classmethod
    def convert_to_vstk(
            cls,
            gt,
            width_px,
            height_px,
            width_mm,
            height_mm,
    ):
        
        gt_data = gt["data"]
        coords = pd.DataFrame(
            {
                vt.GAZE_X: gt_data["x"] * width_mm / width_px,
                vt.GAZE_Y: gt_data["y"] * height_mm / height_px,
            }
        )
        labels = pd.Series(gt_data[HANDLABELLER_FINAL]).replace(LABELS_STR_TO_ORDINAL)
        is_out_of_bounds = (
            (coords[vt.GAZE_X] <= 0)
            | (coords[vt.GAZE_X] > width_mm)
            | (coords[vt.GAZE_Y] <= 0)
            | (coords[vt.GAZE_Y] > height_mm)
        )

        labels[is_out_of_bounds] = vt.NOISE
        coords.loc[labels == vt.NOISE, [vt.GAZE_X, vt.GAZE_Y]] = np.nan

        return coords[[vt.GAZE_X, vt.GAZE_Y]].interpolate(), labels, gt_data["time"]
        
    @classmethod
    def build_predictions_from_results(cls, r, gt, gt_vstk):
        predictions = cls.build_labels_ordinal_from_res(
            r,
            {
                vt.FIXATION_INTERVALS: FIX_STR,
                vt.SACCADE_INTERVALS: SACCADE_STR,
                vt.PURSUIT_INTERVALS: SP_STR,
            },
            default_ordinal=NOISE_STR,
        )

        predictions_sp = np.lib.recfunctions.rename_fields(
            gt["data"].copy(),
            {
                HANDLABELLER_FINAL: EYE_MOVEMENT_TYPE,
            }
        )

        predictions_sp[EYE_MOVEMENT_TYPE] = predictions

        return {"data": predictions_sp}

    @classmethod
    def debug(cls, nary, method_name, gt_list, pred_list):
        return

        import pandas as pd

        print("#"*30)
        print()
        print("NARY:", nary)
        print("METHOD_NAME:", method_name)

        for i, pred in enumerate(pred_list):
            print(f"[{i: 2}] --- ")
            print(pd.Series(pred["data"][EYE_MOVEMENT_TYPE]).value_counts())

    @classmethod
    def summarize_report_into_serie(cls, report):
        return pd.Series({
            method_name: r["all"]["F1"]
            for method_name, r in {**report["BINARY"], **report["TERNARY"]}.items()
        })


# if __name__ == "__main__":
#     arg_parser = argparse.ArgumentParser()

#     arg_parser.add_argument("cutoff", type=int)
#     arg_parser.add_argument("report_name", nargs="?", default="report")

#     args = arg_parser.parse_args()
#     main(args.cutoff, args.report_name)


class EntryPoint(vt.EntryPoint):
    paths = (pathlib.Path(__file__).parent / "data" / "test").glob(
         "**/*.arff"
    )
    ReportForEachMethod = Hollywood2ReportForEachMethod

    @classmethod
    def load_ground_truth_file(cls, fn):
        with open(fn) as f:
            arff_data = ArffHelper.load(f)

        metadata = arff_data["metadata"]

        width_px = metadata["width_px"]
        height_px = metadata["height_px"]
        width_mm = metadata["width_mm"]
        height_mm = metadata["height_mm"]
        distance_mm = metadata["distance_mm"]

        df = pd.DataFrame(arff_data["data"])
        return (
            arff_data,
    #        df[["time", "x", "y", "handlabeller_final"]],
            {
                "width_px": width_px,
                "height_px": height_px,
                "width_mm": width_mm,
                "height_mm": height_mm,
            },
        )
