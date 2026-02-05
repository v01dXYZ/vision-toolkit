import argparse
import pathlib

import numpy as np
import pandas as pd

import vision_toolkit as v

NOISE = 0
FIX = 1
SACCADE = 2
SP = 3

FIXATION_INTERVALS = "fixation_intervals"
SACCADE_INTERVALS = "saccade_intervals"
PURSUIT_INTERVALS = "pursuit_intervals"

GAZE_X = "gazeX"
GAZE_Y = "gazeY"

EVENT_LABEL = "event_label"

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


def normalize_report(d):
    """
    Sort the dict keys and convert any Numpy object into a Python one.
    The output should normally be JSON-dumpable.
    """
    d = {k: d[k] for k in sorted(d.keys())}

    return {
        k: getattr(v, "item")()
        if getattr(v, "item", None) is not None
        else (normalize_report(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }

class ReportForEachMethod:
    @classmethod
    def run_segmentation(
            cls,
            gt,
            gt_vstk,
            dimensions,
            nary,
            method,
            segmentation_kwargs={},
    ):
        gt_coords, gt_labels, gt_time = gt_vstk

        if gt_coords[GAZE_X].isna().sum() >= 1:
            return None

        if nary == "BINARY":
            Segmentation = v.BinarySegmentation
        elif nary == "TERNARY":
            Segmentation = v.TernarySegmentation
        else:
            raise RuntimeError()

        kwargs = {
            "sampling_frequency": 500,
            "segmentation_method": method,
            "distance_type": "euclidean",
            "display_segmentation": True,
            "size_plan_x": dimensions["width_mm"],
            "size_plan_y": dimensions["height_mm"],
            "smoothing": "savgol",
            "savgol_window_length": 31,
            "savgol_polyorder": 3,
            "verbose": False,
            **segmentation_kwargs,
        }

        r = Segmentation(
            gt_coords,
            **kwargs,
        )

        r.process()

        return cls.build_predictions_from_results(r, gt, gt_vstk)

    def build_labels_ordinal_from_res(
            r,
            key_ordinal,
            default_ordinal,
    ):
        if not isinstance(key_ordinal, dict):
            key_ordinal = dict(key_ordinal)

        if isinstance(default_ordinal, str):
            dtype = "<U8"
        elif isinstance(default_ordinal, int):
            dtype = int
        else:
            raise ValueError()

        predictions = np.full(
            r.config["nb_samples"],
            default_ordinal,
            dtype=dtype,
        )
        for k, val in key_ordinal.items():
            intervals = r.segmentation_results.get(k)

            if intervals is None:
                continue

            for start, end in intervals:
                predictions[start : end + 1] = val

        return predictions
    

    @classmethod
    def evaluate(cls, gt_dim_list):
        # first we convert to VSTK ground truth
        gt_dim_list = [
            (gt, dim, gt_vstk)
            for gt, dim in gt_dim_list
            if (gt_vstk := cls.convert_to_vstk(gt, **dim)) is not None
        ]

        report = {}

        for nary, methods in METHODS_CONFIG.items():
            report[nary] = {}
            for method_name, config in methods.items():
                gt_list = []
                pred_list = []

                for gt, dimensions, gt_vstk in gt_dim_list:
                    pred = cls.run_segmentation(
                        gt,
                        gt_vstk,
                        dimensions,
                        nary,
                        method_name,
                    )
                    if pred is None:
                        continue

                    pred_list.append(pred)
                    gt_list.append(gt)

                cls.debug(nary, method_name, gt_list, pred_list)
                report[nary][method_name] = cls.evaluate_predictions(
                    gt_list,
                    pred_list,
                )

        return normalize_report(report), cls.summarize_report_into_serie(report)

    @classmethod
    def debug(cls, nary, method_name, gt_list, pred_list): 
        pass

    @classmethod
    def evaluate_predictions(cls, gt, pred):
        """
        evaluate gt and pred (test format) and typicall return a report
        """
        pass

    @classmethod
    def convert_to_vstk(
            cls,
            data,
            width_px,
            height_px,
            width_mm,
            height_mm,
    ):
        """
        convert test format to vstk
        """
        pass

    @classmethod
    def build_predictions_from_results(cls, r, gt, gt_vstk):
        """
        Using both ground truth and result (test format), build predictions data as test format
        """
        pass


class VSTKReportForEachMethod(ReportForEachMethod):
    """
    Test format is already vstk
    """

    @classmethod
    def convert_to_vstk(
            cls,
            gt,
            width_px,
            height_px,
            width_mm,
            height_mm,
    ):
        return (*gt, None)

    @classmethod
    def build_predictions_from_results(cls, r, gt, gt_vstk):
        pred_np = cls.build_labels_ordinal_from_res(
            r,
            {
                "fixation_intervals": FIX,
                "saccade_intervals": SACCADE,
                "pursuit_intervals": SP,
            },
            NOISE,
        )

        coords, *_ = gt_vstk
        return (coords, pd.DataFrame({EVENT_LABEL: pred_np}))

    @classmethod
    def summarize_report_into_serie(cls, report):
        pass


class EntryPoint:
    paths: list[pathlib.Path]
    ReportForEachMethod: type
    


    @classmethod
    def load_ground_truth_file(cls, fn):
        pass

    @classmethod
    def main(cls, args):
        cutoff = args.cutoff
        report_name = args.report_name
        directory = args.directory

        # we sort it
        paths = sorted(cls.paths)

        if len(paths) >= cutoff:
            paths_cutoff = [paths[i] for i in range(0, len(paths), len(paths) // cutoff)]
        else:
            paths_cutoff = paths

        gt_dim_list = [
            res for p in paths_cutoff
            if (res := cls.load_ground_truth_file(p)) is not None
        ]

        report, report_summary_serie = cls.ReportForEachMethod.evaluate(gt_dim_list=gt_dim_list)

        if not directory.exists():
            directory.mkdir(exist_ok=True,
                            parents=True)

        report_path = directory / report_name

        report_summary_serie.to_json(
            report_path.with_suffix(".json"),
        )
        report_summary_serie.to_markdown(
            report_path.with_suffix(".md"),
        )
        report_summary_serie.plot.bar().figure.savefig(
            report_path.with_suffix(".png")
        )
