import argparse
import pathlib

import numpy as np
import pandas as pd

import vision_toolkit as v1
import vision_toolkit2 as v2
from vision_toolkit2 import AugmentedSerie, Config, Serie, Smoothing, StackedConfig
from vision_toolkit2.segmentation.binary.implementations import IMPLEMENTATIONS as BINARY_IMPLEMENTATIONS
from vision_toolkit2.segmentation.ternary.implementations import IMPLEMENTATIONS as TERNARY_IMPLEMENTATIONS
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
    "BINARY": {method: {} for method in BINARY_IMPLEMENTATIONS},
    "TERNARY": {method: {} for method in TERNARY_IMPLEMENTATIONS},
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

class V2Gateway:
    @classmethod
    def get_segmentation_cls(cls, nary):
        return v2.Segmentation

    @classmethod
    def run_segmentation_on_serie(cls, Segmentation, serie, config):
        config = StackedConfig([
            serie.config,
            Config(**config),
        ])

        segmentation = Segmentation(
            serie,
            config=config,
        )

        r = segmentation.process()

        return r

    @classmethod
    def create_serie(cls, gt_coords, dimensions, config):
        gt_s = Serie.from_df(
            gt_coords,
            size_plan_x=dimensions["width_mm"],
            size_plan_y=dimensions["height_mm"],
            sampling_frequency = config.get("sampling_frequency")
        )       

        config = {
            "smoothing": "savgol",
            "savgol_window_length": 31,
            "savgol_polyorder": 3,
            "distance_type": "euclidean",
            **config,
        }
        gt_smoothed = Smoothing.create_and_process(
            gt_s,
            Config(**config),
        )
        gt_augmented_smoothed = AugmentedSerie.augment_serie(gt_smoothed)

        return gt_augmented_smoothed


class V1Gateway:
    @classmethod
    def get_segmentation_cls(cls, nary):
        if nary == "BINARY":
            Segmentation = v1.BinarySegmentation
        elif nary == "TERNARY":
            Segmentation = v1.TernarySegmentation
        else:
            raise RuntimeError()

        return Segmentation

    @classmethod
    def run_segmentation_on_serie(cls, Segmentation, serie, config):
        segmentation = Segmentation(
            serie,
            **config,
        )

        return segmentation.segmentation_results

    @classmethod
    def create_serie(cls, gt_coords, dimensions, config):
        return gt_coords

class ReportForEachMethod:
    @classmethod
    def run_segmentation(
            cls,
            version,
            gt,
            gt_vstk,
            dimensions,
            nary,
            method,
            config={},
    ):
        gt_coords, gt_labels, gt_time = gt_vstk

        if gt_coords[GAZE_X].isna().sum() >= 1:
            return None

        config = {
            "segmentation_method": method,
            "distance_type": "euclidean",
            "display_segmentation": True,
            "size_plan_x": dimensions["width_mm"],
            "size_plan_y": dimensions["height_mm"],
            "smoothing": "savgol",
            "savgol_window_length": 31,
            "savgol_polyorder": 3,
            "verbose": False,
            **cls.CONFIG,
            **config,
        }

        VersionGateway = V2Gateway if version == 2 else V1Gateway

        serie = VersionGateway.create_serie(gt_coords, dimensions, config)
        Segmentation = VersionGateway.get_segmentation_cls(nary)
        results = VersionGateway.run_segmentation_on_serie(Segmentation, serie, config)

        return cls.build_predictions_from_results(results, gt, gt_vstk)

    def build_labels_ordinal_from_res(
            r,
            nb_samples,
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
            nb_samples,
            default_ordinal,
            dtype=dtype,
        )
        for k, val in key_ordinal.items():
            # hacky but works for now
            intervals = r.get(k) if isinstance(r, dict) else getattr(r, k, None)

            if intervals is None:
                continue

            for start, end in intervals:
                predictions[start : end + 1] = val

        return predictions
    

    @classmethod
    def evaluate(cls, version, gt_dim_list, config):
        # first we convert to VSTK ground truth
        gt_dim_list = [
            (gt, dim, gt_vstk)
            for gt, dim in gt_dim_list
            if (gt_vstk := cls.convert_to_vstk(gt, **dim)) is not None
        ]

        report = {}

        for nary, methods in METHODS_CONFIG.items():
            report[nary] = {}
            for method_name, method_config in methods.items():
                gt_list = []
                pred_list = []

                # TO TIRED ME: do not rename it config! (unless you want it to accumulate)
                updated_config = {
                    **method_config,
                    **config,
                }
                for gt, dimensions, gt_vstk in gt_dim_list:
                    pred = cls.run_segmentation(
                        version,
                        gt,
                        gt_vstk,
                        dimensions,
                        nary,
                        method_name,
                        updated_config,
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
    def build_predictions_from_results(cls, gt, gt_vstk):
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
    def build_predictions_from_results(cls, results, gt, gt_vstk):
        coords, *_ = gt_vstk

        pred_np = cls.build_labels_ordinal_from_res(
            results,
            coords.shape[0],
            {
                "fixation_intervals": FIX,
                "saccade_intervals": SACCADE,
                "pursuit_intervals": SP,
            },
            NOISE,
        )

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
    def main(cls,
             *,
             cutoff,
             report_name,
             directory,
             version,
             config,
    ):
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

        report, report_summary_serie = cls.ReportForEachMethod.evaluate(
            version,
            gt_dim_list=gt_dim_list,
            config=config,
        )

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
