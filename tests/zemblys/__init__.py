import pathlib

import numpy as np
import pandas as pd
import sklearn as sk

import vision_toolkit_test as vt


# Here, screen dimensions do not change
EYE_DISTANCE = 565.0
SCREEN_WIDTH = 533.0
SCREEN_HEIGHT = 301.0


# blinking too many times
FAULTY_PARTICIPANT_ID = 5

class ZemblysReportForEachMethod(vt.VSTKReportForEachMethod):
    SEGMENTATION_KWARGS = {
        "sampling_frequency": 1000,
    }

    @classmethod
    def evaluate_predictions(cls, gt, pred):
        concat_gt_labels = pd.concat([e for _, e in gt])[vt.EVENT_LABEL]
        concat_pred_labels = pd.concat([e for _, e in pred])[vt.EVENT_LABEL]

        return sk.metrics.cohen_kappa_score(
            concat_gt_labels,
            concat_pred_labels,
        )

    @classmethod
    def summarize_report_into_serie(cls, report):
        return pd.Series({
            method_name: r
            for method_name, r in {**report["BINARY"], **report["TERNARY"]}.items()
        })

class EntryPoint(vt.EntryPoint):
    paths = [
        pathlib.Path(__file__).parent / "data" / f"lookAtPoint_EL_S{i}.npy"
        for i in range(1, 6) if i != FAULTY_PARTICIPANT_ID
    ]
    ReportForEachMethod = ZemblysReportForEachMethod

    @classmethod
    def load_ground_truth_file(cls, fn):
        seq = np.load(fn)

        theta_x = np.array([
            seq[i][1] for i in range(len(seq))
            ])

        theta_y = np.array([
            seq[i][2] for i in range(len(seq))
            ])

        e_ = np.array([
            seq[i][4] for i in range(len(seq))
            ])

        labels_df = pd.DataFrame(e_, columns = ['event_label'])

        # Merge fixations and post-saccadic oscillations
        labels_df = labels_df.replace(3, 1)

        # Not interested in SP
        labels_df = labels_df.replace(4, 0)

        # Merge blinks and undefined
        labels_df = labels_df.replace(5, 0)

        # Save as cartesian coordinates
        x = np.tan(theta_x * (np.pi/180)) * EYE_DISTANCE + SCREEN_WIDTH/2
        y = np.tan(theta_y * (np.pi/180)) * EYE_DISTANCE + SCREEN_HEIGHT/2

        gaze_df = pd.DataFrame(np.array([x, y]).T, 
                            columns = [vt.GAZE_X, vt.GAZE_Y])

        # Undefined and blinks are replaced by np.nan and interpolated
        gaze_df.loc[labels_df['event_label'] == 0] = np.nan 
        gaze_df = gaze_df.interpolate()

        noise = np.where(labels_df['event_label'] == 0)[0]
        ratio = len(noise)/len(labels_df)

        if ratio <= 0.10:
            return (
                (gaze_df, labels_df),
                {
                    "width_px": None,
                    "height_px": None,
                    "width_mm": None,
                    "height_mm": None,
                },
            )

        return 
