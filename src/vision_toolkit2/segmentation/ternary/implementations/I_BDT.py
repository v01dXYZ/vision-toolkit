# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy.stats import norm

from vision_toolkit2.segmentation.utils import interval_merging
from vision_toolkit2.config import Config
from vision_toolkit2.config import IBDT, Segmentation

from ..ternary_segmentation_results import TernarySegmentationResults


def process_impl(s, config, segmentation_config, distance_type, verbose):
    """

    Parameters
    ----------
    s : Serie
        DESCRIPTION.
    config : Config
        DESCRIPTION.

    Returns
    -------
    TernarySegmentationResults
        DESCRIPTION.

    """
    if verbose:
        print("Processing BDT Identification...")
        start_time = time.time()

    n_s = int(s.min_config.nb_samples)
    s_f = float(s.min_config.sampling_frequency)

    d_t = max(1, int(np.ceil(float(segmentation_config.ibdt.duration_threshold) * s_f)))

    fix_t = float(segmentation_config.ibdt.fixation_threshold)
    sac_t = float(segmentation_config.ibdt.saccade_threshold)
    pur_t = float(segmentation_config.ibdt.pursuit_threshold)

    fix_sd = max(1e-9, float(segmentation_config.ibdt.fixation_sd))
    sac_sd = max(1e-9, float(segmentation_config.ibdt.saccade_sd))

    a_s = s.absolute_speed

    priors = {"fix": np.zeros(n_s), "sac": np.zeros(n_s), "pur": np.zeros(n_s)}
    likelihoods = {"fix": None, "sac": None, "pur": np.zeros(n_s)}
    posteriors = {"fix": None, "sac": None, "pur": None}

    for i in range(min(d_t, n_s)):
        likelihoods["pur"][i] = np.sum(a_s[: i + 1] > pur_t) / (i + 1)
        priors["pur"][i] = np.mean(likelihoods["pur"][: i + 1])
        priors["fix"][i] = priors["sac"][i] = (1.0 - priors["pur"][i]) / 2.0

    for i in range(d_t, n_s):
        likelihoods["pur"][i] = np.sum(a_s[i - d_t + 1 : i + 1] > pur_t) / d_t
        priors["pur"][i] = np.mean(likelihoods["pur"][i - d_t + 1 : i + 1])
        priors["fix"][i] = priors["sac"][i] = (1.0 - priors["pur"][i]) / 2.0

    lk_f = a_s.copy()
    lk_f[lk_f < fix_t] = fix_t
    likelihoods["fix"] = norm.pdf(lk_f, loc=fix_t, scale=fix_sd)

    lk_s = a_s.copy()
    lk_s[lk_s > sac_t] = sac_t
    likelihoods["sac"] = norm.pdf(lk_s, loc=sac_t, scale=sac_sd)

    eps = 1e-6
    likelihoods["pur"] = np.clip(likelihoods["pur"], eps, 1 - eps)

    for ev in ("fix", "sac", "pur"):
        posteriors[ev] = priors[ev] * likelihoods[ev]

    a_m = np.argmax(
        np.vstack((posteriors["fix"], posteriors["sac"], posteriors["pur"])),
        axis=0,
    )

    is_fixation = a_m == 0
    is_saccade = a_m == 1
    is_pursuit = a_m == 2

    fixation_intervals = interval_merging(np.where(is_fixation)[0])
    saccade_intervals = interval_merging(np.where(is_saccade)[0])
    pursuit_intervals = interval_merging(np.where(is_pursuit)[0])

    if verbose:
        print("\n...BDT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return TernarySegmentationResults(
        is_fixation=is_fixation,
        fixation_intervals=fixation_intervals,
        is_saccade=is_saccade,
        saccade_intervals=saccade_intervals,
        is_pursuit=is_pursuit,
        pursuit_intervals=pursuit_intervals,
        input=s,
        config=config,
    )


def default_config_impl(config, vf_diag):
    if config.distance_type == "euclidean":
        fix_t = 0.1 * vf_diag
        pur_t = 0.15 * vf_diag
        sac_t = 1.0 * vf_diag
        ibdt_config = IBDT(
            duration_threshold=0.050,
            fixation_threshold=fix_t,
            saccade_threshold=sac_t,
            pursuit_threshold=pur_t,
            fixation_sd=0.01,
            saccade_sd=0.01,
        )
        return Config(
            segmentation=Segmentation(ibdt_config),
        )
    elif config.distance_type == "angular":
        ibdt_config = IBDT(
            duration_threshold=0.050,
            fixation_threshold=5,
            saccade_threshold=50,
            pursuit_threshold=8,
            fixation_sd=0.01,
            saccade_sd=0.01,
        )
