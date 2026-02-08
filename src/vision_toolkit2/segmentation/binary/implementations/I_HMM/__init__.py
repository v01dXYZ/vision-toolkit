# -*- coding: utf-8 -*-


import time

import vision_toolkit.segmentation.segmentation_algorithms.c_I_HMM.c_I_HMM as c_I_HMM

def process_impl(data_set, config):
    if config.verbose:
        print("Processing HMM Identification...")
        start_time = time.time()

    out = c_I_HMM.process_IHMM(data_set, config)

    if config.verbose:
        print("\n...HMM Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return out


def default_config_impl(config, vf_diag):
    i_l = 0.001 * vf_diag
    i_h = 10.0 * vf_diag
    i_v = 100 * vf_diag**2

    return Config(
        HMM_init_low_velocity = i_l,
        HMM_init_high_velocity = i_h,
        HMM_init_variance = i_v,
        HMM_nb_iters = 10,
    )
