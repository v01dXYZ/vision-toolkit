# -*- coding: utf-8 -*-
import time

from . import c_I_DeT


def process_impl(data_set, config):
    if config.verbose:
        print("Processing DeT Identification...")
        start_time = time.time()

    out = c_I_DeT.process_IDeT(data_set, config)

    if config.verbose:
        print("\n...DeT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return out

def default_config_impl(config, vf_diag):
    ## To accelerate computation, duration threshold should be equal to 5 time stamps
    du_t = 5 / sampling_frequency
    if config.distance_type == "euclidean":
        ## The default density threshold is thus defined from the sampling frequency
        de_t = vf_diag / sampling_frequency
        return Config(
            IDeT_duration_threshold =  du_t,
            IDeT_density_threshold = de_t,
        )
    elif config.distance_type == "angular":
        ## The default density threshold is thus defined from the sampling frequency
        de_t = 30 / sampling_frequency
        return Config(
            IDeT_duration_threshold = du_t,
            IDeT_density_threshold = de_t,
        )
