# -*- coding: utf-8 -*-
"""
Config Compatibility Module - Dictionary-based Prefix Tree Approach

This module provides bidirectional compatibility between the old flat config format
and the new nested config format using a dictionary-based prefix tree approach.

The key idea is that each nested path (represented as a dictionary key chain)
maps to the corresponding old flat config key, making the conversion logic
transparent and easy to maintain.
"""

from typing import Any, Dict, Optional
from .config import (
    Config,
    Segmentation,
    Identification,
    Smoothing,
    SerieMetadata,
    ScreenDimensions,
    SegmentationFilter,
    FilterRange,
    BaseSmoothing,
    MovingAverage,
    SpeedMovingAverage,
    Savgol,
    Fixation,
    Saccade,
    Pursuit,
)
from .config_old import Config as ConfigOld


# Prefix tree mapping: nested_dict_path -> flat_key
# This makes the conversion logic explicit and easy to maintain
# Format: {"segmentation": {"filter": {"status_threshold": "status_threshold"}}}
PREFIX_TREE = {
    # Serie Metadata
    "serie_metadata": {
        "sampling_frequency": "sampling_frequency",
        "nb_samples": "nb_samples",
    },
    # Screen Dimensions
    "screen_dimensions": {
        "x": "size_plan_x",
        "y": "size_plan_y",
        "diagonal": "screen_diagonal",
    },
    # Segmentation Filter
    "segmentation": {
        "method": "segmentation_method",
        "filter": {
            "fixation_duration": {
                "min": "min_fix_duration",
                "max": "max_fix_duration",
            },
            "pursuit_duration": {
                "min": "min_pursuit_duration",
                "max": "max_pursuit_duration",
            },
            "saccade_duration": {
                "min": "min_sac_duration",
                "max": "max_sac_duration",
            },
            "interval_size": {
                "min": "min_interval_size",
                "max": "max_interval_size",
            },
            "status_threshold": "status_threshold",
        },
    },
    # Smoothing - Savgol
    "smoothing": {
        "method": "smoothing",
        "savgol": {
            "window_length": "savgol_window_length",
            "polyorder": "savgol_polyorder",
        },
        # Smoothing - Moving Average
        "moving_average": {
            "window_length": "moving_average_window",
        },
        # Smoothing - Speed Moving Average
        "speed_moving_average": {
            "window_length": "moving_average_window",
        },
    },
    # Segmentation Algorithms - HMM
    "segmentation": {
        "hmm": {
            "init_high_velocity": "HMM_init_high_velocity",
            "init_low_velocity": "HMM_init_low_velocity",
            "init_variance": "HMM_init_variance",
            "nb_iters": "HMM_nb_iters",
        },
        # Segmentation Algorithms - I2MC
        "i2mc": {
            "merging_distance_threshold": "I2MC_merging_distance_threshold",
            "merging_duration_threshold": "I2MC_merging_duration_threshold",
            "moving_threshold": "I2MC_moving_threshold",
            "window_duration": "I2MC_window_duration",
        },
        # Segmentation Algorithms - IBDT
        "ibdt": {
            "duration_threshold": "IBDT_duration_threshold",
            "fixation_sd": "IBDT_fixation_sd",
            "fixation_threshold": "IBDT_fixation_threshold",
            "pursuit_threshold": "IBDT_pursuit_threshold",
            "saccade_sd": "IBDT_saccade_sd",
            "saccade_threshold": "IBDT_saccade_threshold",
        },
        # Segmentation Algorithms - I_DiT
        "i_dit": {
            "dispersion_threshold": "I_DiT_dispersion_threshold",
            "window_duration": "I_DiT_window_duration",
        },
        # Segmentation Algorithms - IFC
        "ifc": {
            "bcea_prob": "IFC_bcea_prob",
            "i2mc": "IFC_i2mc",
            "i2mc_moving_threshold": "IFC_i2mc_moving_threshold",
            "i2mc_window_duration": "IFC_i2mc_window_duration",
        },
        # Segmentation Algorithms - IKF
        "ikf": {
            "chi2_sigma": "IKF_chi2_sigma",
            "chi2_threshold": "IKF_chi2_threshold",
            "chi2_window": "IKF_chi2_window",
            "sigma_1": "IKF_sigma_1",
            "sigma_2": "IKF_sigma_2",
        },
        # Segmentation Algorithms - IMST
        "imst": {
            "distance_threshold": "IMST_distance_threshold",
            "min_cluster_size": "IMST_min_cluster_size",
            "step_samples": "IMST_step_samples",
            "window_duration": "IMST_window_duration",
        },
        # Segmentation Algorithms - IVT
        "ivt": {
            "velocity_threshold": "IVT_velocity_threshold",
        },
        # Segmentation Algorithms - IVDT
        "ivdt": {
            "dispersion_threshold": "IVDT_dispersion_threshold",
            "saccade_threshold": "IVDT_saccade_threshold",
            "window_duration": "IVDT_window_duration",
        },
        # Segmentation Algorithms - IVMP
        "ivmp": {
            "rayleigh_threshold": "IVMP_rayleigh_threshold",
            "saccade_threshold": "IVMP_saccade_threshold",
            "window_duration": "IVMP_window_duration",
        },
        # Segmentation Algorithms - IVVT
        "ivvt": {
            "pursuit_threshold": "IVVT_pursuit_threshold",
            "saccade_threshold": "IVVT_saccade_threshold",
        },
        # Segmentation Algorithms - IDeT
        "i_det": {
            "density_threshold": "IDeT_density_threshold",
            "duration_threshold": "IDeT_duration_threshold",
            "min_pts": "IDeT_min_pts",
        },
        # Segmentation Pursuit
        "pursuit": {
            "start_idx": "pursuit_start_idx",
        },
    },
    # Display config
    "display_AoI": "display_AoI",
    "display_AoI_path": "display_AoI_path",
    "display_scanpath": "display_scanpath",
    "display_scanpath_path": "display_scanpath_path",
    "display_segmentation_path": "display_segmentation_path",
    # Smoothing
    "smoothing": "smoothing",
    # Top-level config
    "curve_nb_points": "curve_nb_points",
    "distance_type": "distance_type",
    "distance_projection": "distance_projection",
    "mannan_distance_nb_random_scanpaths": "mannan_distance_nb_random_scanpaths",
    "moving_average_window": "moving_average_window",
    "subsmatch_ngram_length": "subsmatch_ngram_length",
    "task": "task",
    "verbose": "verbose",
    "display_segmentation": "display_segmentation",
    "display_results": "display_results",
    "nb_samples_pursuit": "nb_samples_pursuit",
}


# Inverted prefix tree for nested_to_flat conversion
# Format: {"status_threshold": "segmentation.filter.status_threshold"}
INVERTED_PREFIX_TREE = {}


# Build inverted prefix tree
def _build_inverted_prefix_tree(prefix_tree, path_so_far=""):
    """Recursively build the inverted prefix tree."""
    if isinstance(prefix_tree, dict):
        for key, value in prefix_tree.items():
            new_path = f"{path_so_far}.{key}" if path_so_far else key
            _build_inverted_prefix_tree(value, new_path)
    else:
        # This is a leaf node (flat key)
        INVERTED_PREFIX_TREE[prefix_tree] = path_so_far


_build_inverted_prefix_tree(PREFIX_TREE)


def flat_to_nested(flat_config: Dict[str, Any]) -> Config:
    """
    Convert a flat config dictionary to the new nested config structure.

    This function uses the PREFIX_TREE to map flat keys to nested paths,
    making the conversion logic explicit and easy to maintain.

    The flat_config is validated against config_old.Config.__dataclass_fields__
    to ensure all keys are valid old config fields.

    Args:
        flat_config: Dictionary with flat config keys (from config_old.Config)

    Returns:
        Config: New nested config object

    Raises:
        KeyError: If flat_config contains invalid keys not in config_old.Config
    """
    # Validate flat_config against config_old.Config fields
    valid_fields = set(ConfigOld.__dataclass_fields__.keys())
    invalid_fields = set(flat_config.keys()) - valid_fields

    if invalid_fields:
        raise KeyError(
            f"Invalid config keys: {invalid_fields}. "
            f"Valid keys are: {sorted(valid_fields)}"
        )
    # Create empty nested config structure
    config = Config(
        serie_metadata=SerieMetadata(
            sampling_frequency=flat_config.get("sampling_frequency", 1000),
            nb_samples=flat_config.get("nb_samples", 0),
            distance_type=flat_config.get("distance_type", "euclidean"),
        ),
        screen_dimensions=ScreenDimensions(
            x=flat_config.get("size_plan_x", 0),
            y=flat_config.get("size_plan_y", 0),
            diagonal=flat_config.get("screen_diagonal", 0),
        ),
        segmentation=None,
        smoothing=None,
        distance_type=flat_config.get("distance_type", "angular"),
        distance_projection=flat_config.get("distance_projection", None),
        verbose=flat_config.get("verbose", True),
        nb_samples_pursuit=flat_config.get("nb_samples_pursuit", 0),
    )

    # Create segmentation with filter, fixation, saccade, pursuit
    config.segmentation = Segmentation(
        flat_config.get("segmentation_method", "I_HMM"),
        filter=SegmentationFilter(
            fixation_duration=FilterRange[float](
                min=flat_config.get("min_fix_duration", 0.07),
                max=flat_config.get("max_fix_duration", 2.0),
            ),
            pursuit_duration=FilterRange[float](
                min=flat_config.get("min_pursuit_duration", 0.1),
                max=flat_config.get("max_pursuit_duration", 2.0),
            ),
            saccade_duration=FilterRange[float](
                min=flat_config.get("min_sac_duration", 0.015),
                max=flat_config.get("max_sac_duration", None),
            ),
            interval_size=FilterRange[int](
                min=flat_config.get("min_interval_size", 0),
                max=flat_config.get("max_interval_size", None),
            ),
            status_threshold=flat_config.get("status_threshold", 0.5),
        ),
        fixation=Fixation(
            weighted_average_velocity_means=flat_config.get("fixation_weighted_average_velocity_means", False),
            BCEA_probability=flat_config.get("fixation_BCEA_probability", 0.68),
        ),
        saccade=Saccade(
            absolute_horizontal_deviations=flat_config.get("saccade_absolute_horizontal_deviations", False),
            init_direction_duration_threshold=flat_config.get("saccade_init_direction_duration_threshold", 0.020),
            init_deviation_duration_threshold=flat_config.get("saccade_init_deviation_duration_threshold", 0.020),
            weighted_average_velocity_means=flat_config.get("saccade_weighted_average_velocity_means", False),
            weighted_average_acceleration_profiles=flat_config.get("saccade_weighted_average_acceleration_profiles", False),
            weighted_average_acceleration_means=flat_config.get("saccade_weighted_average_acceleration_means", False),
            weighted_average_deceleration_means=flat_config.get("saccade_weighted_average_deceleration_means", False),
        ),
        pursuit=Pursuit(
            end_idx=flat_config.get("pursuit_end_idx", None),
            onset_baseline_length=flat_config.get("pursuit_onset_baseline_length", None),
            onset_slope_length=flat_config.get("pursuit_onset_slope_length", None),
            onset_threshold=flat_config.get("pursuit_onset_threshold", None),
            start_idx=flat_config.get("pursuit_start_idx", 0),
        ),
    )

    # Create smoothing config
    smoothing_type = flat_config.get("smoothing", "savgol")

    if smoothing_type == "savgol":
        config.smoothing = Smoothing(
            Savgol(
                window_length=flat_config.get("savgol_window_length", 31),
                polyorder=flat_config.get("savgol_polyorder", 3),
            ),
        )
    elif smoothing_type == "moving_average":
        config.smoothing = Smoothing(
            smoothing="moving_average",
            moving_average=MovingAverage(
                window_length=flat_config.get("moving_average_window", 5),
            ),
        )
    elif smoothing_type == "speed_moving_average":
        config.smoothing = Smoothing(
            smoothing="speed_moving_average",
            speed_moving_average=SpeedMovingAverage(
                window_length=flat_config.get("moving_average_window", 5),
            ),
        )

    # Create algorithm-specific config
    _create_algorithm_config(config, flat_config)

    return config


def _create_algorithm_config(config: Config, flat_config: Dict[str, Any]) -> None:
    """
    Create algorithm-specific config based on flat config keys.

    This function detects which algorithm is being used based on the flat config
    keys and creates the appropriate nested config.
    """
    # IVT
    if any(k.startswith("IVT_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_VT import IVT

        config.segmentation.ivt = IVT(
            velocity_threshold=flat_config.get("IVT_velocity_threshold", 0),
        )

    # I2MC
    elif any(k.startswith("I2MC_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_2MC import I2MC

        config.segmentation.i2mc = I2MC(
            window_duration=flat_config.get("I2MC_window_duration", 0.04),
            moving_threshold=flat_config.get("I2MC_moving_threshold", 0),
            merging_duration_threshold=flat_config.get(
                "I2MC_merging_duration_threshold", 0
            ),
            merging_distance_threshold=flat_config.get(
                "I2MC_merging_distance_threshold", 0
            ),
        )

    # I_DiT
    elif any(k.startswith("I_DiT_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_DiT import I_DiT

        config.segmentation.i_dit = I_DiT(
            dispersion_threshold=flat_config.get("I_DiT_dispersion_threshold", 0),
            window_duration=flat_config.get("I_DiT_window_duration", 0.04),
        )

    # I_KF
    elif any(k.startswith("IKF_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_KF import IKF

        config.segmentation.ikf = IKF(
            kalman_gain=flat_config.get("IKF_kalman_gain", 0.5),
            velocity_threshold=flat_config.get("IKF_velocity_threshold", 0),
        )

    # I_MST
    elif any(k.startswith("IMST_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_MST import IMST

        config.segmentation.imst = IMST(
            distance_threshold=flat_config.get("IMST_distance_threshold", 0),
            min_cluster_size=flat_config.get("IMST_min_cluster_size", 0),
            step_samples=flat_config.get("IMST_step_samples", 0),
            window_duration=flat_config.get("IMST_window_duration", 0),
        )

    # I_VDT
    elif any(k.startswith("IVDT_") for k in flat_config.keys()):
        from .segmentation.ternary.implementations.I_VDT import IVDT

        config.segmentation.ivdt = IVDT(
            saccade_threshold=flat_config.get("IVDT_saccade_threshold", 0),
            dispersion_threshold=flat_config.get("IVDT_dispersion_threshold", 0),
            window_duration=flat_config.get("IVDT_window_duration", 0.04),
        )

    # I_VMP
    elif any(k.startswith("IVMP_") for k in flat_config.keys()):
        from .segmentation.ternary.implementations.I_VMP import IVMP

        config.segmentation.ivmp = IVMP(
            velocity_threshold=flat_config.get("IVMP_velocity_threshold", 0),
            dispersion_threshold=flat_config.get("IVMP_dispersion_threshold", 0),
            window_duration=flat_config.get("IVMP_window_duration", 0.04),
        )

    # I_VVT
    elif any(k.startswith("IVVT_") for k in flat_config.keys()):
        from .segmentation.ternary.implementations.I_VVT import IVVT

        config.segmentation.ivvt = IVVT(
            velocity_threshold=flat_config.get("IVVT_velocity_threshold", 0),
            dispersion_threshold=flat_config.get("IVVT_dispersion_threshold", 0),
            window_duration=flat_config.get("IVVT_window_duration", 0.04),
        )

    # I_BDT
    elif any(k.startswith("IBDT_") for k in flat_config.keys()):
        from .segmentation.ternary.implementations.I_BDT import IBDT

        config.segmentation.ibdt = IBDT(
            velocity_threshold=flat_config.get("IBDT_velocity_threshold", 0),
            dispersion_threshold=flat_config.get("IBDT_dispersion_threshold", 0),
            window_duration=flat_config.get("IBDT_window_duration", 0.04),
        )

    # I_DeT
    elif any(k.startswith("IDeT_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_DeT import IDeT

        config.segmentation.i_det = IDeT(
            density_threshold=flat_config.get("IDeT_density_threshold", 0),
            duration_threshold=flat_config.get("IDeT_duration_threshold", 0),
            min_pts=flat_config.get("IDeT_min_pts", 0),
        )

    # I_HMM
    elif any(k.startswith("HMM_") for k in flat_config.keys()):
        from .segmentation.binary.implementations.I_HMM import HMM

        config.segmentation.ihmm = HMM(
            init_low_velocity=flat_config.get("HMM_init_low_velocity", 0),
            init_high_velocity=flat_config.get("HMM_init_high_velocity", 0),
            init_variance=flat_config.get("HMM_init_variance", 0),
            nb_iters=flat_config.get("HMM_nb_iters", 10),
        )


def nested_to_flat(nested_config: Config) -> ConfigOld:
    """
    Convert a nested config object to a config_old.Config object (old format).

    This function uses the INVERTED_PREFIX_TREE to map nested paths back to
    flat keys, making the conversion logic explicit and easy to maintain.

    The resulting config_old.Config object contains all the fields from the nested config,
    ensuring full compatibility with the old config format.

    Args:
        nested_config: Config object with nested structure (new format)

    Returns:
        ConfigOld: config_old.Config object with flat structure (old format)
    """
    flat_config = {}

    # Traverse the nested config and extract values using the inverted prefix tree
    _extract_from_nested(flat_config, nested_config, [])

    # Create config_old.Config object from the flat dictionary
    return ConfigOld(**flat_config)


def _extract_from_nested(flat_config: Dict[str, Any], obj: Any, path: list) -> None:
    """
    Recursively extract values from nested config object.

    This helper function traverses the nested config structure and uses the
    INVERTED_PREFIX_TREE to map nested paths back to flat keys.
    """
    if not hasattr(obj, "__dict__"):
        return

    for key, value in obj.__dict__.items():
        new_path = path + [key]
        path_str = ".".join(new_path)

        # Check if this path has a corresponding flat key
        if path_str in INVERTED_PREFIX_TREE:
            flat_key = INVERTED_PREFIX_TREE[path_str]
            flat_config[flat_key] = value

        # Recursively traverse nested objects
        if hasattr(value, "__dict__"):
            _extract_from_nested(flat_config, value, new_path)


def update_config(old_config: Config, new_config_dict: Dict[str, Any]) -> Config:
    """
    Update an existing config with new values from a dictionary.

    This function intelligently merges old and new config values,
    preserving the nested structure while allowing selective updates.

    The new_config_dict is validated against config_old.Config.__dataclass_fields__
    to ensure all keys are valid old config fields.

    Args:
        old_config: Existing config object
        new_config_dict: Dictionary with new config values (flat format, from config_old.Config)

    Returns:
        Config: Updated config object

    Raises:
        KeyError: If new_config_dict contains invalid keys not in config_old.Config
    """
    # Validate new_config_dict against config_old.Config fields
    valid_fields = set(ConfigOld.__dataclass_fields__.keys())
    invalid_fields = set(new_config_dict.keys()) - valid_fields

    if invalid_fields:
        raise KeyError(
            f"Invalid config keys: {invalid_fields}. "
            f"Valid keys are: {sorted(valid_fields)}"
        )
    # If new_config_dict is already a Config object, return it directly
    if isinstance(new_config_dict, Config):
        return new_config_dict

    # Convert flat dict to nested if needed
    if isinstance(new_config_dict, dict):
        nested_new = flat_to_nested(new_config_dict)
    else:
        nested_new = new_config_dict

    # Merge configs using StackedConfig
    from .config import StackedConfig

    return StackedConfig([old_config, nested_new])


def get_flat_key(nested_path_str: str) -> Optional[str]:
    """
    Get the flat config key for a given nested path string.

    This is a utility function that makes the prefix tree mapping explicit.

    Args:
        nested_path_str: String representing the nested path (e.g., "segmentation.filter.status_threshold")

    Returns:
        Optional[str]: The corresponding flat config key, or None if not found
    """
    return INVERTED_PREFIX_TREE.get(nested_path_str)


def get_nested_path(flat_key: str) -> Optional[str]:
    """
    Get the nested path string for a given flat config key.

    This is a utility function that makes the inverted prefix tree mapping explicit.

    Args:
        flat_key: The flat config key (e.g., "status_threshold")

    Returns:
        Optional[str]: The corresponding nested path string, or None if not found
    """
    # Find the key in the PREFIX_TREE by searching recursively
    for key, value in _search_prefix_tree(PREFIX_TREE, flat_key):
        return key
    return None


def _search_prefix_tree(tree: Any, target: str, path_so_far: str = "") -> list:
    """Recursively search the prefix tree for a target value."""
    results = []

    if isinstance(tree, dict):
        for key, value in tree.items():
            new_path = f"{path_so_far}.{key}" if path_so_far else key
            results.extend(_search_prefix_tree(value, target, new_path))
    elif tree == target:
        results.append((path_so_far, tree))

    return results


__all__ = [
    "flat_to_nested",
    "nested_to_flat",
    "update_config",
    "get_flat_key",
    "get_nested_path",
]
