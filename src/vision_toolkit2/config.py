import dataclasses


def asdict(dc_instance):
    d = {}
    for k in Common.__dataclass_fields__:
        v = getattr(dc_instance, k)

        if v is None:
            continue

        d[k] = v

    return d


def prefix(prefix="", class_name=False, lower=True):
    def f(cls):
        return dataclasses.make_dataclass(
            cls.__name__,
            [
                (
                    (
                        ((cls.__name__.lower() if lower else cls.__name__) + "_")
                        if class_name and not a.startswith("_")
                        else prefix
                    )
                    + a.lstrip("_"),
                    t,
                    dataclasses.field(default=None),
                )
                for c in (cls, *cls.__bases__)
                if c is not object
                for a, t in c.__annotations__.items()
            ],
            #            bases=cls.__bases__,
        )

    return f


class HMM:
    __config_merge__ = prefix(class_name=True, lower=False)

    init_low_velocity: float
    init_high_velocity: float
    init_variance: float
    nb_iters: int


class I2MC:
    __config_merge__ = prefix(class_name=True, lower=False)

    merging_distance_threshold: float  # [0, 1]?
    merging_duration_threshold: float  # [0, 1]?
    moving_threshold: float  # [0, 1]?
    window_duration: float


class CDBA:
    __config_merge__ = prefix(class_name=True, lower=False)

    initial_random_state: int
    initialization_length: str  # min | max
    maximum_iterations: int


class IAP:
    __config_merge__ = prefix(class_name=True, lower=False)

    centers: str  # mean | raw_IAP


class IDP:
    __config_merge__ = prefix(class_name=True, lower=False)

    centers: str  # mean | raw_IDP
    gaussian_kernel_sd: float


class IDT:
    __config_merge__ = prefix(class_name=True, lower=False)

    density_threshold: float
    min_samples: int


class IKM:
    __config_merge__ = prefix(class_name=True, lower=False)

    cluster_number: str  # search
    min_clusters: int
    max_clusters: int


class IMS:
    __config_merge__ = prefix(class_name=True, lower=False)

    bandwidth: float


class IDeT:
    __config_merge__ = prefix(class_name=True, lower=False)

    density_threshold: float
    duration_threshold: float
    min_pts: int


class String_Distance:
    __config_merge__ = prefix(class_name=False)

    deletion_cost: float
    insertion_cost: float
    normalization: str  # min | max


class Levenshtein_Distance(String_Distance):
    __config_merge__ = prefix(class_name=True)

    substitution_cost: float


class Generalized_Edit_Distance(String_Distance):
    __config_merge__ = prefix(class_name=True)

    pass


class Needleman_Wunsch_Distance:
    __config_merge__ = prefix(class_name=True)

    concordance_bonus: float
    gap_cost: 0.25
    normalization: str  # min | max


class Smith_Waterman:
    __config_merge__ = prefix(class_name=True)

    base_deletion_cost: float
    iterative_deletion_cost: float
    similarity_threshold: float
    similarity_weight: float


class Temporal_Binning:
    __config_merge__ = prefix()

    temporal_binning: bool
    temporal_binning_length: float


class AoI(
    CDBA,
    IAP,
    IDP,
    IDT,
    IKM,
    IMS,
    Levenshtein_Distance,
    Generalized_Edit_Distance,
    Needleman_Wunsch_Distance,
    Smith_Waterman,
    Temporal_Binning,
):
    __config_merge__ = prefix(class_name=True, lower=False)

    SPAM_support: float
    coordinates: None  # array[2, 2]

    identification_method: None
    longest_common_subsequence_normalization: str  # min | max

    predefined_all: bool
    predefined_coordinates: None  # list[array[2, 2]]

    trend_analysis_tolerance_level: None


class IBDT:
    __config_merge__ = prefix(class_name=True, lower=False)

    duration_threshold: float
    fixation_sd: float
    fixation_threshold: float
    pursuit_threshold: float
    saccade_sd: float
    saccade_threshold: float


class ICNN:
    __config_merge__ = prefix(class_name=True, lower=False)

    batch_size: int
    learning_rate: float
    num_epochs: int
    temporal_window_size: int


class I_DiT:
    __config_merge__ = prefix(class_name=True, lower=False)

    dispersion_threshold: float
    window_duration: float


class IFC:
    __config_merge__ = prefix(class_name=True, lower=False)

    bcea_prob: float
    classifier: str  # to specify more closely
    i2mc: bool
    i2mc_moving_threshold: float
    i2mc_window_duration: float


class IKF:
    __config_merge__ = prefix(class_name=True, lower=False)

    chi2_threshold: float
    chi2_window: float
    chi2_sigma: float
    sigma_1: float
    sigma_2: float


class IMST:
    __config_merge__ = prefix(class_name=True, lower=False)

    distance_threshold: float
    window_duration: float
    step_samples: int | None
    min_cluster_size: int | None


class IVDT:
    __config_merge__ = prefix(class_name=True, lower=False)

    dispersion_threshold: float
    saccade_threshold: float
    window_duration: float


class IVMP:
    __config_merge__ = prefix(class_name=True, lower=False)

    # Paper link: Part2 l.345
    rayleigh_threshold: float
    saccade_threshold: float
    window_duration: float


class IVT:
    __config_merge__ = prefix(class_name=True, lower=False)

    # Paper link: Part2 l.189
    velocity_threshold: float


class IVVT:
    __config_merge__ = prefix(class_name=True, lower=False)

    pursuit_threshold: float
    saccade_threshold: float


class TDE_distance:
    __config_merge__ = prefix(class_name=True, lower=False)

    method: None
    scaling: None
    subsequence_length: None


class Display:
    __config_merge__ = prefix(class_name=True)

    _display: None  # means display itself

    AoI: bool
    AoI_path: None | str
    path: None | str
    results: bool
    scanpath: bool
    scanpath_path: None | str
    segmentation: bool
    segmentation_path: None | str


class Multimatch_Simplification:
    __config_merge__ = prefix(class_name=True)

    amplitude_threshold: float
    angular_threshold: float
    duration_threshold: float
    iterations: int


class Persistence:
    __config_merge__ = prefix(class_name=True)

    display: bool
    landscape_order: int


class Scanmatch_Score:
    __config_merge__ = prefix(class_name=True)

    concordance_bonus: None
    gap_cost: None
    substitution_threshold: None


class Scanpath(
    Levenshtein_Distance,
    Needleman_Wunsch_Distance,
    Generalized_Edit_Distance,
    Temporal_Binning,
):
    __config_merge__ = prefix(class_name=True)

    CRQA_distance_threshold: float
    CRQA_minimum_length: float
    RQA_distance_threshold: float
    RQA_minimum_length: float

    spatial_binning_nb_pixels_x: int
    spatial_binning_nb_pixels_y: int


# class Smoothing:
#     __config_merge__ = prefix()

#     smoothing: str  # moving_average | speed_moving_average | savgol

Smoothing = tagged_union_disjoint_types(
    "Smoothing",
    "smoothing",
    {
        "savgol": Savgol,
    }
)


class Savgol:
    __config_merge__ = prefix(class_name=True)

    polyorder: int
    window_length: int


class Fixation:
    __config_merge__ = prefix(class_name=True)

    weighted_average_velocity_means: bool
    BCEA_probability: float


class Saccade:
    __config_merge__ = prefix(class_name=True)

    absolute_horizontal_deviations: bool
    init_direction_duration_threshold: float
    init_deviation_duration_threshold: float
    weighted_average_velocity_means: bool
    weighted_average_acceleration_profiles: bool
    weighted_average_acceleration_means: bool
    weighted_average_deceleration_means: bool


class Pursuit:
    __config_merge__ = prefix(class_name=True)

    end_idx: None | int

    onset_baseline_length: None
    onset_slope_length: None
    onset_threshold: None

    start_idx: int


@prefix()
class Common(
    AoI,
    TDE_distance,
    Display,
    Multimatch_Simplification,
    Persistence,
    Scanmatch_Score,
    Scanpath,
    Fixation,
    Saccade,
    Pursuit,
):
    curve_nb_points: int

    distance_projection: int | None
    distance_type: str  # euclidean | angular

    mannan_distance_nb_random_scanpaths: int

    max_fix_duration: float
    min_fix_duration: float
    max_pursuit_duration: float
    min_pursuit_duration: float

    min_int_size: int
    min_sac_duration: float

    # from reference_image_mapper, so ??
    #    model: None

    moving_average_window: int

    nb_samples: int
    nb_samples_pursuit: int

    # commented out in the code
    #    normalized_scanpath_saliency_delta: None

    # from reference_image_mapper, so ??
    #    processing: None

    sampling_frequency: int

    screen_diagonal: float

    # do it later
    segmentation_method: None

    segmentation: Segmentation

    smoothing: Smoothing

    size_plan_x: float
    size_plan_y: float

    status_threshold: float  # (between 0 and 1?)
    subsmatch_ngram_length: int
    task: str  # binary | ternary

    verbose: bool


class ConfigMixin:
    def update(self, **kwargs):
        # UGLY BUT WORKS
        return type(self)(
            **(asdict(self) | kwargs),
        )

    def merge(self, other):
        return type(self)(**(asdict(self) | asdict(other)))

    def print(self):
        print("Config used")
        print("-" * 15)

        for k, v in asdict(self).items():
            print(f"# {k}: {v}")

    def __repr__(self):
        prologue = f"{type(self).__name__}("

        s_v = []
        for k in Common.__dataclass_fields__:
            v = getattr(self, k)

            if v is None:
                continue

            s_v.append(f"{k}={v}")

        return f"{prologue}{','.join(s_v)})"


class Config(ConfigMixin, Common):
    pass


class StackedConfig(ConfigMixin):
    # Quickly implemented class for stacking config which is useful to
    # build bit by bit a config
    def __init__(self, config=None):
        if config is None:
            self.stack = []
            return

        self.stack = config if isinstance(config, list) else [config]

    def __iadd__(self, other: Config):
        self.stack.append(other)

        return self

    def __getattr__(self, attr_name: str):
        if attr_name not in Config.__dataclass_fields__:
            raise AttributeError

        for c in reversed(self.stack):
            attr = getattr(c, attr_name, None)

            if attr is not None:
                return attr

        return None


def tagged_union_disjoint_types(class_name: str, tag_name: str, classes: dict[str, type]) -> type:
    """
    Creates a tagged union dataclass with __disjoint_types__ attribute.

    Args:
        class_name: Name of the generated dataclass
        tag_name: Name of the tag attribute
        classes: Dictionary mapping tag values to dataclass types

    Returns:
        A dataclass with __disjoint_types__ attribute and smart initialization
    """
    # Convert classes dict to list of (tag_value, type) tuples
    disjoint_types = list(classes.items())

    def __init__(self, variant):
        # Determine the tag based on the variant's type
        variant_class = type(variant)

        # Build reverse mapping from class to tag using self.__disjoint_types__
        class_to_tag = {
            cls: tag
            for tag, cls in self.__disjoint_types__
        }

        if variant_class not in class_to_tag:
            raise TypeError(
                f"Unsupported variant type: {variant_class}. "
                f"Supported types: {[cls.__name__ for cls in class_to_tag.keys()]}"
            )

        tag_value = class_to_tag[variant_class]

        # Set the tag using self.__tag__
        setattr(self, self.__tag__, tag_value)

        # Set the variant in the correct attribute
        setattr(self, tag_value, variant)

    # Create the dataclass using make_dataclass
    cls = dataclasses.make_dataclass(
        class_name,
        [(tag_name, str)] + [cls for _, cls in disjoint_types],
        namespace={
            '__tag__': tag_name,
            '__disjoint_types__': disjoint_types,
            '__init__': __init__
        }
    )

    return cls

Segmentation = tagged_union_disjoint_types(
    "Segmentation",
    "segmentation_method",
    {
        "HMM": HMM,
        "I2MC": I2MC,
        "IBDT": IBDT,
        "ICNN": ICNN,
        "I_DiT": I_DiT,
        "IFC": IFC,
        "IKF": IKF,
        "IMST": IMST,
        "IVDT": IVDT,
        "IVMP": IVMP,
        "IVT": IVT,
        "IVVT": IVVT,
        "IDeT": IDeT,
    }
)
