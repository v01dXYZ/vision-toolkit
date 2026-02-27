import dataclasses
from typing import TypeVar, Generic, Union


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


@dataclasses.dataclass
class HMM:

    init_low_velocity: float
    init_high_velocity: float
    init_variance: float
    nb_iters: int


@dataclasses.dataclass
class I2MC:

    merging_distance_threshold: float  # [0, 1]?
    merging_duration_threshold: float  # [0, 1]?
    moving_threshold: float  # [0, 1]?
    window_duration: float


@dataclasses.dataclass
class CDBA:

    initial_random_state: int
    initialization_length: str  # min | max
    maximum_iterations: int


@dataclasses.dataclass
class IAP:

    centers: str  # mean | raw_IAP


@dataclasses.dataclass
class IDP:

    centers: str  # mean | raw_IDP
    gaussian_kernel_sd: float


@dataclasses.dataclass
class IDT:

    density_threshold: float
    min_samples: int


@dataclasses.dataclass
class IKM:

    cluster_number: str  # search
    min_clusters: int
    max_clusters: int


@dataclasses.dataclass
class IMS:

    bandwidth: float


@dataclasses.dataclass
class IDeT:

    density_threshold: float
    duration_threshold: float
    min_pts: int


@dataclasses.dataclass
class String_Distance:

    deletion_cost: float
    insertion_cost: float
    normalization: str  # min | max


@dataclasses.dataclass
class Levenshtein_Distance(String_Distance):

    substitution_cost: float


@dataclasses.dataclass
class Generalized_Edit_Distance(String_Distance):

    pass


@dataclasses.dataclass
class Needleman_Wunsch_Distance:

    concordance_bonus: float
    gap_cost: 0.25
    normalization: str  # min | max


@dataclasses.dataclass
class Smith_Waterman:

    base_deletion_cost: float
    iterative_deletion_cost: float
    similarity_threshold: float
    similarity_weight: float


@dataclasses.dataclass
class Temporal_Binning:

    temporal_binning: bool
    temporal_binning_length: float


Identification = tagged_union_disjoint_types(
    "Identification",
    "identification_method",
    {
        "iap": IAP,
        "idp": IDP,
        "idt": IDT,
        "ikm": IKM,
        "ims": IMS,
    }
)


@dataclasses.dataclass
class AoI:

    # Distance classes as attributes
    levenshtein_distance: Levenshtein_Distance
    generalized_edit_distance: Generalized_Edit_Distance
    needleman_wunsch_distance: Needleman_Wunsch_Distance
    smith_waterman: Smith_Waterman

    # Identification methods as attribute
    identification: Identification

    # Other classes as attributes
    cdba: CDBA
    temporal_binning: Temporal_Binning

    SPAM_support: float
    coordinates: None  # array[2, 2]

    longest_common_subsequence_normalization: str  # min | max

    predefined_all: bool
    predefined_coordinates: None  # list[array[2, 2]]

    trend_analysis_tolerance_level: None


@dataclasses.dataclass
class IBDT:

    duration_threshold: float
    fixation_sd: float
    fixation_threshold: float
    pursuit_threshold: float
    saccade_sd: float
    saccade_threshold: float


@dataclasses.dataclass
class I_DiT:

    dispersion_threshold: float
    window_duration: float


@dataclasses.dataclass
class IFC:

    bcea_prob: float
    classifier: str  # to specify more closely
    i2mc: bool
    i2mc_moving_threshold: float
    i2mc_window_duration: float


@dataclasses.dataclass
class IKF:

    chi2_threshold: float
    chi2_window: float
    chi2_sigma: float
    sigma_1: float
    sigma_2: float


@dataclasses.dataclass
class IMST:

    distance_threshold: float
    window_duration: float
    step_samples: int | None
    min_cluster_size: int | None


@dataclasses.dataclass
class IVDT:

    dispersion_threshold: float
    saccade_threshold: float
    window_duration: float


@dataclasses.dataclass
class IVMP:

    # Paper link: Part2 l.345
    rayleigh_threshold: float
    saccade_threshold: float
    window_duration: float


@dataclasses.dataclass
class IVT:

    # Paper link: Part2 l.189
    velocity_threshold: float


@dataclasses.dataclass
class IVVT:

    pursuit_threshold: float
    saccade_threshold: float


@dataclasses.dataclass
class TDE_distance:

    method: None
    scaling: None
    subsequence_length: None


@dataclasses.dataclass
class Display:

    _display: None  # means display itself

    AoI: bool
    AoI_path: None | str
    path: None | str
    results: bool
    scanpath: bool
    scanpath_path: None | str
    segmentation: bool
    segmentation_path: None | str


@dataclasses.dataclass
class Multimatch_Simplification:

    amplitude_threshold: float
    angular_threshold: float
    duration_threshold: float
    iterations: int


@dataclasses.dataclass
class Persistence:

    display: bool
    landscape_order: int


@dataclasses.dataclass
class Scanmatch_Score:

    concordance_bonus: None
    gap_cost: None
    substitution_threshold: None


@dataclasses.dataclass
class Scanpath:

    # Distance classes as attributes
    levenshtein_distance: Levenshtein_Distance
    generalized_edit_distance: Generalized_Edit_Distance
    needleman_wunsch_distance: Needleman_Wunsch_Distance

    # Temporal binning as attribute
    temporal_binning: Temporal_Binning

    CRQA_distance_threshold: float
    CRQA_minimum_length: float
    RQA_distance_threshold: float
    RQA_minimum_length: float

    spatial_binning_nb_pixels_x: int
    spatial_binning_nb_pixels_y: int


# class Smoothing:

#     smoothing: str  # moving_average | speed_moving_average | savgol

Smoothing = tagged_union_disjoint_types(
    "Smoothing",
    "smoothing",
    {
        "savgol": Savgol,
        "moving_average": MovingAverage,
        "speed_moving_average": SpeedMovingAverage,
    }
)


@dataclasses.dataclass
class BaseSmoothing:
    window_length: int


@dataclasses.dataclass
class MovingAverage(BaseSmoothing):
    pass

@dataclasses.dataclass
class SpeedMovingAverage(BaseSmoothing):
    pass

@dataclasses.dataclass
class Savgol(BaseSmoothing):
    polyorder: int

@dataclasses.dataclass
class Fixation:

    weighted_average_velocity_means: bool
    BCEA_probability: float


@dataclasses.dataclass
class Saccade:

    absolute_horizontal_deviations: bool
    init_direction_duration_threshold: float
    init_deviation_duration_threshold: float
    weighted_average_velocity_means: bool
    weighted_average_acceleration_profiles: bool
    weighted_average_acceleration_means: bool
    weighted_average_deceleration_means: bool


@dataclasses.dataclass
class Pursuit:

    end_idx: None | int

    onset_baseline_length: None
    onset_slope_length: None
    onset_threshold: None

    start_idx: int


@dataclasses.dataclass
class ScreenDimensions:
    x: float
    y: float
    diagonal: float


T = TypeVar('T', bound=Union[float, int])


@dataclasses.dataclass
class FilterRange(Generic[T]):
    min: T | None
    max: T | None


@dataclasses.dataclass
class SegmentationFilter:
    fixation_duration: FilterRange[float]
    pursuit_duration: FilterRange[float]
    saccade_duration: FilterRange[float]
    interval_size: FilterRange[int]
    status_threshold: float | None


@dataclasses.dataclass
class SerieMetadata:
    sampling_frequency: int
    nb_samples: int


@prefix()
class Common:
    curve_nb_points: int

    distance_projection: int | None
    distance_type: str  # euclidean | angular

    mannan_distance_nb_random_scanpaths: int

    # from reference_image_mapper, so ??
    #    model: None
    moving_average_window: int

    # commented out in the code
    #    normalized_scanpath_saliency_delta: None

    # from reference_image_mapper, so ??
    #    processing: None

    serie_metadata: SerieMetadata

    segmentation: Segmentation

    smoothing: Smoothing

    fixation: Fixation
    saccade: Saccade
    pursuit: Pursuit

    screen_dimensions: ScreenDimensions

    subsmatch_ngram_length: int
    task: str  # binary | ternary

    verbose: bool
    nb_samples_pursuit: int

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


def tagged_union_disjoint_types(class_name: str, tag_name: str, classes: dict[str, type], **extra_fields) -> type:
    """
    Creates a tagged union dataclass with __disjoint_types__ attribute and extra fields.

    Args:
        class_name: Name of the generated dataclass
        tag_name: Name of the tag attribute
        classes: Dictionary mapping tag values to dataclass types
        **extra_fields: Additional fields to add to the dataclass (field_name: type)

    Returns:
        A dataclass with __disjoint_types__ attribute, extra fields, and smart initialization
    """
    # Convert classes dict to list of (tag_value, type) tuples
    disjoint_types = list(classes.items())

    def __init__(self, variant, **kwargs):
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

        # Set extra fields
        for field_name, value in kwargs.items():
            setattr(self, field_name, value)

    # Build fields list: tag field + variant fields + extra fields
    fields = [(tag_name, str)] + [cls for _, cls in disjoint_types]
    
    # Add extra fields
    for field_name, field_type in extra_fields.items():
        fields.append((field_name, field_type))

    # Create the dataclass using make_dataclass
    cls = dataclasses.make_dataclass(
        class_name,
        fields,
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
        "I_DiT": I_DiT,
        "IFC": IFC,
        "IKF": IKF,
        "IMST": IMST,
        "IVDT": IVDT,
        "IVMP": IVMP,
        "IVT": IVT,
        "IVVT": IVVT,
        "IDeT": IDeT,
    },
    filter=SegmentationFilter  # Add filter as an extra field
)
