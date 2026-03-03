import dataclasses
from typing import TypeVar, Generic, Union, Optional


@dataclasses.dataclass(frozen=True)
class UnionTag:
    attr: str
    value: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "attr", self.attr.lower())
        if self.value is None:
            object.__setattr__(self, "value", self.attr)

    @classmethod
    def ensure(cls, value):
        if isinstance(value, cls):
            return value
        else:
            return cls(value)


def all_nullable_dataclass(cls):
    """
    Modify a dataclass to make all fields Optional with default=None.

    This function modifies the original dataclass in-place by replacing its
    __init__ method with one that makes all fields optional.

    Args:
        cls: The dataclass to modify

    Returns:
        The modified dataclass (same as input)

    Example:
        @dataclass
        class Original:
            name: str
            age: int

        all_nullable_dataclass(Original)
        obj = Original()  # All fields are None now
    """
    cls = dataclasses.dataclass(cls, kw_only=True)

    # Get the dataclass fields using dataclasses.fields()
    fields = dataclasses.fields(cls)

    # Create new field definitions with Optional types and default=None
    new_fields = []
    for field in fields:
        # Get the original type
        original_type = field.type

        # Make the type Optional
        optional_type = Optional[original_type]

        # Create field with default=None
        new_field = dataclasses.field(default=None)

        new_fields.append((field.name, optional_type, new_field))

    # Create a new dataclass with nullable fields
    nullable_cls = dataclasses.make_dataclass(
        cls.__name__,
        fields=new_fields,
    )

    # Copy the __init__ method from nullable_cls to cls
    cls.__init__ = nullable_cls.__init__
    cls.__dataclass_fields__ = nullable_cls.__dataclass_fields__

    return cls


def tagged_union_disjoint_types(
    class_name: str, tag_name: str, classes: dict[str, type], **extra_fields
) -> type:
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
    disjoint_types = [(UnionTag.ensure(tag), cls) for tag, cls in classes.items()]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}<{getattr(self, self.__tag__)}="
            f"{getattr(self, self.__tag_attr__, None)}"
        )

    # Build fields list: tag field + variant fields + extra fields
    no_init = lambda: dataclasses.field(init=False)
    fields = [(tag_name, str, no_init())] + [
        (tag.attr, cls, no_init()) for tag, cls in disjoint_types
    ]

    nullable_kw_only = lambda: dataclasses.field(kw_only=True, default=None)
    # Add extra fields
    for field_name, field_type in extra_fields.items():
        fields.append((field_name, field_type, nullable_kw_only()))

    # Create the dataclass using make_dataclass
    cls = dataclasses.make_dataclass(
        class_name,
        fields,
        namespace={
            "__tag__": tag_name,
            "__disjoint_types__": disjoint_types,
            "__types__": tuple(cls for _, cls in disjoint_types),
            "__repr__": __repr__,
        },
    )

    def __init__(self, variant=None, **kwargs):
        # Allow None variant for flexible initialization
        if variant is None:
            setattr(self, self.__tag__, None)
            self.__tag_attr__ = None
            self.__original_init__(**kwargs)
            return

        if isinstance(variant, str):
            value_to_tag = {tag.value: tag for tag, _ in disjoint_types}

            if variant not in value_to_tag:
                raise ValueError(f"Invalid tag: {variant}. Valid tags: {value_to_tag.keys()}")

            tag = value_to_tag[variant]
        elif isinstance(variant, self.__types__):
            # Determine the tag based on the variant's type
            variant_class = type(variant)

            # Build reverse mapping from class to tag using self.__disjoint_types__
            class_to_tag = {cls: tag for tag, cls in self.__disjoint_types__}

            if variant_class not in class_to_tag:
                raise TypeError(
                    f"Unsupported variant type: {variant_class}. "
                    f"Supported types: {[cls.__name__ for cls in class_to_tag.keys()]}"
                )

            tag = class_to_tag[variant_class]
            setattr(self, tag.attr, variant)
        else:
            ValueError()

        setattr(self, self.__tag__, tag.value)
        self.__tag_attr__ = tag.attr

        self.__original_init__(**kwargs)

    cls.__original_init__ = cls.__init__
    cls.__init__ = __init__

    return cls


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


@all_nullable_dataclass
class HMM:
    init_low_velocity: float
    init_high_velocity: float
    init_variance: float
    nb_iters: int


@all_nullable_dataclass
class I2MC:
    merging_distance_threshold: float  # [0, 1]?
    merging_duration_threshold: float  # [0, 1]?
    moving_threshold: float  # [0, 1]?
    window_duration: float


@all_nullable_dataclass
class CDBA:
    initial_random_state: int
    initialization_length: str  # min | max
    maximum_iterations: int


@all_nullable_dataclass
class IAP:
    centers: str  # mean | raw_IAP


@all_nullable_dataclass
class IDP:
    centers: str  # mean | raw_IDP
    gaussian_kernel_sd: float


@all_nullable_dataclass
class IDT:
    density_threshold: float
    min_samples: int


@all_nullable_dataclass
class IKM:
    cluster_number: str  # search
    min_clusters: int
    max_clusters: int


@all_nullable_dataclass
class IMS:
    bandwidth: float


@all_nullable_dataclass
class IDeT:
    density_threshold: float
    duration_threshold: float
    min_pts: int


@all_nullable_dataclass
class String_Distance:
    deletion_cost: float
    insertion_cost: float
    normalization: str  # min | max


@all_nullable_dataclass
class Levenshtein_Distance(String_Distance):
    substitution_cost: float


@all_nullable_dataclass
class Generalized_Edit_Distance(String_Distance):
    pass


@all_nullable_dataclass
class Needleman_Wunsch_Distance:
    concordance_bonus: float
    gap_cost: 0.25
    normalization: str  # min | max


@all_nullable_dataclass
class Smith_Waterman:
    base_deletion_cost: float
    iterative_deletion_cost: float
    similarity_threshold: float
    similarity_weight: float


@all_nullable_dataclass
class Temporal_Binning:
    temporal_binning: bool
    temporal_binning_length: float


Identification = tagged_union_disjoint_types(
    "Identification",
    "method",
    {
        "iap": IAP,
        "idp": IDP,
        "idt": IDT,
        "ikm": IKM,
        "ims": IMS,
    },
)


@all_nullable_dataclass
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


@all_nullable_dataclass
class IBDT:
    duration_threshold: float
    fixation_sd: float
    fixation_threshold: float
    pursuit_threshold: float
    saccade_sd: float
    saccade_threshold: float


@all_nullable_dataclass
class I_DiT:
    dispersion_threshold: float
    window_duration: float


@all_nullable_dataclass
class IFC:
    bcea_prob: float
    classifier: str  # to specify more closely
    i2mc: bool
    i2mc_moving_threshold: float
    i2mc_window_duration: float


@all_nullable_dataclass
class IKF:
    chi2_threshold: float
    chi2_window: float
    chi2_sigma: float
    sigma_1: float
    sigma_2: float


@all_nullable_dataclass
class IMST:
    distance_threshold: float
    window_duration: float
    step_samples: int | None
    min_cluster_size: int | None


@all_nullable_dataclass
class IVDT:
    dispersion_threshold: float
    saccade_threshold: float
    window_duration: float


@all_nullable_dataclass
class IVMP:
    # Paper link: Part2 l.345
    rayleigh_threshold: float
    saccade_threshold: float
    window_duration: float


@all_nullable_dataclass
class IVT:
    # Paper link: Part2 l.189
    velocity_threshold: float


@all_nullable_dataclass
class IVVT:
    pursuit_threshold: float
    saccade_threshold: float


@all_nullable_dataclass
class TDE_distance:
    method: None
    scaling: None
    subsequence_length: None


@all_nullable_dataclass
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


@all_nullable_dataclass
class Multimatch_Simplification:
    amplitude_threshold: float
    angular_threshold: float
    duration_threshold: float
    iterations: int


@all_nullable_dataclass
class Persistence:
    display: bool
    landscape_order: int


@all_nullable_dataclass
class Scanmatch_Score:
    concordance_bonus: None
    gap_cost: None
    substitution_threshold: None


@all_nullable_dataclass
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


@all_nullable_dataclass
class BaseSmoothing:
    window_length: int


@all_nullable_dataclass
class MovingAverage(BaseSmoothing):
    pass


@all_nullable_dataclass
class SpeedMovingAverage(BaseSmoothing):
    pass


@all_nullable_dataclass
class Savgol(BaseSmoothing):
    polyorder: int


Smoothing = tagged_union_disjoint_types(
    "Smoothing",
    "method",
    {
        "savgol": Savgol,
        "moving_average": MovingAverage,
        "speed_moving_average": SpeedMovingAverage,
    },
)


@all_nullable_dataclass
class Fixation:
    weighted_average_velocity_means: bool
    BCEA_probability: float


@all_nullable_dataclass
class Saccade:
    absolute_horizontal_deviations: bool
    init_direction_duration_threshold: float
    init_deviation_duration_threshold: float
    weighted_average_velocity_means: bool
    weighted_average_acceleration_profiles: bool
    weighted_average_acceleration_means: bool
    weighted_average_deceleration_means: bool


@all_nullable_dataclass
class Pursuit:
    end_idx: None | int

    onset_baseline_length: None
    onset_slope_length: None
    onset_threshold: None

    start_idx: int


@all_nullable_dataclass
class ScreenDimensions:
    x: float
    y: float
    diagonal: float


T = TypeVar("T", bound=Union[float, int])


@all_nullable_dataclass
class FilterRange(Generic[T]):
    min: T | None
    max: T | None


@all_nullable_dataclass
class SegmentationFilter:
    fixation_duration: FilterRange[float]
    pursuit_duration: FilterRange[float]
    saccade_duration: FilterRange[float]
    interval_size: FilterRange[int]
    status_threshold: float | None


@all_nullable_dataclass
class SerieMetadata:
    sampling_frequency: int
    nb_samples: int


Segmentation = tagged_union_disjoint_types(
    "Segmentation",
    "method",
    {
        UnionTag(attr="iHMM", value="I_HMM"): HMM,
        UnionTag(attr="I2MC", value="I_2MC"): I2MC,
        UnionTag(attr="IBDT", value="I_BDT"): IBDT,
        UnionTag(attr="I_DiT", value="I_DiT"): I_DiT,
        UnionTag(attr="IFC", value="IFC"): IFC,
        UnionTag(attr="IKF", value="I_KF"): IKF,
        UnionTag(attr="IMST", value="I_MST"): IMST,
        UnionTag(attr="IVDT", value="I_VDT"): IVDT,
        UnionTag(attr="IVMP", value="I_VMP"): IVMP,
        UnionTag(attr="IVT", value="I_VT"): IVT,
        UnionTag(attr="IVVT", value="I_VVT"): IVVT,
        UnionTag(attr="I_DeT", value="I_DeT"): IDeT,
    },
    filter=SegmentationFilter,  # Add filter as an extra field
    fixation=Fixation,
    saccade=Saccade,
    pursuit=Pursuit,
)


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

    screen_dimensions: ScreenDimensions

    subsmatch_ngram_length: int
    task: str  # binary | ternary

    verbose: bool
    nb_samples_pursuit: int
    display: Display


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


def isatomic(obj):
    return isinstance(
        obj,
        (
            str,
            int,
            float,
            bytes,
        ),
    )


def common_fields(objs):
    return set.intersection(
        *(set(getattr(obj, "__dataclass_fields__", [])) for obj in objs)
    )


class StackedObject:
    def __init__(self, stack):
        fields = common_fields(stack)
        if not fields:
            raise TypeError

        self.fields = fields
        self.stack = stack

    @property
    def __dataclass_fields__(self):
        return self.fields

    def __getattr__(self, attr_name: str):
        if attr_name not in self.fields:
            raise AttributeError

        for c in reversed(self.stack):
            value = getattr(c, attr_name, None)

            if value is not None:
                if isatomic(value):
                    return value

                stack = [
                    value
                    for el in self.stack
                    if (value := getattr(el, attr_name, None)) is not None
                ]
                return StackedObject(stack)

        return None


class StackedConfig(StackedObject, ConfigMixin):
    __dataclass_fields__ = Common.__dataclass_fields__

    # Quickly implemented class for stacking config which is useful to
    # build bit by bit a config
    def __init__(self, config=None):
        super().__init__(
            config
            if isinstance(config, list)
            else ([config] if config is not None else [])
        )

    def __iadd__(self, other: Config):
        self.stack.append(other)

        return self
