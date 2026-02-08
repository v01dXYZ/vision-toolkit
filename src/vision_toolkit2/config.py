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
            [ (

(((cls.__name__.lower() if lower else cls.__name__) + "_") if class_name and not a.startswith("_") else prefix)

+ a.lstrip("_"), t, dataclasses.field(default=None)) 
              for c in (cls, *cls.__bases__)
              if c is not object
              for a, t in c.__annotations__.items()

             ],
#            bases=cls.__bases__,
            
        )

    return f

@prefix(class_name=True, lower=False)
class HMM:
    init_low_velocity: float
    init_high_velocity: float
    init_variance: float
    nb_iters: int

@prefix(class_name=True, lower=False)
class I2MC:
    merging_distance_threshold: float # [0, 1]?
    merging_duration_threshold: float # [0, 1]?
    moving_threshold: float # [0, 1]?
    window_duration: float

@prefix(class_name=True, lower=False)
class CDBA:
    initial_random_state: int
    initialization_length: str # min | max
    maximum_iterations: int

@prefix(class_name=True, lower=False)
class IAP:
    centers    : str # mean | raw_IAP

@prefix(class_name=True, lower=False)
class IDP:
    centers: str # mean | raw_IDP
    gaussian_kernel_sd: float

@prefix(class_name=True, lower=False)
class IDT:
    density_threshold: float
    min_samples: int

@prefix(class_name=True, lower=False)
class IKM:
    cluster_number: str # search
    min_clusters: int
    max_clusters: int


@prefix(class_name=True, lower=False)
class IMS:
    bandwidth: float

@prefix(class_name=True, lower=False)
class IDeT:
    density_threshold: float
    duration_threshold: float

@prefix(class_name=False)
class String_Distance:
    deletion_cost: float
    insertion_cost: float
    normalization: str # min | max

@prefix(class_name=True)
class Levenshtein_Distance(String_Distance):
    substitution_cost: float

@prefix(class_name=True)
class Generalized_Edit_Distance(String_Distance):
    pass

@prefix(class_name=True)
class Needleman_Wunsch_Distance:
    concordance_bonus: float
    gap_cost: 0.25
    normalization: str # min | max

@prefix(class_name=True)
class Smith_Waterman:
    base_deletion_cost: float
    iterative_deletion_cost: float
    similarity_threshold: float
    similarity_weight: float

@prefix()
class Temporal_Binning:
    temporal_binning: bool
    temporal_binning_length: float

@prefix(class_name=True, lower=False)
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
    SPAM_support: float
    coordinates: None # array[2, 2]

    identification_method: None
    longest_common_subsequence_normalization: str # min | max

    predefined_all: bool
    predefined_coordinates: None # list[array[2, 2]]


    trend_analysis_tolerance_level: None

@prefix(class_name=True, lower=False)
class IBDT:
    duration_threshold: float
    fixation_sd: float
    fixation_threshold: float
    pursuit_threshold: float
    saccade_sd: float
    saccade_threshold: float

@prefix(class_name=True, lower=False)
class ICNN:
    batch_size: int
    learning_rate: float
    num_epochs: int
    temporal_window_size: int

@prefix(class_name=True, lower=False)
class I_DiT:
    dispersion_threshold: float
    window_duration: float

@prefix(class_name=True, lower=False)
class IFC:
    bcea_prob: float
    classifier: str # to specify more closely
    i2mc: bool
    i2mc_moving_threshold: float
    i2mc_window_duration: float

@prefix(class_name=True, lower=False)
class IHOV:
    angular_bin_nbr: int
    averaging_threshold: float
    classifier: str # to specify more closely
    duration_threshold: float

@prefix(class_name=True, lower=False)
class IKF:
    chi2_threshold: float
    chi2_window: float
    sigma_1: float
    sigma_2: float

@prefix(class_name=True, lower=False)
class IMST:
    distance_threshold: float
    window_duration: float

@prefix(class_name=True, lower=False)
class IVDT:
    dispersion_threshold: float
    saccade_threshold: float
    window_duration: float

@prefix(class_name=True, lower=False)
class IVMP:
    # Paper link: Part2 l.345
    rayleigh_threshold: float
    saccade_threshold: float
    window_duration: float

@prefix(class_name=True, lower=False)
class IVT:
    # Paper link: Part2 l.189
    velocity_threshold: float

@prefix(class_name=True, lower=False)
class IVVT:
    pursuit_threshold: float
    saccade_threshold: float

@prefix(class_name=True, lower=False)
class TDE_distance:
    method: None
    scaling: None
    subsequence_length: None

@prefix(class_name=True)
class Display:
    _display: None # means display itself

    AoI: bool
    AoI_path: None | str
    path: None | str
    results: bool
    scanpath: bool
    scanpath_path: None | str
    segmentation: bool
    segmentation_path: None | str

@prefix(class_name=True)
class Multimatch_Simplification:
    amplitude_threshold: float
    angular_threshold: float
    duration_threshold: float
    iterations: int

@prefix(class_name=True)
class Persistence:
    display: bool
    landscape_order: int


@prefix(class_name=True)
class Scanmatch_Score:
    concordance_bonus: None
    gap_cost: None
    substitution_threshold: None


@prefix(class_name=True)
class Scanpath(
        Levenshtein_Distance,
        Needleman_Wunsch_Distance,
        Generalized_Edit_Distance,
        Temporal_Binning,
):
    CRQA_distance_threshold: float
    CRQA_minimum_length: float
    RQA_distance_threshold: float
    RQA_minimum_length: float

    spatial_binning_nb_pixels_x: int
    spatial_binning_nb_pixels_y: int

@prefix()
class Smoothing:
    smoothing: str # moving_average | speed_moving_average | savgol

@prefix(class_name=True)
class Savgol:
    polyorder: int
    window_length: int
    
@prefix(class_name=True)
class Pursuit:
    end_idx: None | int

    onset_baseline_length: None
    onset_slope_length: None
    onset_threshold: None

    start_idx: int

@prefix()
class Common(
        HMM,
        I2MC,
        AoI,
        IBDT,
        ICNN,
        I_DiT,
        IFC,
        IHOV,
        IKF,
        IMST,
        IVDT,
        IVMP,
        IVT,
        IVVT,
        IDeT,
        TDE_distance,
        Display,
        Multimatch_Simplification,
        Persistence,
        Scanmatch_Score,
        Scanpath,
        Smoothing,
        Savgol,
        Pursuit,
):
    curve_nb_points: int

    distance_projection: int | None
    distance_type: str # euclidean | angular

    mannan_distance_nb_random_scanpaths: int

    max_fix_duration: float
    min_fix_duration: float
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

    size_plan_x: float
    size_plan_y: float



    status_threshold: float # (between 0 and 1?)
    subsmatch_ngram_length: int
    task: str # binary | ternary

    verbose: bool

class ConfigMixin:
    def update(self, **kwargs):
        # UGLY BUT WORKS
        return type(self)(
            **( asdict(self) | kwargs),
        )

    def merge(self, other):
        return type(self)(
            **(asdict(self) | asdict(other))
        )

    def print(self):
        print("Config used")
        print("-"*15)

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
