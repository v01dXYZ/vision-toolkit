# Analysis

## Loading the data

```sh
pip install \
    --extra-index-url __VISION_TOOLKIT_PYINDEX__ \
    vision-toolkit==__VISION_TOOLKIT_VERSION_IDENTIFIER__
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import vision_toolkit2 as vt2
```

```python
df = pd.read_csv(f"{vt2.example_dataset_dir()}/data_1.csv")
df
```

```python
serie = vt2.Serie.read_csv(
    f"{vt2.example_dataset_dir()}/data_1.csv",
    size_plan_x=512,
    size_plan_y=512,
    sampling_frequency=1000,
    distance_type="euclidean",
)
```

```python
sns.lineplot({"x": serie.x, "y": serie.y});
```

```python
plt.plot(serie.x, serie.y);
```

```python
sns.histplot(x=serie.x, y=serie.y, bins=50);
```

```python
sns.lineplot(serie.absolute_speed);
```

```python
from vision_toolkit2.segmentation.analysis import (
    saccade as sac_analysis,
    fixation as fix_analysis,
    pursuit as pur_analysis,
    pursuit_task as pt_analysis,
)
```

```python
sac_analysis.count(serie)
```

## Fixation Analysis Functions

The Vision Toolkit provides comprehensive fixation analysis functions:

### Basic Metrics
- `count()` - Count the number of fixations
- `frequency()` - Calculate fixation frequency
- `frequency_wrt_labels()` - Frequency with respect to labels
- `durations()` - Fixation durations

**Example:**
```python
fix_analysis.count(serie)
```
```python
fix_analysis.frequency(serie)
```
```python
fix_analysis.frequency_wrt_labels(serie)
```
```python
fix_analysis.durations(serie)
```

### Spatial Analysis
- `centroids()` - Fixation centroids (center points)
- `drift_displacements()` - Displacement during fixations
- `drift_distances()` - Total distance traveled during fixations
- `drift_velocities()` - Velocities during fixations

**Example:**
```python
fix_analysis.centroids(serie)
```
```python
fix_analysis.drift_displacements(serie)
```
```python
fix_analysis.drift_distances(serie)
```
```python
fix_analysis.drift_velocities(serie)
```

### Advanced Metrics
- `BCEA()` - Bivariate Contour Ellipse Area (spatial dispersion)
- `average_velocity_means()` - Average velocity means
- `average_velocity_deviations()` - Deviations in average velocities

**Example:**
```python
fix_analysis.BCEA(serie)
```
```python
fix_analysis.average_velocity_means(serie)
```
```python
fix_analysis.average_velocity_deviations(serie)
```

## Saccade Analysis Functions

The Vision Toolkit provides 30 easy-access saccade analysis functions:

### Basic Metrics
- `count()` - Count the number of saccades
- `frequency()` - Calculate saccade frequency
- `frequency_wrt_labels()` - Frequency with respect to labels
- `durations()` - Saccade durations

**Example:**
```python
sac_analysis.count(serie)
```
```python
sac_analysis.frequency(serie)
```
```python
sac_analysis.frequency_wrt_labels(serie)
```
```python
sac_analysis.durations(serie)
```

### Spatial Analysis
- `amplitudes()` - Saccade amplitudes (start to end distance)
- `travel_distances()` - Total distance traveled during saccades
- `efficiencies()` - Saccade efficiency (amplitude/travel distance)

**Example:**
```python
sac_analysis.amplitudes(serie)
```
```python
sac_analysis.travel_distances(serie)
```
```python
sac_analysis.efficiencies(serie)
```

### Directional Analysis
- `directions()` - Saccade directions
- `horizontal_deviations()` - Horizontal deviations from straight path
- `successive_deviations()` - Deviations between consecutive saccades
- `initial_directions()` - Initial direction of saccades
- `initial_deviations()` - Initial deviation from final direction

**Example:**
```python
sac_analysis.directions(serie)
```
```python
sac_analysis.horizontal_deviations(serie)
```
```python
sac_analysis.successive_deviations(serie)
```
```python
sac_analysis.initial_directions(serie)
```
```python
sac_analysis.initial_deviations(serie)
```

### Velocity Analysis
- `mean_velocities()` - Mean velocities during saccades
- `average_velocity_means()` - Average of mean velocities
- `average_velocity_deviations()` - Deviations in average velocities
- `peak_velocities()` - Peak velocities during saccades

**Example:**
```python
sac_analysis.mean_velocities(serie)
```
```python
sac_analysis.average_velocity_means(serie)
```
```python
sac_analysis.average_velocity_deviations(serie)
```
```python
sac_analysis.peak_velocities(serie)
```

### Acceleration Analysis
- `mean_acceleration_profiles()` - Mean acceleration profiles
- `mean_accelerations()` - Mean accelerations
- `mean_decelerations()` - Mean decelerations
- `average_acceleration_profiles()` - Average acceleration profiles
- `average_acceleration_means()` - Average acceleration means
- `average_deceleration_means()` - Average deceleration means
- `peak_accelerations()` - Peak accelerations
- `peak_decelerations()` - Peak decelerations

**Example:**
```python
sac_analysis.mean_acceleration_profiles(serie)
```
```python
sac_analysis.mean_accelerations(serie)
```
```python
sac_analysis.mean_decelerations(serie)
```
```python
sac_analysis.average_acceleration_profiles(serie)
```
```python
sac_analysis.average_acceleration_means(serie)
```
```python
sac_analysis.average_deceleration_means(serie)
```
```python
sac_analysis.peak_accelerations(serie)
```
```python
sac_analysis.peak_decelerations(serie)
```

### Curvature Analysis
- `max_curvatures()` - Maximum curvatures during saccades
- `area_curvatures()` - Area under curvature curves

**Example:**
```python
sac_analysis.max_curvatures(serie)
```
```python
sac_analysis.area_curvatures(serie)
```

### Advanced Metrics
- `skewness_exponents()` - Skewness exponents
- `gamma_skewness_exponents()` - Gamma distribution skewness
- `amplitude_duration_ratios()` - Amplitude/duration ratios
- `peak_velocity_amplitude_ratios()` - Peak velocity/amplitude ratios
- `peak_velocity_duration_ratios()` - Peak velocity/duration ratios
- `peak_velocity_velocity_ratios()` - Peak velocity/velocity ratios
- `acceleration_deceleration_ratios()` - Acceleration/deceleration ratios
- `main_sequence()` - Main sequence analysis

**Example:**
```python
sac_analysis.skewness_exponents(serie)
```
```python
sac_analysis.gamma_skewness_exponents(serie)
```
```python
sac_analysis.amplitude_duration_ratios(serie)
```
```python
sac_analysis.peak_velocity_amplitude_ratios(serie)
```
```python
sac_analysis.peak_velocity_duration_ratios(serie)
```
```python
sac_analysis.peak_velocity_velocity_ratios(serie)
```
```python
sac_analysis.acceleration_deceleration_ratios(serie)
```
```python
sac_analysis.main_sequence(serie)
```

## Pursuit Analysis Functions

The Vision Toolkit provides comprehensive pursuit analysis functions:

### Basic Metrics
- `count()` - Count the number of pursuits
- `frequency()` - Calculate pursuit frequency
- `durations()` - Pursuit durations
- `proportion()` - Proportion of time spent in pursuit

**Example:**
```python
pur_analysis.count(serie)
```
```python
pur_analysis.frequency(serie)
```
```python
pur_analysis.durations(serie)
```
```python
pur_analysis.proportion(serie)
```

### Velocity Analysis
- `velocity()` - Pursuit velocities
- `velocity_means()` - Mean velocities during pursuits
- `peak_velocity()` - Peak velocities during pursuits

**Example:**
```python
pur_analysis.velocity(serie)
```
```python
pur_analysis.velocity_means(serie)
```
```python
pur_analysis.peak_velocity(serie)
```

### Spatial Analysis
- `amplitude()` - Pursuit amplitudes
- `distance()` - Total distance traveled during pursuits
- `efficiency()` - Pursuit efficiency (amplitude/distance)

**Example:**
```python
pur_analysis.amplitude(serie)
```
```python
pur_analysis.distance(serie)
```
```python
pur_analysis.efficiency(serie)
```

## Pursuit Task Analysis Functions

The Vision Toolkit provides specialized pursuit task analysis functions that compare eye movements with theoretical target trajectories:

**Note:** Pursuit task analysis functions require a `df_theo` DataFrame parameter containing theoretical target coordinates.

### Loading Example Pursuit Data

```python
# Load example pursuit data with theoretical coordinates
df_pt = pd.read_csv(f"{vt2.example_dataset_dir()}/example_pursuit.csv")
df_theo_pt = pd.read_csv(f"{vt2.example_dataset_dir()}/example_pursuit_theo.csv")
```

```python
df_pt
```

```python
df_theo_pt
```

```python
# Create a Serie from the pursuit data
serie_pt = vt2.Serie.read_csv(
    f"{vt2.example_dataset_dir()}/example_pursuit.csv",
    size_plan_x=512,
    size_plan_y=512,
    sampling_frequency=1000,
    distance_type="euclidean",
)
```

### Basic Metrics
- `count()` - Count the number of pursuits
- `frequency()` - Calculate pursuit frequency  
- `durations()` - Pursuit durations
- `proportion()` - Proportion of time spent in pursuit

**Example:**
```python
pt_analysis.count(serie_pt, df_theo_pt)
```
```python
pt_analysis.frequency(serie_pt, df_theo_pt)
```
```python
pt_analysis.durations(serie_pt, df_theo_pt)
```
```python
pt_analysis.proportion(serie_pt, df_theo_pt)
```

### Comparison Metrics
- `slope_ratios()` - Ratio of eye movement slope to theoretical slope
- `slope_gain()` - Gain between eye and theoretical movement slopes
- `overall_gain()` - Overall gain across all dimensions
- `overall_gain_x()` - Gain in X dimension
- `overall_gain_y()` - Gain in Y dimension

**Example:**
```python
pt_analysis.slope_ratios(serie_pt, df_theo_pt)
```
```python
pt_analysis.slope_gain(serie_pt, df_theo_pt)
```
```python
pt_analysis.overall_gain(serie_pt, df_theo_pt)
```
```python
pt_analysis.overall_gain_x(serie_pt, df_theo_pt)
```
```python
pt_analysis.overall_gain_y(serie_pt, df_theo_pt)
```

### Advanced Analysis
- `sinusoidal_phase()` - Phase difference for sinusoidal movements
- `crossing_time()` - Time when eye crosses theoretical target path
- `accuracy()` - Accuracy of pursuit relative to theoretical target
- `entropy()` - Entropy measure of pursuit performance

**Example:**
```python
pt_analysis.sinusoidal_phase(serie_pt, df_theo_pt)
```
```python
pt_analysis.crossing_time(serie_pt, df_theo_pt)
```
```python
pt_analysis.accuracy(serie_pt, df_theo_pt)
```
```python
pt_analysis.entropy(serie_pt, df_theo_pt)
```