# Features and metrics for assessing oculomotor signal
## Introduction

Eye movement research has a rich history, beginning with foundational work by Dodge and Cline (1901)[^B42] in the early <inline-formula id="inf1">
<mml:math id="m1">
<mml:mrow>
<mml:mn>2</mml:mn>
<mml:msup>
<mml:mrow>
<mml:mn>0</mml:mn>
</mml:mrow>
<mml:mrow>
<mml:mi>t</mml:mi>
<mml:mi>h</mml:mi>
</mml:mrow>
</mml:msup>
</mml:mrow>
</mml:math>
</inline-formula> century. Technological advancements have since enhanced the measurement, storage, and analysis of eye movements, enabling significant progress in understanding their underlying mechanisms. The growing accessibility of eye-tracking tools has expanded their use across global research laboratories, fostering specialized subfields like neuroscience, psychology, marketing, and medicine. Each discipline has provided critical insights, collectively shaping modern eye movement research.


A primary goal in eye movement research is to extract metrics that characterize the oculomotor system. Due to their close link with visual attention, eye movements analysis is a powerful tool for studying cognitive and behavioral processes. Recent studies have integrated eye movement analysis into cognitive psychology, exploring areas like language processing, reading, and problem-solving (Rayner, 1998[^B165]). Research has also investigated connections between eye movements, visual attention, and perception (Collins and Doré-Mazars, 2006[^B34]; Schütz et al., 2011[^B184]). Additionally, individual differences in oculomotor patterns have paved the way for eye movement biometrics (Rigas and Komogortsev, 2016[^B168]).


Clinical research increasingly employs eye movement analysis as a non-invasive method to identify neural irregularities linked to neurodegenerative and neurological disorders (MacAskill and Anderson, 2016[^B132]). Distinct oculomotor patterns have been observed in individuals with early-stage Alzheimer’s disease (Fernández et al., 2013[^B53]) and Parkinson’s disease (Wetzel et al., 2011[^B223]), highlighting their potential as biomarkers for early diagnosis and disease monitoring. Furthermore, a growing body of evidence explores oculomotor features in behavioral disorders such as attention deficit hyperactivity disorder (ADHD) (Fried et al., 2014[^B59]) and autism spectrum disorder (ASD) (Klin et al., 2002[^B103]; Shirama et al., 2016[^B192]), offering valuable insights into the neurocognitive mechanisms underlying these conditions.


The rapid growth of eye movement research has also brought significant challenges. The increasing volume of publications can obscure critical insights, while fragmentation across sub-disciplines hinders effective knowledge integration. As the different research communities pursue distinct objectives, definitions and methodologies often become highly specialized, which limits their generalizability. This has contributed to a fragmented conceptual framework within the field. Notably, a recent study highlights that even fundamental terms such as *fixation* and *saccade* are defined inconsistently, resulting in *conceptual confusion* (Hessels et al., 2018[^B85]). These definitions vary considerably depending on whether the perspective is functional, oculomotor, or computational, with little consensus even within individual subfields.


Beyond conceptual and terminological inconsistencies, the field lacks standardized methods for defining and extracting eye movement features. Most studies emphasize feature subsets tailored to specific research questions, and the methodological variability in segmenting raw gaze data into canonical movements—such as fixations, saccades, and smooth pursuits—undermines reproducibility. The growing availability of portable, cost-effective eye-tracking devices has facilitated the study of naturalistic behavior in both laboratory and real-world settings (Hayhoe and Ballard, 2005[^B79]; Land, 2009[^B118]). However, the absence of standardized analysis protocols limits comparability between studies and hinders the integration of knowledge. This work aims to address these challenges by proposing a unified methodological framework to improve interoperability across research communities and improve comparison across experimental contexts.


This review focuses on methods for segmenting, extracting and analyzing fixations, saccades, and smooth pursuits, building on prior comprehensive reviews of fixation and saccade features (Sharafi et al., 2015[^B187]; Rigas et al., 2018[^B169]; Brunyé et al., 2019[^B22]; Skaramagkas et al., 2021[^B193]; Mahanama et al., 2022a[^B134]; Spering, 2022[^B199]) and pursuit-based features (Skaramagkas et al., 2021[^B193]; Mahanama et al., 2022a[^B134]; Spering, 2022[^B199]). Some reviews target specific domains, such as emotional and cognitive processes (Skaramagkas et al., 2021[^B193]) or decision-making (Spering, 2022[^B199]). Additionally, several studies, including Komogortsev et al. (2010b)[^B108]; Birawo and Kasprowski (2022)[^B17]; Startsev and Zemblys (2023)[^B201], evaluate segmentation algorithms, often comparing their performance on open-source datasets and proposing quality metrics. This work aligns with these efforts by reviewing segmentation methods and their associated oculomotor features.


Specifically, this review surveys methodologies for quantifying oculomotor system activity and explores their diverse applications. While not exhaustive due to the breadth and specialization of some methods, it provides a concise overview of key approaches for characterizing canonical eye movements and their oculometric signals. The following sections are organized as follows.  introduces segmentation algorithms for classifying fixations, saccades, and smooth pursuits. Two primary analytical approaches are then explored: physiological analysis— — which extracts meaningful features like shape, dynamics, and kinematics from segmented sequences, and signal-based analysis— — which applies time-series descriptors to examine eye movement behavior from a global dynamic perspective. Although a detailed discussion of metrics is beyond the scope of this review, we aim to provide a unified framework for oculometric signal analysis.


This article is part of a series of four reviews dedicated to methods for analyzing oculomotor signals and gaze trajectories. The overarching goals of the series are to evaluate the application of eye movement and gaze analysis techniques across diverse scientific disciplines and to work toward a unified methodological framework by defining standardized representations and concepts for quantifying eye-tracking data. The first article in the series, already published in *Frontiers in Physiology* (Laborde et al., 2025[^B115]), provided an overview of current knowledge on canonical eye movements, with particular emphasis on distinguishing findings obtained in controlled laboratory settings from those observed in more natural, head-free conditions.


## Segmentation algorithms

Three archetypal gaze patterns can typically be observed in eye-tracking data: periods of relative stability, rapid eye shifts, and slower shifts corresponding to the tracking of moving objects. These are commonly assumed to reflect the three main canonical oculomotor events that direct gaze movements, namely, fixations, saccades and smooth pursuits. Thus, a necessary preliminary step in eye-movement analysis is often to identify these canonical events from a continuous stream of gaze data using segmentation algorithms. Segmentation algorithms employ a number of predefined criteria, based on the underlying characteristics of the oculomotor events, in order to distinguish them. Such a process aligns with the traditional neurophysiological view, which postulates that distinct neural mechanisms govern specific movement types, such as the superior colliculus for saccades or the cerebellum for smooth pursuits.


However, the organization of the oculomotor system as a discrete set of events has been questioned, notably in the context of natural viewing conditions (Steinman et al., 1990[^B203]). Under ecological conditions, a richer repertoire of ocular behavior can be observed. This results in potential overlap between the characteristics of the oculomotor events, which makes the segmentation task more challenging. Therefore, it seems more appropriate to refer to segmentation algorithms as event classification rather than event detection, since they merely assign a discrete event type to each data period based on some computationally inferred features—*e.g.*, velocity thresholds for saccades or duration thresholds for fixations. This distinction is critical, as misclassification can distort interpretations of visual attention in fields such as psychology, neuroscience, and human-computer interaction.


A major challenge in eye movement segmentation is the dependence on user-defined parameters, such as velocity thresholds for saccades or minimum fixation durations. Although these events are grounded in physiological phenomena, no theoretical consensus exists on parameter values that definitively distinguish movement types. For instance, the transition from slow movements, such as smooth pursuits or drifts, to rapid saccades lacks a clear, physiologically validated threshold. Studies investigating optimal parameterization for specific algorithms (Blignaut, 2009[^B18]; Shic et al., 2008[^B191]) indicate that variations in parameter settings significantly influence classification outcomes (Komogortsev et al., 2010b[^B108]; Salvucci and Goldberg, 2000[^B176]). This sensitivity hampers reproducibility and can distort findings in fields requiring precise event classification, such as psychology or human-computer interaction. In psychology, for example, precision in detecting fixations is crucial for analyzing attention strategies, such as in studies on reading or visual information processing (Rayner, 1998[^B165]). For instance, in experimental paradigms measuring cognitive load, accurate identification of fixations enables reliable quantification of the time spent on specific stimuli, thereby revealing underlying attentional processes (Duchowski and Duchowski, 2017[^B45]). In human-computer interaction (HCI), precise classification of eye movement events is equally important for evaluating the usability of user interfaces (Jacob and Karn, 2003[^B93]). Correct detection of saccades and fixations, for example, allows for the identification of interface areas that attract users’ attention or pose accessibility issues, directly influencing the design of more intuitive interfaces.


Conversely, errors in the detection of fixations or saccades can have significant repercussions on the interpretation of data in studies in cognitive psychology and human-computer interaction (HCI). As shown by Duchowski and Duchowski (2017)[^B45] and Nyström and Holmqvist (2010)[^B153], erroneous classification of eye movement events can bias the analysis of attentional processes or user behaviors. For example, a fixation incorrectly identified as a saccade can distort measures of cognitive load in experimental paradigms, leading to erroneous conclusions about underlying cognitive mechanisms (Rayner, 1998[^B165]). Similarly, in HCI, imprecise detection of eye movement events can result in an incorrect evaluation of an interface’s usability, affecting recommendations for its optimization (Jacob and Karn, 2003[^B93]). As such, threshold-based methods, including velocity or dispersion thresholding, provide computational interpretations of oculomotor events, but their criteria often vary across studies and implementations, leading to inconsistent classifications of identical gaze data due to insufficient standardization, which compromises the reproducibility of results in contexts requiring high precision (Holmqvist et al., 2011[^B86]).


Finally, researchers must consider the coordinate system used when analyzing eye-tracking data, particularly with mobile eye trackers that permit free head movement. Unlike stationary trackers, which use a head-referenced coordinate system, mobile trackers record gaze in a world-referenced system, where head movements can complicate event classification. To avoid such conceptual confusion, researchers should ensure proper head movement compensation and clearly report their coordinate system. For a detailed discussion of challenges in defining oculomotor events, see the review by Hessels et al. (2018)[^B85]. Note that considerations regarding the utilization and transformation of these coordinates in relation to a moving observer’s visual field are addressed in the first part of this review series (Laborde et al., 2025[^B115])


Although some authors have called for the standardization of eye movement classification algorithms and evaluation tools (Komogortsev et al., 2010a[^B107]), Startsev and Zemblys (2023)[^B201], there is currently no clear consensus on how to benchmark these methods. This lack of agreement poses challenges to the development and comparison of new segmentation approaches. To address this gap, several concrete proposals have been suggested in the literature. First, minimal reporting standards could be established, requiring authors to clearly specify algorithm parameters, eye-tracker sampling rates, stimulus types, and data preprocessing steps. Second, the use of shared, openly available datasets would enable reproducible evaluation across diverse conditions, including static, dynamic, and naturalistic stimuli. Third, benchmark competitions or challenges could be organized, similar to practices in computer vision and machine learning, where algorithms are tested on identical datasets using standardized metrics such as precision, recall, F1-score, Cohen’s Kappa, and RMSD. By adopting these practices, the field could facilitate more transparent, reproducible, and comparable assessments of eye movement segmentation algorithms, ultimately accelerating methodological improvements.


In this review, we focus on fixations, saccades, and smooth pursuit eye movements, as these are the most commonly studied and well-characterized oculomotor events in the literature. Other canonical eye movement events, such as vergence, optokinetic reflexes, and vestibulo-ocular reflex (VOR), are not included. These events are less frequently analyzed in eye-tracking studies, and their detection often requires specialized experimental setups or instrumentation beyond conventional gaze-tracking paradigms. By concentrating on fixations, saccades, and pursuits, we ensure that the discussion is grounded in well-supported empirical evidence while acknowledging that additional eye movement types remain an important direction for future work. Despite these challenges, the following sections provide an overview of widely used segmentation methods (Salvucci and Goldberg, 2000[^B176]; Komogortsev and Karpov, 2013[^B105]; Andersson et al., 2016[^B5]).


### Separating saccades from fixations

Numerous algorithms have been developed to address the challenge of distinguishing saccades from fixations, a process known as binary segmentation. This is illustrated in , which depicts alternating periods of relative gaze stability—fixations, marked in purple—and rapid gaze reorientations—saccadic eye movements. The recording shown in  is of exceptionally high quality, with minimal noise or signal loss. In contrast, real-world eye-tracking data often exhibit lower quality due to several factors. For instance, blinks or partial eyelid closures interrupt the signal, while head movements or poor participant stabilization can introduce spatial jitter. Changes in lighting conditions or reflections on glasses can reduce the accuracy of gaze detection, and low sampling rates or occasional data dropouts may cause missing or irregular samples. Additionally, physiological variability, such as micro-saccades or pupil size fluctuations, can further complicate event classification. These factors collectively increase the difficulty of distinguishing fixations from saccades, emphasizing the need for robust segmentation algorithms that can tolerate noise and handle incomplete or variable-quality data.

<figure id="F1" position="float">

<img src="/assets/fphys-16-1661026-g001.webp"/><figcaption>
<p>Binary Segmentation. This example illustrates an oculomotor recording containing both fixations and saccades. Panel <bold>(a)</bold> depicts the two-dimensional gaze trajectory, with alternating periods of stability—fixations shown in purple—and rapid ballistic reorientations—saccades shown in gray. Panels <bold>(b,c)</bold> present the horizontal and vertical gaze positions over time, respectively, using the same color scheme. These characteristic patterns form the basis of <italic>binary segmentation algorithms</italic>, which aim to distinguish fixation sequences from saccadic sequences.</p>
</figcaption>

</figure>

Binary segmentation algorithms are broadly categorized into *threshold-based* and *learning-based* approaches. Threshold-based methods rely on predefined computational criteria, such as velocity or spatial dispersion, to classify fixations and saccades, ensuring transparent, rule-based classification. In contrast, learning-based methods, encompassing machine learning and deep learning techniques, infer patterns from annotated training data, which reflect expert or task-specific interpretations of fixations and saccades. These annotations may reduce the transparency of classification criteria compared to threshold-based methods due to their reliance on subjective or context-dependent definitions.


#### Threshold-based algorithms

The velocity-threshold identification (I-VT) algorithm (Salvucci and Goldberg, 2000[^B176]) is a widely adopted method for distinguishing fixations from saccades in eye movement data. It leverages the distinct velocity profiles of eye movements: low velocities characterize fixations, while high velocities indicate saccades. The I-VT algorithm calculates the absolute velocity between consecutive gaze samples and classifies each sample as a fixation or saccade based on a user-defined velocity threshold. To address the subjectivity of manual threshold selection, Nyström and Holmqvist (2010)[^B153] proposed an adaptive I-VT variant that dynamically computes thresholds for peak velocities and saccade onset/offset detection based on statistical properties of the data. This method incorporates constraints derived from the physical characteristics of eye movements—such as minimum and maximum velocities, accelerations, and event durations—to filter noise and enhance classification accuracy.


In contrast to velocity-based methods, the dispersion-threshold identification (I-DiT) algorithm offers an alternative approach by leveraging the tendency of fixation points—characterized by relatively low velocity—to cluster spatially (Salvucci and Goldberg, 2000[^B176]; Komogortsev et al., 2010a[^B107]; Andersson et al., 2016[^B5]). The I-DiT algorithm distinguishes fixations from saccades based on the spatial dispersion of consecutive gaze points within a defined temporal window. Dispersion is quantified by summing the ranges—*i.e.*, the differences between the maximum and minimum values—of the gaze coordinates in both the horizontal and vertical dimensions. If the resulting dispersion value falls below a predefined threshold, the corresponding gaze points are classified as a fixation. Otherwise, if the dispersion exceeds the threshold, the sequence is identified as a saccade.


Another notable approach is the minimum spanning tree (MST)-based method (Goldberg and Schryver, 1995[^B70]; Salvucci and Goldberg, 2000[^B176]; Komogortsev et al., 2010a[^B107]; Andersson et al., 2016[^B5]), which also employs a dispersion-based strategy to evaluate local gaze dispersion within a temporal window of eye position data. Unlike traditional methods, MST-based algorithms model gaze points as nodes in a graph, with edges weighted by the Euclidean distance between corresponding positions. A minimum spanning tree is constructed—typically using Prim’s algorithm (Camerini et al., 1988[^B25]) — to connect all nodes while minimizing total edge length. The identification by minimum spanning tree (I-MST) algorithm classifies gaze points by applying edge-distance thresholds: points connected by edges shorter than the threshold are grouped as fixation components, while those separated by longer edges are classified as saccadic components. Thresholds may be applied globally across the graph (Komogortsev et al., 2010a[^B107]) or adapted locally based on vertex density (Goldberg and Schryver, 1995[^B70]). The MST-based approach offers flexibility, adapts to local data structures, and demonstrates robustness in handling missing or noisy data, making it suitable for complex eye-tracking datasets.


The Density-Threshold Identification (I-DeT) algorithm is an adaptation of the DBSCAN clustering method (Ester et al., 1996[^B49]). I-DeT extends DBSCAN by incorporating the temporal dimension of gaze data, ensuring that segmented events reflect the sequential nature of eye movements. As introduced by Li et al. (2016)[^B124], a gaze point is classified as a core point if: <inline-formula id="inf2">
<mml:math id="m2">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> at least a minimum number of gaze points lie within a specified spatial radius of the reference point, forming a local neighborhood; and <inline-formula id="inf3">
<mml:math id="m3">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> these neighboring points form a temporally contiguous sequence in the gaze dataset. Fixations are identified as clusters comprising core points and their associated neighborhoods, while non-core, non-neighbor points are classified as saccades or noise. This integration of spatial and temporal constraints makes I-DeT robust for segmenting gaze data, though its performance depends on careful parameter tuning to avoid over—or under—segmentation.


Building on classical signal processing, Kalman filter-based algorithms (I-KF) model eye movements as a dynamic system. The two-state Kalman filter, as proposed by Komogortsev and Khan (2007)[^B106], represents eye movements using position and velocity states, assuming linear dynamics and Gaussian noise. The algorithm operates recursively in two phases: (i) the predict phase, which forecasts the next state based on the system model, and (ii) the update phase, which refines the prediction using observed data to produce a more accurate state estimate. Saccade detection employs a Chi-square test (Sauter et al., 1991[^B178]) to assess discrepancies between predicted and observed gaze velocities, with a threshold determining whether a sample is classified as a saccade—high velocity—or fixation—low velocity. This approach excels in handling noisy data by combining predictive modeling with statistical testing, offering a robust framework for eye movement classification applicable in fields such as human-computer interaction and clinical research.



#### Learning-based algorithms

The Hidden Markov Model Identification (I-HMM) algorithm, introduced by Salvucci and Goldberg (2000)[^B176], extends the velocity-threshold identification (I-VT) approach by employing a probabilistic framework to segment eye movements into fixations and saccades. I-HMM models eye movements as a sequence of two latent states—fixation and saccade—each characterized by a Gaussian velocity distribution. Fixations typically exhibit low mean velocity, while saccades are defined by high mean velocity—*e.g.*, <inline-formula id="inf4">
<mml:math id="m4">
<mml:mrow>
<mml:mo>&gt;</mml:mo>
<mml:mn>200</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula> degrees per second. Transitions between these states are modeled as a first-order Markov process, capturing the temporal dependencies inherent in gaze data. The approach leverages the Baum-Welch algorithm (Bilmes et al., 1998[^B15]) to estimate model parameters, including state transition probabilities and emission distribution parameters—*e.g.*, mean and variance of velocity distributions—from training data. Subsequently, the Viterbi algorithm infers the optimal sequence of states for a given gaze dataset. Unlike deterministic threshold-based methods like I-VT, I-HMM accounts for noise and sequential patterns, providing robust segmentation that is particularly effective for noisy or complex eye-tracking datasets.


The Two-Means Clustering Identification (I2MC) algorithm, introduced by Hessels et al. (2017)[^B84], is designed to extract fixations from gaze data with high noise levels, such as those recorded from infants. The algorithm employs two-means clustering—k-means with <inline-formula id="inf5">
<mml:math id="m5">
<mml:mrow>
<mml:mi>k</mml:mi>
<mml:mo>=</mml:mo>
<mml:mn>2</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula> — on a fixed-length temporal window—typically 200–400 milliseconds—to partition gaze samples into stable—fixation—and rapid—saccade—clusters based on their spatial coordinates. For each window, the number of transitions between clusters is calculated, and each gaze sample is assigned a weight inversely proportional to the number of transitions, reflecting the stability of the cluster assignment. To enhance robustness to noise, this process is applied across multiple down-sampled versions of the gaze signal. The clustering weights for each gaze sample are aggregated and averaged to generate a weight signal, which is then thresholded using an empirically determined cut-off to identify fixation periods, effectively distinguishing fixations from ballistic saccades. I2MC demonstrates robustness to data loss—*e.g.*, due to blinks or tracker errors—and was shown to outperform seven state-of-the-art algorithms on noisy infant data, making it well-suited for applications in developmental psychology, clinical research, and longitudinal studies with variable data quality (Hessels et al., 2017[^B84]).


Building upon established machine learning techniques, Zemblys et al. (2018)[^B230] introduced the Random Forest Classifier (I-RF) algorithm to distinguish fixations, saccades, and potentially other eye movement events from raw gaze data. The I-RF model is trained on a set of 14 features, including spatial measures—*e.g.*, root mean square of sample-to-sample displacement, standard deviation of gaze positions, bivariate contour ellipse area—and statistical measures—*e.g.*, sample dispersion, kurtosis. The random forest classifier leverages these features to model complex, non-linear relationships, achieving high classification accuracy. However, a key limitation is the reliance on hand-tagged training data, which is labor-intensive and hinders scalability. Reproducibility is also challenging, as model performance depends on the quality and representativeness of training datasets. Additional limitations include the computational cost of feature extraction and the risk of overfitting to specific datasets. Nevertheless, I-RF is particularly valuable in eye-tracking research for applications in cognitive psychology, human-computer interaction, and clinical diagnostics, offering robustness to noise and the potential to detect diverse eye movement types when trained appropriately.


The evaluation of binary segmentation algorithms, which aim to distinguish fixations from saccades, has been reported in benchmark studies comparing algorithm outputs to human coders using high-frequency datasets that include static images, text, moving dots, and videos (Andersson et al., 2016[^B5]). These studies provide a valuable baseline for assessing segmentation quality. Performances are generally summarized using metrics such as Cohen’s Kappa, which captures agreement with human annotations, or RMSD for event durations, which reflects temporal precision. However, reported values vary considerably depending on the dataset, the type of stimulus, and the specific evaluation protocol, making it difficult to directly compare results across studies.


Among threshold-based methods, the velocity-threshold approach (I-VT) typically reaches Kappa values around <inline-formula id="inf6">
<mml:math id="m6">
<mml:mrow>
<mml:mn>0.65</mml:mn>
<mml:mo>−</mml:mo>
<mml:mo>−</mml:mo>
<mml:mn>0.75</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula> for static image datasets but drops markedly in dynamic conditions, particularly for fixations (Andersson et al., 2016[^B5]). The dispersion-based algorithm (I-DiT) rarely exceeds 0.45 and shows high sensitivity to noise, while I-MST adapts better to missing data but yields modest agreement overall, usually between 0.3 and 0.5 (Andersson et al., 2016[^B5]). Kalman filter approaches (I-KF) report reasonable performance for saccades—up to 0.6 — but poor fixation detection. More recently, density-based methods such as I-DeT, inspired by clustering techniques, have been proposed as more robust under noise and data loss, though systematic benchmarks remain scarce (Li et al., 2016[^B124]).


Learning-based approaches tend to report more robust and generalizable performance, particularly in challenging or noisy datasets. Hidden Markov models (I-HMM) achieve balanced results across stimulus types, with Kappa values close to 0.7 for saccades (Andersson et al., 2016[^B5]). The two-means clustering method (I2MC), developed specifically for noisy infant recordings, reports an average F1-score of 0.83 across seven independent datasets, consistently outperforming several threshold-based methods (Hessels et al., 2017[^B84]). Random forest classifiers (I-RF) have achieved state-of-the-art sample-level results, with F1-scores near 0.97 and Kappa values around 0.85 in validation data, though performance decreases to about 0.70 on independent test sets (Zemblys et al., 2018[^B230]).


In summary, threshold-based methods are attractive for their simplicity and efficiency and remain effective under controlled static conditions, but they degrade substantially in noisy or dynamic environments. Learning-based methods demonstrate greater resilience, adaptability, and the ability to model complex data patterns, although they require annotated training datasets and greater computational resources. It is important to emphasize that these are reported performances drawn from heterogeneous studies, and differences in dataset characteristics, sampling frequency, and evaluation protocols likely account for a substantial part of the observed variability across algorithms.




### Separating smooth pursuits from fixations and saccades

The detection of smooth pursuit events, characterized by low-velocity, consistent-directionality eye movements that track moving targets, has received less attention compared to saccade and fixation classification. This task, known as *ternary segmentation*—classifying fixations, saccades, and smooth pursuits—is illustrated in , which depicts smooth pursuits—marked in purple—alongside fixations and saccades in high-quality eye-tracking data. Methods for identifying smooth pursuits are broadly categorized into threshold-based and learning-based approaches. Both approaches encounter the same limitations outlined in , including sensitivity to predefined thresholds in threshold-based methods and reliance on annotated training datasets in learning-based methods, which can be labor-intensive and specific to the dataset. Smooth pursuit detection is particularly challenging in noisy or low-quality data—*e.g.*, from low-frequency eye trackers or studies involving infants—often necessitating preprocessing steps such as noise filtering or blink removal to improve accuracy.

<figure id="F2" position="float">

<img src="/assets/fphys-16-1661026-g002.webp"/><figcaption>
<p>Ternary Segmentation. This example illustrates an oculomotor recording comprising fixations, saccades, and smooth pursuits. Panel <bold>(a)</bold>shows the two-dimensional gaze trajectory, where fixations are marked in purple, saccades in gray, and smooth pursuits in blue. Panels <bold>(b,c)</bold>display the corresponding horizontal and vertical gaze positions over time, highlighting the gradual directional displacements characteristic of smooth pursuit movements. These distinguishing features are the focus of <italic>ternary segmentation algorithms</italic>, which aim to isolate pursuit sequences from other phases.</p>
</figcaption>

</figure>

#### Threshold-based algorithms

Typically, a simple velocity threshold is first applied to isolate saccadic events, followed by a second step to distinguish between the remaining movements, namely, *fixation* and *pursuit* events. A straightforward but effective method for this task, known as the I-VVT approach, was proposed by Komogortsev and Karpov (2013)[^B105]. This method builds upon the I-VT algorithm by introducing a second velocity threshold to specifically isolate fixation events. Any remaining data points are then classified as pursuit events. However, a potential limitation of this approach is that eye movement velocities can vary between individuals and even within the same individual depending on the specific task being performed. As such, establishing universally effective thresholds to differentiate smooth pursuits from fixations—both of which are low-velocity movements—presents a challenge. This variability can complicate the application of this algorithm in real-world scenarios, particularly those involving dynamic scenes (Kasneci et al., 2015[^B97]).


To reduce reliance on velocity thresholds, Komogortsev and Karpov (2013)[^B105] proposed to distinguish between pursuit and fixation movements using a dispersion threshold combined with a temporal window—an approach commonly referred to as I-VDT. This method naturally extends the I-DiT approach by isolating fixation samples based on their spatial proximity. Similarly, Lopez (2009)[^B130] proposed an alternative strategy where the standard deviation of movement direction within a time window is used to differentiate between fixation and pursuit events. This approach provides an additional method for segmentation that focuses on directional variability rather than relying solely on velocity-based thresholds.


The Velocity and Movement Pattern Identification (I-VMP) algorithm, proposed by Lopez (2009)[^B130], provides an advanced method for detecting smooth pursuits in eye-tracking data. I-VMP employs a two-stage approach: it first applies a velocity threshold to isolate saccades, then analyzes the angular displacement between consecutive gaze points to identify smooth pursuits among low-velocity movements. Specifically, the angle between the horizontal axis and the line connecting successive gaze points is projected onto a unit circle, and a Rayleigh score is computed to quantify directional consistency within a defined temporal window. High Rayleigh scores indicate stable directionality, characteristic of smooth pursuits, distinguishing them from fixations, which exhibit random or minimal directional changes. While this method reduces dependence on velocity thresholds compared to traditional approaches, it requires preprocessing steps, such as noise filtering and blink removal, and knowledge of stimulus motion for optimal performance.


Finally, Santini et al. (2016)[^B177] introduced a Bayesian decision theory-based approach (I-BDT), specifically designed for the classification of smooth pursuit eye movements when viewing dynamic stimuli. Unlike earlier methods that rely on a velocity-based initial step to isolate non-saccadic sequences, this approach directly separates smooth pursuits from saccades and fixations without the need for an initial velocity threshold. Grounded in physiological hypotheses, the I-BDT approach incorporates explicit formulas to compute the likelihoods and priors for each type of eye movement—fixation, saccade, and smooth pursuit. These formulas enable the efficient classification of eye movement events by applying Bayes’ theorem, offering a probabilistic framework for distinguishing between different types of oculomotor behavior.



#### Learning-based algorithms


Fuhl et al. (2018)[^B62] introduced the Histogram of Oriented Velocities (I-HOV) method, which adapts a computer vision technique to classify fixations, saccades, and smooth pursuits in eye-tracking data. The I-HOV algorithm computes velocity-weighted angles between a gaze point and its predecessors or successors within a defined temporal window, generating a histogram that serves as a meta-representation of local gaze behavior for each sample. These histograms are used as feature vectors for machine learning algorithms, such as random forests, k-nearest neighbors, and support vector machines, to classify eye movement types. Similar to the I-VMP algorithm (Lopez, 2009[^B130]), I-HOV leverages the consistent directionality and low-velocity profiles of smooth pursuits to distinguish them from fixations and saccades. While effective for ternary segmentation, I-HOV relies on high-quality annotated training data and is computationally intensive. Its performance is also sensitive to noise and the limitations of low-frequency eye trackers, which may reduce the accuracy of velocity and angle calculations.


Recent advances in eye movement classification have leveraged deep learning techniques to distinguish smooth pursuit sequences from fixations and saccades. One such approach, proposed by Hoppe and Bulling (2016)[^B87], employs a convolutional neural network (CNN) combined with data windowing. In this method, gaze points within each temporal window are transformed into the frequency domain using a Fourier transform and then input to the CNN, which classifies the eye movement type. Similarly, Fuhl et al. (2021)[^B63] introduced a CNN-based method, termed I-CNN, that operates directly on windowed raw eye data to isolate oculomotor events. These deep learning approaches demonstrate significant effectiveness, particularly when trained on datasets tailored to specific experimental conditions and eye-tracking devices, underscoring their potential for robust eye movement classification. However, their performance remains heavily dependent on the quality and annotation of training data, which can substantially impact model accuracy and generalizability.


Ternary segmentation, tasked with classifying fixations, saccades, and smooth pursuits, presents greater challenges than binary segmentation due to the subtle low-velocity characteristics of smooth pursuits. Insights from Komogortsev and Karpov (2013)[^B105], Santini et al. (2016)[^B177], Fuhl et al. (2018)[^B62], and Fuhl et al. (2021)[^B63], evaluated on varied datasets with dynamic stimuli, provide a foundation for assessing performance, although quantitative benchmarks remain less comprehensive than for binary segmentation. Moreover, the different evaluations were conducted on distinct datasets, making it challenging to provide a reliable comparative analysis of the various segmentation methods. As such, the following paragraphs will focus on qualitative considerations.


Among threshold-based approaches, extensions of velocity- and dispersion-threshold methods—*e.g.*, I-VVT, I-VDT—have been applied to pursuits, while variants such as I-VMP incorporate directional information to reduce velocity ambiguities. Bayesian decision theory (I-BDT) has been reported to outperform dispersion-based methods (I-VDT) on several dynamic datasets at <inline-formula id="inf7">
<mml:math id="m7">
<mml:mrow>
<mml:mn>30</mml:mn>
<mml:mspace width="0.3333em"/>
<mml:mi>H</mml:mi>
<mml:mi>z</mml:mi>
</mml:mrow>
</mml:math>
</inline-formula>, leveraging priors to enhance pursuit detection (Santini et al., 2016[^B177]). Learning-based methods show greater adaptability. Histogram-based classification (I-HOV) and convolutional neural networks (I-CNN) have been reported to provide robust detection of pursuits in noisy or low-resolution dynamic data, outperforming threshold-based methods in these contexts (Fuhl et al., 2018[^B62]; 2021[^B63]).


In summary, ternary segmentation highlights the intrinsic difficulty of reliably detecting smooth pursuits, particularly at low velocities where they overlap with fixations. Threshold-based methods capture faster pursuits but remain sensitive to noise and sampling rate. Bayesian and direction-based extensions have been reported to reduce some of these ambiguities, though results vary across datasets. Learning-based methods appear more promising for handling complex or noisy recordings, especially with CNNs and histogram-based approaches, yet their effectiveness still depends on the availability of well-annotated training corpora. Reported performances point to relative strengths of each family of methods, but the absence of standardized benchmarks makes it difficult to establish a consensus hierarchy of algorithms.




## Physiological features

Applying the segmentation algorithms presented in  produces a sequence of fixations, saccades, and possibly smooth pursuits from raw gaze data. The following sections will review the most common metrics found in the literature to describe and analyze these oculomotor events.


The fundamental features and metrics for fixations, saccades, and smooth pursuits are summarized in –, respectively. The tables provide a concise description of each feature and references from the literature that offer guidance for their implementation.


| Feature name       | Description                                                                                                                                                                              | References              |
|:-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------|
| Count              | Given a set of fixation sequences, computes the number of fixations                                                                                                                      | Rigas et al. (2018)     |
| Frequency          | Given a set of fixation sequences, computes the number of fixations occurring per second                                                                                                 | Rigas et al. (2018)     |
| Duration           | Given a fixation sequence, computes the duration of the sequence                                                                                                                         | Rigas et al. (2018)     |
| First duration     | Given a set of fixation sequences, computes the duration of the first fixation sequence identified                                                                                       | Inhoff et al. (2000)    |
| Centroid           | Given a fixation sequence, computes centroid position by averaging coordinates of data samples                                                                                           | Rigas et al. (2018)     |
| Drift displacement | Given a fixation sequence, computes the distance between the starting and ending points of the sequence                                                                                  | Rigas et al. (2018)     |
| Drift distance     | Given a fixation sequence, computes the sum of distances between each data sample within this sequence                                                                                   | Rigas et al. (2018)     |
| Mean velocity      | Given a fixation sequence, computes the mean velocity of data sample within this sequence                                                                                                | Rigas et al. (2018)     |
| Drift velocity     | Given a fixation sequence, computes the drift displacement normalized by the fixation duration                                                                                           | Rigas et al. (2018)     |
| BCEA               | Given a fixation sequence, computes the bivariate contour ellipse area (BCEA) as the area of the elliptical contour that encompasses a given percentage of sample points of the sequence | Crossland et al. (2004) |


| Feature name                     | Description                                                                                                                                                                                | References                  |
|:---------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------|
| Duration                         | Given a saccade sequence, computes the duration of the sequence                                                                                                                            | Rigas et al. (2018)         |
| Frequency                        | Given a set of saccade sequences, computes the number of saccades occurring per second                                                                                                     | Rigas et al. (2018)         |
| Amplitude                        | Given a saccade sequence, computes the distance between the starting and ending points of the sequence                                                                                     | Rigas et al. (2018)         |
| Travel distance                  | Given a saccade sequence, computes the sum of distances between each data sample of the sequence                                                                                           | Rigas et al. (2018)         |
| Efficiency                       | Given a saccade sequence, computes the ratio of saccadic amplitude over the distance traveled                                                                                              | Rigas et al. (2018)         |
| Direction                        | Given a saccade sequence, computes the deviation from the horizontal plane of the line connecting the start and end points of the sequence                                                 | Foulsham et al. (2008)      |
| Successive deviation             | Given a set of saccade sequences, computes the angle formed by successive saccadic trajectories, where each saccade is modeled as a vector connecting its start and end points             | Foulsham et al. (2008)      |
| Initial direction                | Given a saccade sequence, computes the initial direction of the saccadic trajectory after a fixed number of data measures                                                                  | Ludwig and Gilchrist (2002) |
| Initial deviation                | Given a saccade sequence, computes the angle between the overall direction determined at the endpoint of the saccade, and the initial direction after a fixed number of data measures      | Ludwig and Gilchrist (2002) |
| Maximum curvature                | Given a saccade sequence, computes the maximum perpendicular distance from any point along the saccadic trajectory to the straight line connecting the start and end points of the saccade | Ludwig and Gilchrist (2002) |
| Area curvature                   | Given a saccade sequence, computes the area under the curve of the sampled saccadic trajectory, relative to the straight-line distance between the saccade starting and ending points      | Ludwig and Gilchrist (2002) |
| Mean velocity                    | Given a saccade sequence, computes the mean velocity of data samples within the sequence                                                                                                   | Rigas et al. (2018)         |
| Peak velocity                    | Given a saccade sequence, computes the peak velocity of data samples belonging to the sequence                                                                                             | Rigas et al. (2018)         |
| Acceleration profile             | Given a saccade sequence, computes the mean acceleration of data sample within the sequence                                                                                                | Rigas et al. (2018)         |
| Mean acceleration                | Given a saccade sequence, computes the mean absolute acceleration during the acceleration phase of the saccade, measured from the start point to the timestamp of peak acceleration        | Rigas et al. (2018)         |
| Skewness exponent                | Given a saccade sequence, computes the shape parameter obtained by fitting a gamma function to the sequence velocity profile                                                               | Chen et al. (2002)          |
| Amplitude to duration ratio      | Given a saccade sequence, computes the sequence amplitude over duration ratio                                                                                                              | Rigas et al. (2018)         |
| Peak velocity to amplitude ratio | Given a saccade sequence, computes the sequence peak velocity over amplitude ratio                                                                                                         | Rigas et al. (2018)         |
| Peak velocity to duration ratio  | Given a saccade sequence, computes the sequence peak velocity over duration ratio                                                                                                          | Rigas et al. (2018)         |
| Peak velocity to velocity ratio  | Given a saccade sequence, computes the sequence peak velocity over mean velocity ratio                                                                                                     | Rigas et al. (2018)         |
| Main sequence                    | Given a set of saccade sequences, computes slopes of the amplitude/duration curve and the log peak velocity/log amplitude curve                                                            | Bahill et al. (1975)        |
| Latency                          | Given a saccade sequence and a theoretical trajectory, computes the time difference between the onset of the theoretical saccade and the start time of the corresponding saccade           | Whelan (2008)               |
| Latency quantiles                | Given a set of saccade sequences and a theoretical trajectory, computes the set of saccade latencies, before evaluating quantiles of the latency distribution                              | Vullings (2018)             |
| Gain                             | Given a saccade sequence and a theoretical trajectory, computes the ratio between saccade and target amplitudes                                                                            | Holmqvist et al. (2011)     |


| Feature name            | Description                                                                                                                                                                                                                                                                 | References                     |
|:------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|
| Duration                | Given a pursuit sequence, computes the duration of the sequence                                                                                                                                                                                                             | Murray et al. (2020)           |
| Frequency               | Given a set of pursuit sequences, computes the number of pursuits occurring per second                                                                                                                                                                                      | Murray et al. (2020)           |
| Amplitude               | Given a pursuit sequence, computes the distance between the starting and ending points of the sequence                                                                                                                                                                      | Mahanama et al. (2022a)        |
| Direction               | Given a pursuit sequence, computes the deviation from the horizontal plane of the line connecting the start and end points of the sequence                                                                                                                                  | Rottach et al. (1996)          |
| Mean velocity           | Given a pursuit sequence, computes the mean velocity of data sample within the sequence                                                                                                                                                                                     | Mahanama et al. (2022b)        |
| Peak velocity           | Given a pursuit sequence, computes the peak velocity of data samples                                                                                                                                                                                                        | Mahanama et al. (2022b)        |
| Latency                 | Given a pursuit sequence and a theoretical trajectory, computes the time difference between the onset of the theoretical smooth pursuit and the start time of the corresponding experimental pursuit                                                                        | Carl and Gellman (1987)        |
| Initial acceleration    | Given a pursuit sequence and a theoretical trajectory, computes the mean second-order position derivative of the sequence in a time interval immediately following pursuit onset                                                                                            | Kao and Morrow (1994)          |
| Triangular overall gain | Given a pursuit sequence and a triangular theoretical trajectory, computes the ratio between pursuit sequence and target mean velocities                                                                                                                                    | Rashbass (1961)                |
| Sinusoidal overall gain | Given a pursuit sequence and a sinusoidal theoretical trajectory, computes the ratio between pursuit sequence and target mean velocities                                                                                                                                    | O’Driscoll and Callahan (2008) |
| Sinusoidal gain         | Given a pursuit sequence and a theoretical trajectory, fits the eye velocity with a trigonometrical curve, before computing the ratio between the peak velocity of the best fitting curve over the target’s peak velocity                                                   | Accardo et al. (1995)          |
| Sinusoidal phase        | Given a pursuit sequence and a theoretical trajectory, computes the difference between the phases of the best-fitting velocity curve and the target’s velocity profile                                                                                                      | Accardo et al. (1995)          |
| Error entropy           | Given a pursuit sequence and a theoretical trajectory, computes the pursuit velocity error series as the difference between the experimental pursuit velocities and theoretical stimulus velocities, before evaluating the approximate entropy of the velocity error series | Pincus et al. (1991)           |
| Cross-correlation       | Given a pursuit sequence and a theoretical trajectory, computes normalized cross-correlation between the experimental pursuit velocity and theoretical stimulus velocity signals                                                                                            | Rabiner (1978)                 |


### Fixation measures

A fixation is defined as a period during which the gaze is stabilized on a specific spatial location, projecting visual stimuli onto the *fovea centralis*, the retinal region with maximal photoreceptor density and visual acuity. Despite attempts to maintain steady fixation on a stationary target, the eyes exhibit continuous, involuntary micromovements, including microsaccades—rapid, small-amplitude saccades—drifts—slow, curvilinear deviations—and tremors—high-frequency, low-amplitude oscillations. This section examines the quantitative features characterizing fixations, including temporal, positional attributes, and dynamic characteristics. These properties are typically analyzed under head-constrained conditions using high-resolution eye-tracking systems to isolate oculomotor behavior.


#### Temporal features


*Fixation count* is defined as the total number of fixations within a defined time interval or stimulus region. Despite its simplicity, the fixation count remains a cornerstone metric in eye-tracking research due to its robustness and interpretability. It is frequently employed in exploratory analyses before applying more advanced techniques. Fixation count is widely utilized to assess visual attention allocation to regions of interest (ROIs) in textual or pictorial stimuli (Scheiter and Eitel, 2017[^B179]), infer the depth and efficiency of visual processing (Jacob and Karn, 2003[^B93]; Park et al., 2015[^B157]), and investigate how expertise influences oculomotor behavior in visual tasks (Schoonahd et al., 1973[^B182]; Megaw and Richardson, 1979[^B140]).


Pioneering work by Goldberg and Kotval (1999)[^B69] highlighted that a higher number of fixations directed at a stimulus often indicates inefficiency in the search for relevant information. As such, fixation count has been used in eye-tracking studies to identify visual regions that attract more attention or to infer the amount of cognitive effort required for a particular task. For example, in challenging tasks such as source code reading, a higher fixation count could signify increased visual effort and processing time (Binkley et al., 2013[^B16]; Sharif et al., 2012[^B188]). The *fixation count* is often expressed per unit of time or relative to a specific task or sub-task. For example, in reading tasks, the *fixation count* can be normalized to the length of the text by dividing the number of fixations by the number of words (Sharafi et al., 2015[^B187]).


Another critical metric, *fixation duration*, quantifies the temporal dynamics of gaze behavior. Typical fixations last between 200 and 300 milliseconds; however, longer durations on a stimulus may indicate greater processing complexity (Jacob and Karn, 2003[^B93]; Krejtz et al., 2016b[^B113]; Liu and Chuang, 2011[^B127]). This metric is frequently employed in eye-tracking studies to examine complex cognitive functions such as reading comprehension (Raney et al., 2014[^B163]), learning processes (Liu, 2014[^B126]), and mental workload assessment (Liu et al., 2022[^B128]). Furthermore, individual fixation durations may be analyzed independently. A notable example is the *first fixation duration* during reading, which is a commonly reported linguistic measure used to assess initial processing of a word or phrase (Inhoff et al., 2000[^B92]; Underwood et al., 2000[^B209]).


The temporal characteristics of eye fixations are often analyzed in relation to specific regions within the visual field that are visually explored. These *areas of interest* (AoI), may represent regions particularly relevant to the task at hand, or with semantical meaning. Under this formalism, fixation duration metrics are also used, albeit with slight variations. For instance, the *dwell time* is defined as the cumulative duration of all fixations during a single visit to an AoI. The *total dwell time* sums all *dwell time* within a specific AoI over the entire experimental session. Additional AoI-specific metrics offer further granularity, such as the *fixation ratio*, defined as the sum of fixation durations within an AoI divided by the total fixation duration across all AoIs, or the *average fixation duration* within an AoI, derived by normalizing the sum of fixation durations by the number of fixations in that AoI. The concept of AoI as a symbolic tool will be explored in greater detail in the *Areas of Interest* part of this review series (Part 4).



#### Position and drift

The location of visual fixations is widely studied across various contexts, as it is often assumed to reflect the allocation of visual attention (Findlay and Gilchrist, 2003[^B54]). A robust method for modeling the central position of fixations is the fixation centroid, calculated by averaging the coordinates of gaze points within individual fixation sequences. Analyzing the spatial distribution of these centroids provides valuable insights into the regions of a stimulus that are prioritized during task-specific processing, offering direct evidence of underlying cognitive processes (Henderson, 2003[^B82]; Rayner, 1998[^B165]).


For instance, in studies related to face processing, analyses of fixation patterns have identified specific gaze patterns, such as directing attention to a point just below the eyes (Hsiao and Cottrell, 2008[^B88]; Peterson and Eckstein, 2012[^B158]). Similarly, in reading tasks, research has shown that both the likelihood of misidentifying a word and the time required for identification decrease when the eyes fixate near the center of the word (O’Regan and Jacobs, 1992[^B156]; Brysbaert et al., 1996[^B23]). These phenomena, known as *optimal viewing position* effects, are thought to stem from the rapid decline in visual acuity as retinal eccentricity increases (Nazir et al., 1998[^B152]).


While fixational sequences typically exhibit limited eye mobility, the variability in the micro-movements can provide valuable information related to oculomotor function. Consequently, several additional features—many of which are illustrated in  — have been proposed in the literature to better characterize fixational micro-movements.

<figure id="F3" position="float">

<img src="/assets/fphys-16-1661026-g003.webp"/><figcaption>
<p>Fixation Drift and Stability. An example of gaze data—black crosses—representing a fixation sequence is shown. Note that the raw data have been largely downsampled for presentation clarity. In this illustration, the drift displacement between the starting and ending points of the fixation sequence is denoted as <inline-formula id="inf8">
<mml:math id="m8">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>d</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>0</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>. The cumulative drift distance is computed by summing the distances <inline-formula id="inf9">
<mml:math id="m9">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>d</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>1</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula> to <inline-formula id="inf10">
<mml:math id="m10">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>d</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>24</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>. Additionally, the figure displays the bivariate contour ellipses for probabilities of 0.68 — blue dashed line—and 0.90 —blue dotted line. The areas enclosed by these ellipses are used to compute the BCEA, a commonly used metric for fixation stability.</p>
</figcaption>

</figure>

As such, the *drift displacement* is calculated as the distance between the starting and ending points of each fixation sequence. Similarly, the *cumulative drift distance*, which reflects ocular stability during fixation, is obtained by summing the distances between all consecutive fixational data samples from a given fixation sequence. Another feature, the *drift mean velocity*, is computed as the average of the first-order position derivatives of the fixation data samples and can be used to characterize the minor movements occurring during fixation sequences. Together, these measures can provide valuable insights into the stability of eye movements during fixation, which may be particularly useful for detecting pathological conditions, such as sight impairments and cerebellar diseases (Leech et al., 1977[^B120]; Schor and Westall, 1984[^B183]).


Lastly, fixation stability can be quantified by computing the area of the elliptical contour that encompasses a given percentage of fixation points (Steinman, 1965[^B202]; Crossland et al., 2004[^B36]). Assuming that the fixation positions follow a bivariate normal distribution, the dispersion of these positions is represented by an ellipse. The *bivariate contour ellipse area* (BCEA) thus provides a measure of fixation stability, with smaller values indicating more stable fixation. This metric is considered the current *gold standard* to measure the stability of fixation (Crossland et al., 2009[^B37]) and has been widely used to examine changes in fixational eye movements, particularly in clinical contexts (Shaikh et al., 2016[^B186]; Montesano et al., 2018[^B145]; Leonard et al., 2021[^B123]; Ghasia and Wang, 2022[^B67]).




### Saccade measures

Saccades are rapid, ballistic eye movements that direct the *fovea* toward objects of interest, enabling high-acuity vision. Since the inception of eye movement research, the kinematic properties—*e.g.*, velocity, amplitude—and shape characteristics—*e.g.*, trajectory, curvature—of saccadic eye movements have been extensively studied using diverse measurement techniques, which we will now review and discuss.


In experimental settings, saccadic behavior is investigated using paradigms involving both predictable and unpredictable target conditions. The metrics presented in the following sections are designed to quantify the dynamics of saccadic eye movements in these two conditions, that is free-viewing scenarios and those involving target-based stimuli. These metrics offer critical insights into saccade dynamics and their modulation by experimental manipulations.


#### Temporal features


*Saccade duration* is a commonly analyzed metric in eye movement research, with typical values ranging from 30 to 70 milliseconds. While these values may vary slightly across studies, various factors have been identified in the literature as influencing saccade duration. For example, during coordinated reaching movements, saccades that accompany hand motions tend to have shorter durations (Donkelaar et al., 2004[^B43]; Snyder et al., 2002[^B196]). Conversely, repeated saccades to the same visual stimulus often result in longer durations (Golla et al., 2008[^B71]; Chen-Harris et al., 2008[^B30]). The measurement of *saccade duration* typically involves estimating the onset and offset of the saccade. Given the brief nature of saccadic movements, the accuracy of this measurement is highly sensitive to the thresholds applied to segment raw gaze data—see .


In addition to duration, *saccade count* and *saccade rate*—or *saccade frequency*—are widely used metrics to characterize saccadic sequences. Generally, *saccade frequency* tends to decrease with increasing task difficulty (Nakayama et al., 2002[^B151]) or under conditions of fatigue (Van Orden et al., 2000[^B215]). Like *saccade duration*, *saccade count* is a simple and robust measure commonly employed in studies that investigate cognitive processes such as reading or scene perception (Inhoff and Radach, 1998[^B91]). Furthermore, deviations from typical saccadic temporal characteristics, such as prolonged *saccade duration*, can serve as early indicators of neural disorders (Ramat et al., 2007[^B162]).


In experimental paradigms that involve target-directed saccades, the temporal aspect of saccadic movements is frequently examined using *saccadic latency*, which is the time delay between stimulus onset and saccade initiation. For any given target, while saccade duration, velocity, and amplitude tend to remain relatively consistent, latency is notably variable across trials, ranging from 100 to 1,000 milliseconds (Liversedge et al., 2011[^B129]). The distribution of *saccadic latency* is generally skewed toward shorter latencies, with a long tail representing longer latencies. Additionally, the distribution is often unimodal, although a second peak—referred to as *express saccades*—can sometimes appear, representing shorter saccadic responses (Fischer and Weber, 1993[^B55]).


The mean *saccade latency* is typically used to describe the central tendency of reaction times, while the standard deviation is used to assess variability (Whelan, 2008[^B224]). However, since the latency distribution is not Gaussian, these statistics may not fully capture the nature of the distribution. As a result, more robust statistical measures, such as the median or quantile estimators, are increasingly adopted to describe saccadic latency distributions more accurately (Vullings, 2018[^B219]). In clinical contexts, saccadic latency distributions have shown promise as biomarkers for various neurological conditions. For instance, Michell et al. (2006)[^B143] demonstrated that saccadic latency could be used as a diagnostic marker for Parkinson’s disease, highlighting its potential utility in clinical assessments of cognitive and motor dysfunctions.



#### Amplitude features

Describing saccade morphology is essential for a comprehensive understanding of eye movement dynamics. Among the various morphological features, *saccade amplitude* serves as a fundamental and easily accessible descriptor that reflects the distance the eye travels during a saccadic movement. It is typically calculated as the spatial distance between the starting and ending points of each identified saccade sequence. Alternatively, to model the non-linearity of saccade trajectory, the *traveled distance* can be computed by summing the distances between consecutive saccadic data samples within a saccade sequence. Lastly, *saccade efficiency*, derived as the ratio of saccadic amplitude to the total distance traveled, is often used to quantify the complexity and non-linearity of the saccadic trajectory. This metric provides insight into the degree to which the eye movement follows a straight path versus a more convoluted or inefficient trajectory.



*Saccade amplitude* is highly context-dependent, varying according to the task and visual environment. For example, in reading tasks, saccades are typically constrained to around 2 degrees of visual angle horizontally (Rayner et al., 2012[^B167]). In contrast, during scene perception, the average *saccade amplitude* increases with the size of the visual stimulus, reflecting the broader spatial search required to process larger or more complex images (von Wartburg et al., 2007[^B218]). Cognitive factors also influence *saccade amplitude*, with increases in task difficulty often leading to a decrease in the amplitude of saccadic movements. Phillips and Edelman (2008)[^B159] demonstrated that variability in performance during visual scanning tasks was related to oculomotor variables such as amplitude, with smaller saccades indicating a reduced perceptual span. Similarly, May et al. (1990)[^B137] provided evidence that this metric could serve as an indicator of cognitive workload, with smaller amplitudes reflecting greater cognitive demands. It should also be mentioned that *saccade amplitude* is closely related to its duration and peak velocity through the *main sequence* relationship—see  for further details. These oculomotor characteristics—amplitude, duration, and peak velocity—are often analyzed together as they provide complementary insights into the saccadic process.


When viewers are instructed to follow a visual target, the *saccadic gain*—the ratio between the amplitude of the saccade performed and the amplitude of the target displacement—becomes a critical measure. *Saccadic gain* is particularly useful in assessing saccadic dysmetria, a condition characterized by errors in saccade accuracy. In neurological studies, saccadic dysmetria is often investigated to identify impairments in saccadic control. For instance, in overshoot dysmetria, the saccade initially overshoots the target, requiring a corrective saccade in the opposite direction. While overshoots can occur in healthy individuals, they typically reduce over time as the oculomotor system adjusts to the target location. Persistent overshooting, however, is indicative of a cerebellar lesion (Selhorst et al., 1976[^B185]; Ritchie, 1976[^B170]). Conversely, undershoot dysmetria occurs when the initial saccade is too small, and a corrective saccade is required to bring the eye to the target. Significant undershooting is often associated with basal ganglia disorders, such as Parkinson’s disease (MacAskill et al., 2002[^B133]) or progressive supranuclear palsy (Troost and Daroff, 1977[^B207]).


More intriguingly, saccadic dysmetria—particularly hypometric saccades—has been proposed as a potential objective biomarker for neurodegenerative diseases. Abnormally hypometric saccades, along with other eye movement deficits, have shown promise as early indicators of conditions like Alzheimer’s disease, making them valuable targets for early diagnosis (Fletcher and Sharpe, 1986[^B56]; Cerquera-Jaramillo et al., 2018[^B28]). This highlights the importance of saccade morphology not only for understanding normal visual behavior but also as a potential tool for identifying and monitoring the progression of neurological disorders.



#### Direction and curvature

The direction of a saccadic trajectory—or sequence of saccades—provides a crucial descriptive measure of eye movements. This direction is typically quantified as the angle, measured in degrees or radians, between the horizontal axis and the line connecting the starting and ending points of the saccade. For instance, Walker et al. (2006)[^B220] employed *saccadic direction* to examine the effects of target predictability, while Foulsham et al. (2008)[^B57] explored the *horizon bias* during natural scene viewing, revealing a prevalent tendency for horizontal saccades. More recently, studies have employed *saccadic direction* to classify task-specific gaze patterns, offering valuable insights for designing effective learning strategies (Mozaffari et al., 2020[^B148]).


However, simple metrics such as amplitude, efficiency—as discussed in  — and direction alone are insufficient for fully capturing the complexity and non-linearity of saccadic trajectories. To address this gap, several additional features have been developed to better characterize the curvature of saccadic movements (Ludwig and Gilchrist, 2002[^B131]).


One such metric is *initial deviation*, which measures the angle between the initial direction of the saccade—computed after a fixed number of time samples, *e.g.*, 20 milliseconds (Van Gisbergen et al., 1987[^B212]) — and the overall direction of the saccade at its endpoint. A limitation of this method is that it assigns varying curvature values to saccades with identical trajectories but different velocities, because it relies on a fixed time interval. Another common metric is *maximum curvature*, defined as the greatest perpendicular distance between a point on the saccadic trajectory and the straight line connecting the starting and ending points of the saccade (Smit and Van Gisbergen, 1990[^B194]). Although widely used, this approach has limitations, as it relies on a single point to represent the curvature of a trajectory. This can be especially problematic for double-curved saccades, where the trajectory may involve multiple directional changes (Ludwig and Gilchrist, 2002[^B131]).


To address these shortcomings, the *area curvature* metric has emerged as a more robust and popular approach, as it incorporates the entire trajectory of the saccadic eye movement (Walker et al., 2006[^B220]). This metric is typically calculated by evaluating the area beneath the curve formed by the sampled trajectory, relative to the direct distance between the starting and ending points of the saccade. The curvature metrics discussed so far are illustrated in . Additionally, Ludwig and Gilchrist (2002)[^B131] proposed deriving saccade curvature directly from second- and third-order polynomial fits. Like the *area curvature* approach, this method uses the full set of samples from a given saccade, which enhances its robustness by making it less sensitive to sampling noise.

<figure id="F4" position="float">

<img src="/assets/fphys-16-1661026-g004.webp"/><figcaption>
<p>Saccade Direction and Curvature. Illustration of various metrics used to describe saccade non-linearity in the literature. The line connecting the starting point and the endpoint of the saccade, with amplitude <inline-formula id="inf11">
<mml:math id="m11">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>d</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>1</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, defines the overall saccade direction, denoted as <inline-formula id="inf12">
<mml:math id="m12">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>θ</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>1</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>. The initial direction of the saccade, denoted <inline-formula id="inf13">
<mml:math id="m13">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>θ</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>2</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, is calculated after a fixed number of data points. From these two directions, the initial deviation of the saccade, denoted <inline-formula id="inf14">
<mml:math id="m14">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>θ</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>3</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, can be derived. Additionally, the figure highlights the maximum curvature, represented by <inline-formula id="inf15">
<mml:math id="m15">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>d</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>2</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, and the area of curvature, indicated by the purple shaded region.</p>
</figcaption>

</figure>

To investigate the inherent tendency for curvature observed in saccadic movements—particularly prominent in oblique saccades (Viviani and Swensson, 1982[^B217]) — early research primarily focused on target location and the type of saccade being performed (Viviani, 1977[^B216]; Smit and Van Gisbergen, 1990[^B194]). More recent studies, however, have shown that both the direction and magnitude of saccadic curvature can be modulated by a variety of factors. Notably, strong correlations have been observed between saccade curvature and the modulation of eye movements by distractors. For example, Doyle and Walker (2001)[^B44] found that both reflexive and voluntary saccades tended to curve away from irrelevant distractor stimuli when a target was presented. Similarly, Sheliga et al. (1997)[^B190], Sheliga et al. (1995)[^B189] demonstrated that saccades deviated from a previously attended location. These variations in saccadic trajectory have been attributed to antagonistic interactions between different populations of neurons in the superior colliculus, which help resolve conflicts caused by competing targets in the vicinity at the onset of movement (McPeek et al., 2003[^B139]).



#### Velocity features

The velocity waveform of a saccade is generally described as symmetrical with comparable durations for the acceleration and deceleration phases—. Peak saccadic velocity, the maximum speed attained during a saccade, typically coincides with the cessation of the neural signal pulse and aligns with the point of maximum firing rate of burst neurons within the pontine reticular formation that project to oculomotor neurons (Galley, 1989[^B65]; Leigh and Zee, 2015[^B121]). It is noteworthy that average and peak saccadic velocities are frequently analyzed together due to their strong correlation. Their absolute values generally exhibit a consistent ratio of approximately <inline-formula id="inf16">
<mml:math id="m16">
<mml:mrow>
<mml:mn>1</mml:mn>
<mml:mo>:</mml:mo>
<mml:mn>2</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula>, a relationship commonly referred to as the *Q ratio*. This ratio remains relatively stable across various saccadic amplitudes, underscoring its reliability as a metric for characterizing saccadic dynamics (Harwood et al., 1999[^B78]; Garbutt et al., 2003[^B66]).

<figure id="F5" position="float">

<img src="/assets/fphys-16-1661026-g005.webp"/><figcaption>
<p>Saccade Velocity and Acceleration Profiles. Examples of saccade velocity and acceleration profiles for short — <bold>(a)</bold> — and long — <bold>(b)</bold> —- saccades, illustrating differences in peak values and overall shapes. For both types of saccades, the peak velocity is denoted as <inline-formula id="inf17">
<mml:math id="m17">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>v</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>1</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, the peak acceleration as <inline-formula id="inf18">
<mml:math id="m18">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>a</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>1</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, and the peak deceleration as <inline-formula id="inf19">
<mml:math id="m19">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>a</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>2</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>. Additionally, the duration of the acceleration phase is represented by <inline-formula id="inf20">
<mml:math id="m20">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>t</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>1</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>, while the duration of the deceleration phase is denoted by <inline-formula id="inf21">
<mml:math id="m21">
<mml:mrow>
<mml:msub>
<mml:mrow>
<mml:mi>t</mml:mi>
</mml:mrow>
<mml:mrow>
<mml:mn>2</mml:mn>
</mml:mrow>
</mml:msub>
</mml:mrow>
</mml:math>
</inline-formula>.</p>
</figcaption>

</figure>

More specifically, *saccade mean velocity* is regarded as a reliable metric for assessing the velocity of small saccades, particularly those with symmetrical velocity waveforms. The properties of saccadic velocity have been thoroughly investigated across numerous fields and clinical applications (Di Stasi et al., 2013[^B40]). Early research observed that external factors such as alcohol, drugs, and fatigue lead to reductions in saccadic velocity (Dodge and Benedict, 1915[^B41]; Miles, 1929[^B144]), a phenomenon attributed to diminished central nervous system activation. More recently, studies have highlighted saccadic velocity as a marker for fluctuations in sympathetic nervous system activity (Di Stasi et al., 2013[^B40]), variations in the intrinsic value of visual stimuli (Xu-Wilson et al., 2009[^B226]), and the effects of task experience on oculomotor control (Xu-McGregor and Stern, 1996[^B225]). Clinically, abnormally low saccadic velocities—commonly termed *slow saccades*—are symptomatic of midbrain disorders such as progressive supranuclear palsy, spinocerebellar ataxia type 2, and various cerebellar pathologies (Jensen et al., 2019[^B94]).


While mean velocity provides a useful summary metric, it becomes less effective for saccades larger than 10°, which often exhibit asymmetric velocity profiles—. For such larger saccades, *saccade peak velocity* is typically preferred as it reflects the highest firing rates of burst neurons driving the movement (Galley, 1989[^B65]). Unlike mean velocity, peak velocity has computational advantages: it remains consistent regardless of segmentation thresholds—see  for further details—making it robust to variations in how sharply a saccade terminates during its final phase.


Several methodological considerations are important when calculating velocity features, particularly for saccades, though these principles extend to other canonical gaze movements as well. The simplest and most common method calculates velocity by applying a two-point central difference algorithm to the eye position signal (Schmidt et al., 1979[^B180]). However, this straightforward approach has significant drawbacks. First, the numerical derivative is inherently highly sensitive to noise. Depending on the specific eye-tracking device, characterizing and removing measurement noise can be challenging or even infeasible. While filtering techniques can mitigate noise, they may inadvertently alter velocity estimates, particularly the crucial peak velocity. Second, this method is strongly influenced by sampling frequency. Since *saccade peak velocity* typically occurs between recorded samples, devices with low sampling rates often underestimate this key measure.


To address these limitations, more sophisticated and robust methods have been developed. These include the eight-point central difference derivative algorithm (Inchingolo and Spanio, 1985[^B89]; Federighi et al., 2011[^B51]), which enhances noise resilience, as well as velocity profile fitting using gamma functions (Smit et al., 1987[^B195]), and saccade trajectory curve fitting using sigmoid functions (Gibaldi and Sabatini, 2021[^B68]), both of which provide refined estimates by leveraging model-based approaches. These advanced techniques are robust against noise and sampling artifacts, enabling accurate velocity estimation even when using low-cost, low-sampling-rate eye trackers. This compatibility with accessible technologies broadens the utility of such methods for a wide range of research and practical applications.



#### Acceleration features

To effectively quantify saccade acceleration characteristics, several metrics can be derived from the acceleration profile. As such, *saccade peak acceleration* is defined as the maximum absolute value of acceleration during the acceleration phase, which spans the interval from saccade onset to *saccade peak velocity*. Conversely, *saccade peak deceleration* represents the maximum absolute value of acceleration during the deceleration phase, occurring from peak velocity to saccade termination.


An additional metric of interest is the *acceleration/deceleration ratio*, computed as the ratio of the duration of the acceleration phase to that of the deceleration phase. This ratio reflects the skewness of the velocity profile. As expected, it tends to approximate one for small saccades but decreases as saccade amplitude increases. Finally, *saccade skewness* can be directly quantified through curve fitting, typically using a gamma function applied to the velocity profile. The resulting shape parameter provides a reliable estimate of skewness (Chen et al., 2002[^B29]).


As briefly discussed in , the acceleration and deceleration characteristics of saccades vary markedly with saccade amplitude. Specifically, larger saccades exhibit left-skewed velocity profiles, where the acceleration phase constitutes roughly one-third of the total saccade duration (Baloh et al., 1975[^B8]; Lin et al., 2004[^B125]). This asymmetry correlates strongly with both saccade amplitude and, even more so, its duration (Van Opstal and Van Gisbergen, 1987[^B214]). While the duration of the deceleration phase increases with saccade amplitude and duration, the duration of the acceleration phase remains relatively constant (Becker, 1991[^B10]).


The asymmetry in saccade velocity profiles, as well as its relationship with saccade duration, has been consistently observed and documented over several decades. However, the physiological significance and underlying mechanisms of this phenomenon remain unclear, with no definitive hypothesis currently available in the literature. Research suggests that saccade acceleration characteristics may be subject to modification through motor learning processes (Collins et al., 2008[^B35]). Furthermore, these characteristics have been linked to neurodevelopmental conditions, such as autism spectrum disorder, where abnormal acceleration and deceleration profiles have been observed (Schmitt et al., 2014[^B181]). These findings highlight the potential for saccade dynamics to serve as biomarkers for both cognitive and neurological assessments.



#### Saccadic ratios

Various ratios derived from saccadic characteristics have been extensively studied, revealing valuable insights into the interconnections between oculomotor mechanisms. For instance, Garbutt et al. (2003)[^B66] identified abnormally high *peak velocity-to-mean velocity* ratios in saccadic trajectories recorded from patients with progressive supranuclear palsy. This anomaly suggested that these movements might not be purely saccadic but rather comprise a sequence of small-amplitude saccades.


In healthy individuals, saccadic ratios have been shown to reflect low-level idiosyncrasies. For example, these ratios have been employed as biometric features for individual identification among other eye-movement metrics (Rigas and Komogortsev, 2016[^B168]). Extending this analysis to higher cognitive functions, Gupta and Routray (2012)[^B75] demonstrated a significant correlation between the *peak velocity-to-duration* ratio and human alertness, suggesting its utility for vigilance monitoring. These findings underscore the potential of saccadic ratios as versatile markers, ranging from physiological baselines to cognitive states.


Shifting focus to broader measures of eye movement dynamics, the *saccade-fixation* ratio, introduced by Goldberg and Kotval (1999)[^B69], highlights the balance between exploratory behavior—searching—and cognitive processing—information extraction. A higher value for this ratio reflects increased searching relative to processing. This metric has been used in comparative studies of different layouts or visual representations. Both the total *fixation-to-saccade duration* ratio and the average *fixation-to-saccade duration* ratio per occurrence can be derived from this measure. These simple yet powerful metrics have been employed in diverse experimental contexts to assess attention and cognitive information processing levels (Bhoir et al., 2015[^B13]; Berges et al., 2023[^B12]).


Finally, we mention the *K coefficient* introduced by Krejtz et al. (2016a)[^B112], Krejtz et al., (2017)[^B114]. This metric has emerged as an extension of the *saccade-fixation ratio* and is inherently linked to scanpath analysis. As such, it will be described in greater detail in the corresponding article of this review series.



#### Main sequence

The term *main sequence* describes a consistent relationship between three fundamental saccadic parameters: amplitude, duration, and velocity (Bahill et al., 1975[^B6]). Specifically, the relationship between saccadic peak velocity and amplitude demonstrates three key trends: <inline-formula id="inf22">
<mml:math id="m22">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> a roughly linear increase for small saccades—up to <inline-formula id="inf23">
<mml:math id="m23">
<mml:mrow>
<mml:mn>5</mml:mn>
<mml:mo>−</mml:mo>
<mml:mo>−</mml:mo>
<mml:mn>10</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula> degrees — <inline-formula id="inf24">
<mml:math id="m24">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> an inflection point between 10 and 20°, and <inline-formula id="inf25">
<mml:math id="m25">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> a plateau where peak velocity saturates for larger saccades (Gibaldi and Sabatini, 2021[^B68]). This stereotypical behavior is thought to result from an optimization process that improves visual performance amidst internal noise and peripheral visual uncertainty (Harris and Wolpert, 2006[^B77]; Saeb et al., 2011[^B175]; van Opstal and Goossens, 2008[^B213]). Additionally, the *main sequence* exhibits a linear relationship between saccade duration and amplitude for saccades up to approximately 80° (Baloh et al., 1975[^B8]), as shown in . However, most naturally occurring saccades are confined to a range of about 30° in the absence of head movement (Lebedev et al., 1996[^B119]).

<figure id="F6" position="float">

<img src="/assets/fphys-16-1661026-g006.webp"/><figcaption>
<p>Main Sequence. Main-sequence relationships for saccades, along with the respective linear regression fits, are shown for amplitude-duration <bold>(a)</bold> and the logarithms of peak velocity-amplitude <bold>(b)</bold>. Each colored dot represents a saccade from a set performed by the same individual during a reading task. The data emphasize the linear relationship between the logarithms of amplitude and peak velocity for saccades of moderate amplitude. While the amplitude-duration relationship is well-established in the literature, its experimental clarity appears to be less consistent.</p>
</figcaption>

</figure>

The *main sequence* is widely employed in clinical research as a diagnostic tool to evaluate the integrity of the saccadic system. Deviations from its expected patterns and abnormalities in saccadic behavior are indicative of various neurological and ocular conditions, including palsy of extraocular muscles (Metz et al., 1970[^B141]; Garbutt et al., 2003[^B66]), myasthenia gravis (Yee et al., 1976[^B228]), cerebellar disorders (Selhorst et al., 1976[^B185]), and multiple sclerosis (Frohman et al., 2002[^B60]; Bijvank et al., 2019[^B14]). Recent work by Guadron et al. (2023)[^B74] further highlighted the diagnostic relevance of the *main sequence* by examining patients with central and peripheral retinal defects. Their findings revealed that the characteristic relationships between saccadic parameters were most disrupted when targets were located within the subjects’ blind fields. This disruption underscores the critical role of visual input in planning saccadic kinematics, reinforcing the *main sequence* as a valuable lens through which the interplay between sensory input and motor control can be assessed.


Despite its widespread utility, there remains no universal consensus on the best mathematical model to describe the *main sequence*, particularly the non-linear relationship between peak velocity and saccade amplitude. Early work adopted power-law models to capture the non-linear growth of peak velocity with amplitude (Yarbus and Yarbus, 1967[^B227]; Baloh et al., 1975[^B8]; Lebedev et al., 1996[^B119]). These models have proven useful for detecting performance deficits in clinical settings (Garbutt et al., 2003[^B66]). For larger saccades, <inline-formula id="inf26">
<mml:math id="m26">
<mml:mrow>
<mml:mn>15</mml:mn>
<mml:mo>−</mml:mo>
<mml:mo>−</mml:mo>
<mml:mn>20</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula> degrees and beyond, where the maximum velocity saturates, exponential-based models have gained traction. First proposed by Bahill et al. (1975)[^B6], these models have been extensively utilized in both research and clinical diagnostics (Ramat et al., 2007[^B162]; Federighi et al., 2017[^B52]) and remain popular for their accuracy and applicability in recent studies (Leigh and Zee, 2015[^B121]). Alternatively, logarithmic transformations allow the main sequence to be expressed as linear for saccades within the <inline-formula id="inf27">
<mml:math id="m27">
<mml:mrow>
<mml:mn>1</mml:mn>
<mml:mo>−</mml:mo>
<mml:mo>−</mml:mo>
<mml:mn>15</mml:mn>
</mml:mrow>
</mml:math>
</inline-formula> degree range (Bahill et al., 1975[^B6]; 1981[^B7]), as illustrated in . This approach simplifies analysis while preserving the relationship’s fundamental trends.


In pursuit of greater robustness, alternative approaches have explored simpler models. For example, square-root models have been proposed to enhance the reliability of *main sequence* estimation (Lebedev et al., 1996[^B119]). These models demonstrate strong generalization and repeatability, as highlighted in a recent review by Gibaldi and Sabatini (2021)[^B68]. Despite their simplicity, square-root models effectively capture the main sequence’s three primary trends when applied to saccades larger than 1°—a threshold that aligns with the typical amplitude range of microsaccades (Martinez-Conde et al., 2009[^B136]). In conclusion, while multiple modeling approaches exist, the main sequence remains a foundational tool for understanding saccadic dynamics, with applications ranging from clinical diagnostics to explorations of the fundamental mechanisms underlying oculomotor control.




### Smooth pursuit measures

Smooth pursuits represent another type of eye movement from which valuable metrics can be extracted. In natural scene viewing conditions, smooth pursuits occur alongside fixations and saccades to track moving objects within the field of view. To isolate these pursuit sequences, algorithms outlined in  must first be applied. In real-world scenarios, targets often move unpredictably, changing speed and direction rapidly. Such stimuli are rarely used in laboratory settings, as the performance of the smooth pursuit system is limited under these conditions, often resulting in interfering saccades that complicate the analysis.


In controlled experimental conditions, smooth pursuit tasks typically require the viewer to follow targets moving horizontally or vertically at a fixed frequency, back and forth. Two common types of stimuli used in these protocols are triangular and sinusoidal motion profiles. Triangular stimuli move the target at a constant velocity in one direction before abruptly reversing direction, forming a *triangle* in position-time space. This constant-velocity motion allows researchers to precisely measure the pursuit system’s ability to maintain a steady eye velocity and to detect *catch-up* saccades when the eye lags behind the target. In contrast, sinusoidal stimuli move the target in a smooth, oscillating pattern where velocity continuously varies, peaking at mid-path and slowing near the reversal points. Sinusoidal motion more closely mimics naturalistic motion and tests the pursuit system’s ability to adapt to continuously changing velocities. In these experimental setups, it is typically assumed that the oculomotor signal reflects primarily smooth pursuit eye movements, along with any catch-up saccades, without the inclusion of fixation sequences. The pursuit system is expected to generate smooth, coordinated eye movements that closely follow the target’s trajectory, minimizing interruptions from fixational pauses.


#### Temporal and velocity features

The analysis of smooth pursuit eye movements typically starts with the estimation of fundamental descriptors, such as *pursuit duration*, *pursuit count*, and *pursuit rate*—or *pursuit frequency*. However, interpreting these metrics is not as straightforward as it might initially appear. This complexity arises primarily from the influence of *catch-up saccades*, which are corrective eye movements that compensate for discrepancies between the target’s position and the smooth pursuit response. These saccades interrupt smooth pursuit sequences, effectively shortening their duration while increasing the overall *pursuit frequency*.


More specifically, *catch-up saccades* are rapid eye movements that occur during smooth pursuit when the eye falls behind the target. They help correct the eye’s position by quickly redirecting the gaze to the moving target. These saccades occur when the smooth pursuit mechanism, which is responsible for maintaining the eye’s tracking of a moving object, is unable to keep up with sudden changes in the target’s velocity or direction. Catch-up saccades are particularly common when the target moves too fast for the smooth pursuit system to follow continuously or during pursuit of targets with unexpected changes in velocity or direction (Boman and Hotson, 1992[^B19]). Instead of maintaining a smooth motion, the eyes make these corrective jumps to *catch up* with the target, thus ensuring the target stays within the central vision. Additionally, their occurrence is modulated by factors such as target properties (Heinen et al., 2016[^B81]) and clinical conditions, including schizophrenia and affective disorders (Abel et al., 1991[^B1]).


Characterizing the velocity profile of smooth pursuit typically involves measurements of *pursuit mean velocity* and *pursuit peak velocity*. Smooth pursuit velocities are generally modest, ranging between 15 and 30° per second (Meyer et al., 1985[^B142]; Zuber et al., 1968[^B231]; Ettinger et al., 2003[^B50]; Klein and Ettinger, 2019[^B102]), significantly lower than saccadic velocities. However, trained observers or tasks involving accelerating stimuli can elicit higher peak velocities. For instance, Barmack (1970)[^B9] reported peak pursuit velocities of up to 100° per second during acceleration tasks. In humans, peak eye velocity typically occurs between 200 and 300 milliseconds after pursuit onset when following targets moving at velocities up to 30° per second (Robinson et al., 1986[^B173]).


Importantly, the velocity profile is closely linked to temporal characteristics: as stimulus velocity increases, the frequency of *catch-up saccades* also rises to correct for larger retinal offsets. A valuable descriptor for exploring this relation between velocity and compensation mechanisms is *eye crossing time*, defined as the duration required for the eye to align with the target at constant velocity. De Brouwer et al. (2002)[^B38] demonstrated that catch-up saccades are initiated when the eye crossing time reaches the saccade zone, indicating that smooth acceleration alone is insufficient for target capture.


However, simple spatio-temporal features such as *pursuit mean velocity* and *pursuit duration* do not fully capture the complexity of smooth pursuit dynamics. Smooth pursuit consists of two distinct phases: *open-loop* and *closed-loop*. In the open-loop phase, the eye’s movement is primarily driven by the initial target presentation, with little to no influence from the retinal image changes caused by the eye movement. In contrast, during the closed-loop phase, the eye continuously adjusts to changes in the retinal image that result from its own movements, maintaining the pursuit of the target. In the following , , we will introduce methods to quantify the initiation and maintenance of pursuit, respectively.



#### Smooth pursuit latency and acceleration

In this section, we introduce two classes of features used to characterize the pursuit initiation phase, namely, *pursuit latency* and *pursuit acceleration*. In target pursuit paradigms, *pursuit latency*—or *pursuit onset*—is commonly defined as the delay between the initiation of target motion and the start of ocular pursuit. The onset of smooth pursuit is typically calculated as the intersection point between two regression lines (Carl and Gellman, 1987[^B26]). The first line represents the *pre-response baseline*, which fits the velocity signal during a time window from 100 milliseconds before target motion onset to 80 milliseconds after it begins. This baseline duration may vary depending on the experimental setup, particularly when anticipation of the target motion is expected (De Hemptinne et al., 2006[^B39]). The second regression line fits the *pursuit initiation* velocity signal, typically recorded over a 50 milliseconds window after the pre-response baseline. This duration may differ across studies, often beginning at the first time point when eye velocity exceeds three to 4 standard deviations of the baseline velocity measures (Krauzlis and Miles, 1996[^B111]).


Pursuit typically exhibits much shorter latency than saccades, with *pursuit latency* ranging from 100 to 125 milliseconds, compared to 200–250 milliseconds for saccades (Krauzlis, 2004[^B110]). In experimental conditions involving anticipation, pursuit latency can be reduced to zero or even become negative, especially when pursuit begins before the target motion, such as when the direction and velocity of the stimulus are highly predictable (Burke and Barnes, 2006[^B24]; De Hemptinne et al., 2006[^B39]). Spering and Gegenfurtner (2007)[^B200] further demonstrated that *pursuit latency* is influenced by the surrounding visual context, particularly by contrast and distracting motion orientation. They found that latency decreases when the context moves in the same direction as the target, while a rapidly moving context in the opposite direction tends to *pull* the eyes back, delaying pursuit onset. Additionally, higher contrast enhances the effect of co-linear drifting context motion, further reducing the latency before the pursuit begins.


In addition to latency, pursuit initiation is often examined through *pursuit initial acceleration* (Kao and Morrow, 1994[^B96]). This is typically calculated as the mean second-order position derivative of the saccade-free component extracted from the tracking response within the first 100 milliseconds following pursuit onset. During this initial phase, acceleration continues until the eye velocity matches that of the target. The *pursuit initial peak acceleration* can also be assessed during this period. The first 20–30 milliseconds of eye acceleration show a modest increase with target velocity (Tychsen and Lisberger, 1986[^B208]). However, between 60 and 80 milliseconds after pursuit onset, eye acceleration becomes much more strongly modulated by target velocity, and is also influenced by the eccentricity of the initial eye position (Fukushima et al., 2013[^B64]).


Furthermore, like latency, the *pursuit initial acceleration* is significantly influenced by expectations regarding the target’s trajectory (Kao and Morrow, 1994[^B96]). Prior knowledge of the target’s movement—not only from its motion history but also from static visual cues—profoundly affects eye movements during pursuit initiation (Kao and Morrow, 1994[^B96]; Ladda et al., 2007[^B117]). Notably, Ladda et al. (2007)[^B117] found that cue-induced acceleration during smooth pursuit increases quadratically with target velocity. This behavior aligns with the velocity scaling predicted by the *two-thirds power law*, a natural principle of biological motion (Lacquaniti et al., 1983[^B116]).



#### Pursuit gain and accuracy

Smooth *pursuit gain* refers to the ratio of the eye’s mean velocity to the target’s mean velocity during a pursuit segment, typically under constant target velocity conditions, often referred to as *triangular stimuli*. This metric is generally assessed around 500–1,000 milliseconds after pursuit onset, during the *pursuit maintenance* phase, and serves as a measure of pursuit performance. During pursuit initiation, which occurs within the first 50–100 milliseconds after the target starts moving, pursuit gain is primarily controlled by visual motion (Rashbass, 1961[^B164]). However, in the *pursuit maintenance* phase, the gain is influenced by a combination of visual feedback regarding performance quality and internal cues, such as anticipation and prediction of target velocity (Lencer and Trillenberg, 2008[^B122]). This stable regime facilitates a more accurate assessment of performance compared to the more transient initiation phase. Typically, smooth pursuit gain is lower than 1, indicating that the eye lags behind the target, and it tends to decrease as target velocity increases (Zackon and Sharpe, 1987[^B229]).


In sinusoidal stimulation paradigms, the smooth pursuit response is usually described by two key characteristics: *pursuit velocity phase* and *pursuit velocity gain* (Accardo et al., 1995[^B2]). These values are derived by fitting the eye velocity data with a trigonometric curve for each experimental pursuit sequence. The *pursuit velocity gain* is then computed as the ratio of the peak velocity of the best-fitting curve to the peak velocity of the target’s trajectory. Similarly, the *pursuit velocity phase* is computed as the phase difference between the best-fitting velocity curve and the target’s velocity profile. Note that *overall gain* is also widely used in the literature, calculated as the ratio of eye velocity to target velocity (Churchland and Lisberger, 2002[^B32]).


Smooth pursuit is often conceptualized as a negative feedback control system in which smooth eye acceleration works to eliminate retinal motion by matching the eye velocity to the target velocity. However, substantial evidence suggests that smooth pursuit gain is modulated by an *on-line* gain control mechanism, which implies distinct visual-motor gain processing during pursuit and fixation (Robinson, 1965[^B172]; Churchland and Lisberger, 2002[^B32]). It is now widely accepted that visual inputs are not the sole mediators of smooth pursuit. Higher-order brain functions, such as attention, have been shown to play a significant role in pursuit gain and performance, though their effects have been debated (Březinová and Kendell, 1977[^B21]; Acker and Toone, 1978[^B3]; Kathmann et al., 1999[^B98]; Van Gelder et al., 1995[^B211]). Studies suggest that attention is crucial for pursuit performance (Van Donkelaar and Drew, 2002[^B210]), but Stubbs et al. (2018)[^B206] demonstrated that while increased attentional demands do not alter smooth pursuit gain, they do improve its consistency, as long as attention remains focused on the target.


Furthermore, smooth pursuit performance can be influenced by a trade-off between perceptual discrimination and pursuit efficiency. Specifically, when a perceptual discrimination task involves objects moving at a different velocity from the pursuit target, the ability to maintain smooth pursuit is compromised (Khurana and Kowler, 1987[^B101]). More recently, Kerzel et al. (2009)[^B100] or Souto and Kerzel (2014)[^B197] have further confirmed this interdependence between target selection for pursuit and perceptual processing. This interaction is generally understood as reflecting a shared, limited resource that is required for both steady-state smooth pursuit and perceptual tasks (Stolte et al., 2023[^B205]).


Finally, smooth pursuit gain has become a crucial measure in neuro-pathological research. For example, a review by Franco et al. (2014)[^B58] highlighted studies showing that individuals diagnosed with schizophrenia often exhibit lower smooth pursuit gain. Smooth pursuit performance is also a valuable tool in assessing sensorimotor development in preadolescence and adolescence. Horizontal smooth pursuit typically matures by age 7 (Ingster-Moati et al., 2009[^B90]), while vertical smooth pursuit does not reach maturity until late adolescence (Katsanis et al., 1998[^B99]). This asymmetry between horizontal and vertical pursuit is due to the involvement of different brain structures in controlling these movements (Collewijn and Tamminga, 1984[^B33]; Grönqvist et al., 2006[^B73]), with significant clinical implications. For instance, Robert et al. (2014)[^B171] demonstrated that children with developmental coordination disorder often exhibit impaired vertical pursuit performance, indicating delayed maturation of the pursuit system in this population.




## Signal analysis

In this section, we review time series analysis methods for the study of ocular behavior. Compared to traditional neurophysiological approaches, these methods are underexplored but offer a robust framework for analyzing eye movements as a cohesive, dynamic system. In contrast to neurophysiological methods, which focus on specific neural circuits associated with individual eye movement types, time series approaches capture the temporal and structural patterns of eye behavior across contexts.  summarizes the metrics and algorithms discussed, describes each method and the required input formats, and provides key literature references to facilitate implementation.


| Feature name                           | Description                                                                                                                                                           | References                 |
|:---------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------|
| Periodogram                            | Given a raw gaze signal, estimates power spectral density                                                                                                             | McGillem and Cooper (1991) |
| Welch periodogram                      | Given a raw gaze signal, estimates power spectral density, using a Welch windowed periodogram                                                                         | Welch (1967)               |
| Cross spectral density                 | Given a set of raw gaze signals, estimates the cross power spectral density between pairs of signals                                                                  | McGillem and Cooper (1991) |
| Welch cross spectral density           | Given a set of raw gaze signals, estimates the cross power spectral density between pairs of signals, according to Welch’s method                                     | McGillem and Cooper (1991) |
| Coherency                              | Given a set of raw gaze signals, estimates how strongly pairs of signals are related at specific frequencies                                                          | Bendat and Piersol (1986)  |
| Mean squared displacement              | Given a raw gaze signal, estimates the average squared deviation of the eye-gaze position from a reference position over time                                         | Herrmann et al. (2017)     |
| Displacement auto-correlation function | Given a raw gaze signal, estimates the degree of similarity between the gaze signal and a lagged version of itself over successive time intervals                     | Herrmann et al. (2017)     |
| Detrended fluctuation analysis         | Given a raw gaze signal, estimates long-range correlations and scaling behavior by analyzing signal fluctuations over different time scales                           | Wang and Cong (2015)       |
| Persistence size                       | Given a raw gaze signal, estimates the entropy of the size of the holes in the persistence diagram obtained from gaze signal                                          | Chung et al. (2021)        |
| Persistence robustness                 | Given a raw gaze signal, estimates the entropy of the robustness of the holes in the persistence diagram obtained from gaze signal                                    | Chung et al. (2021)        |
| Betti curve                            | Given a raw gaze signal, estimates a function evaluating the Betti numbers obtained from a persistence diagram, at different levels of filtration                     | Güzel and Kaygun (2023)    |
| persistence curve                      | Given a raw gaze signal, estimates a function that summarizes the total persistence of topological hole of the persistence diagram, at different levels of filtration | Kachan and Onuchin (2021)  |
| Persistence entropy                    | Given a raw gaze signal, estimates the Shannon entropy of the collections of topological holes lifetimes of the persistence diagram obtained from gaze signal         | Kachan and Onuchin (2021)  |


### Frequency variables


 described methods for characterizing eye movements, focusing on spatial and temporal attributes such as fixation locations and saccade kinematics. These approaches often neglect the dynamic processes underlying these patterns. Spectral analysis provides an alternative framework by examining the frequency content of eye movement time series, revealing oscillatory patterns that reflect underlying dynamics (Stoica and Moses, 2005[^B204]).


The spectral content of gaze data is commonly analyzed using the *discrete Fourier transform* (DFT), which converts the ocular signal into a frequency-domain representation (McGillem and Cooper, 1991[^B138]). The DFT decomposes the signal by correlating it with sinusoids of varying frequencies, identifying dominant rhythmic components. The *power spectral density* (PSD) complements this by quantifying the amplitude of these rhythms as a function of frequency, offering insights into the signal’s oscillatory structure. Welch’s method (Welch, 1967[^B222]), a widely adopted PSD estimation technique, segments the signal into overlapping windows, applies a window function, and averages the squared DFT magnitudes across segments. This approach balances frequency resolution and statistical reliability, yielding robust PSD estimates with reduced noise.


Spectral analysis also enables comparative studies of gaze data through metrics such as cross-spectral density and signal coherence, which are valuable for analyzing eye movement behavior across experimental conditions, individuals, or species (Ko et al., 2016[^B104]). *Cross-spectral density* measures the frequency-specific covariance between two signals, while *signal coherence*, derived from cross-spectral density, quantifies the consistency of phase relationships, revealing synchronized rhythmic activities. For instance, Nakayama and Shimizu (2004)[^B150] used cross-spectral density to demonstrate task-related differences in the coordination of horizontal and vertical eye movement components, highlighting the influence of task difficulty. Additionally, spectral analysis has been applied to compare real and synthetic gaze data, enabling evaluation of generative models. Duchowski et al. (2016)[^B46] utilized spectral analysis to distinguish experimentally recorded gaze patterns from synthetic ones, advancing insights into eye movement dynamics.



### Stochastic variables

Directly comparing eye movement data is challenging due to the stochastic, or inherently random, nature of gaze signals, as discussed in . Modeling eye movements as random variables provides an alternative approach, uncovering physiological patterns through their statistical characteristics. A key tool, the *mean squared displacement* (MSD), tracks how gaze positions shift over time. In simple random walks, like Brownian motion with independent steps, the spread grows steadily. In complex cases, such as eye movements, the spread follows a power-law pattern, reflecting diverse neural and behavioral dynamics.


Isolated fixational eye movements, such as microsaccades and drift, are well-suited for stochastic analysis due to their structured yet random nature. Engbert and Kliegl (2004)[^B48] used the MSD to reveal distinct patterns in these movements. On short time scales—tens to hundreds of milliseconds—fixational movements are persistent, following consistent directions to promote retinal shifts that prevent visual fading. On longer time scales, they become *anti-persistent*, with negatively correlated increments that facilitate maintaining gaze on the intended fixation point.


Detrended fluctuation analysis (DFA), another powerful method, quantifies long-term power-law correlations in non-stationary gaze data. Moshel et al. (2008)[^B147] applied DFA to demonstrate that microsaccades enhance persistence more in horizontal than vertical fixational movements, suggesting distinct neural control mechanisms for these components (Sparks, 1986[^B198]; Moschovakis, 1996[^B146]). Beyond physiological studies, DFA has been used in functional research. For example, Wang and Cong (2015)[^B221] employed DFA to investigate how professional experience shapes eye movement patterns in air traffic controllers, linking gaze dynamics to cognitive and task-related factors.


Finally, the MSD analysis of fixational movements exhibits oscillatory behavior over longer time scales (Herrmann et al., 2017[^B83]). The displacement auto-correlation function (DACF) complements MSD by comparing a movement’s trajectory to its delayed versions, highlighting these rhythmic patterns. Such patterns suggest that drift movements are centrally controlled, potentially through time-delayed feedback mechanisms (Herrmann et al., 2017[^B83]). These methods, summarized in , provide insights into the dynamic control of gaze allowing to explore additional temporal patterns.



### Topological variables

Recent studies have applied topological data analysis (TDA) to investigate the complex patterns of eye movement trajectories. Conventional measures, such as fixation durations or saccade amplitudes, often fail to capture the broader spatial and temporal structure of gaze patterns. Pioneering works by Kachan and Onuchin (2021)[^B95] and Onuchin and Kachan (2023)[^B154] addressed this limitation by using TDA to extract novel features from eye movement data, demonstrating improved performance in recognition tasks on new gaze trajectory datasets. More recently, He et al. (2025)[^B80] showed that spatial-temporal topological features derived from eye-tracking data can be informative for neural disorder screening, highlighting the clinical relevance of these TDA-based representations.


A central tool in TDA is persistent homology, which provides a way to measure the *shape* of a dataset across multiple scales. To illustrate, consider a set of eye positions represented as points in space. Persistent homology tracks the formation and disappearance of topological features, including connected clusters of points, circular arrangements forming loops, and higher-dimensional empty regions called voids. These features are identified through a process called a filtration, in which a scale parameter gradually increases. Initially, each point is separate, but as the scale grows, points that are close to each other become connected. A topological feature is said to be *born* when it first appears, for example, when two points merge into a cluster or a loop forms, and it *dies* when it disappears, such as when two clusters merge into one larger cluster or a loop is filled in. By recording the birth and death of each feature, the structural information of the dataset can be summarized in a persistence diagram, where longer-lived features typically represent meaningful structures while short-lived features correspond to noise (Carlsson, 2009[^B27]; Edelsbrunner and Harer, 2022[^B47]).  illustrates this process schematically.

<figure id="F7" position="float">

<img src="/assets/fphys-16-1661026-g007.webp"/><figcaption>
<p>Forming Persistence Diagrams. Given a set of points—gaze data-samples—the Vietoris-Rips filtration approximates the topology of the union of the balls of radius equal to the threshold parameter <inline-formula id="inf28">
<mml:math id="m28">
<mml:mrow>
<mml:mi>α</mml:mi>
</mml:mrow>
</mml:math>
</inline-formula> centered at each point from the dataset. The <bold>(a)</bold> shows, for three values of <inline-formula id="inf29">
<mml:math id="m29">
<mml:mrow>
<mml:mi>α</mml:mi>
</mml:mrow>
</mml:math>
</inline-formula> — also represented by dotted lines in <bold>(b)</bold> — appearance of topological features of dimension 0 — purple lines for connected components—and dimension 1 — blue shaded areas for holes. The persistence diagram, or persistence <italic>barcode</italic>, plotted <bold>(b)</bold> of dimension 0 — purple bars—summarizes the linking of clusters while the persistence diagram of dimension 1 — blue bars—summarizes the number of topological holes between clusters, describing the complexity of clusters arrangement.</p>
</figcaption>

</figure>

One common method to build topological structures is the Vietoris-Rips complex. In this approach, points in a cloud are connected if they are within a certain distance defined by the current scale parameter. Sets of points that are mutually connected form higher-dimensional shapes: a pair of points forms a line segment, three points form a filled triangle, and four points form a tetrahedron. As the scale increases, more connections are added, creating new features or merging existing ones. This gradual growth generates the birth and death events that are tracked in persistent homology.



Kachan and Onuchin (2021)[^B95] proposed two TDA-based approaches for analyzing eye movements. In the first, eye positions are treated as a point cloud, ignoring timestamps, to capture spatial patterns. In the second, horizontal and vertical gaze coordinates are analyzed as separate time series to study temporal dynamics. From these representations, persistence diagrams are derived and transformed into compact features, such as the lifespan of topological features or their stability across scales. These features can be computed for Vietoris-Rips complexes or for sub-level set filtrations, which track the appearance and disappearance of features as the values of the data themselves vary, for example, along intensity or velocity thresholds. Persistence diagrams can then be vectorized into structured formats suitable for machine learning, enabling classification, clustering, or other data-driven analyses. By emphasizing shape-related properties of gaze data, TDA provides tools to capture structural patterns that traditional metrics often overlook, and as shown by He et al. (2025)[^B80], these spatial-temporal topological features can also serve as biomarkers for neural disorder screening.



## Discussion

The segmentation of raw gaze data into a sequence of oculomotor events remains a cornerstone of eye movement research. In this article, we have reviewed the most common segmentation algorithms—). Historically, threshold-based methods dominated the field, relying on predefined criteria such as velocity or displacement thresholds to categorize eye movements. These approaches remain widely used because of their simplicity, computational efficiency, and relatively low barrier to implementation. However, they also exhibit critical limitations: their sensitivity to parameter selection can lead to inconsistent results across laboratories, and their robustness often degrades in noisy or dynamic environments, such as mobile or low-cost eye trackers. These drawbacks highlight the need for approaches that are less dependent on arbitrary thresholds and more adaptable to variability in recording conditions.


In contrast, learning-based approaches have gained prominence by leveraging annotated datasets that encode expert knowledge of eye movement types. By training models on rich and diverse data, these methods can capture complex patterns in the gaze signal that extend beyond traditional definitions of fixations, saccades, and pursuits. For instance, they are better suited to handle ambiguous or overlapping cases, where threshold-based approaches often fail. Nevertheless, their performance is critically dependent on model architecture, hyperparameter optimization, and, above all, the quality, diversity, and size of the training datasets. A model trained on limited or biased data may perform well within a narrow domain but fail to generalize to different populations, tasks, or devices. This dependency underscores the importance of carefully curated datasets and rigorous cross-validation protocols.


To foster transparency and reproducibility in machine learning–based segmentation, detailed methodological reporting is essential. Beyond describing the general algorithmic approach, authors should provide explicit documentation of the algorithms and software packages employed, the hyperparameter configurations chosen, and the strategies used for validation. Where feasible, access to training and validation datasets should also be shared, either through open repositories or upon reasonable request. Such openness ensures that results can be replicated, facilitates the systematic refinement of models, and lowers the entry barrier for new research groups seeking to build upon existing work. Ultimately, transparent reporting practices strengthen confidence in published findings and encourage convergence toward best practices in the field.


In this regard, specialized databases are playing an increasingly central role. Resources such as the GazeBase dataset (Griffith et al., 2021[^B72]) provide large and heterogeneous eye movement recordings across diverse tasks, from controlled guided stimuli designed to elicit specific movements, to goal-directed activities, and free-viewing scenarios such as reading or video watching. These datasets are indispensable for benchmarking both traditional and learning-based algorithms, enabling fair comparisons across methods, and for training models with stronger generalizability across tasks and hardware. By facilitating standardized evaluation, such databases support the transition from isolated methodological contributions toward a cumulative science of eye movement analysis. Looking ahead, the expansion of open repositories covering diverse populations, age groups, and experimental contexts will be critical for building robust segmentation algorithms with real-world applicability.


Beyond segmentation itself, this article has also reviewed the metrics derived from canonical oculomotor events—). These metrics are essential for characterizing fixations, saccades, and smooth pursuits in terms of their temporal, spatial, and kinematic properties, and for linking them to cognitive, clinical, and applied research contexts. For example, fixation duration can be tied to attentional processes, while saccade amplitude and velocity are informative about motor control and neurological function. However, meaningful cross-study comparisons are only possible if these metrics are computed in standardized ways and interpreted within a shared conceptual framework. Advancing this line of work therefore requires: <inline-formula id="inf30">
<mml:math id="m30">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> a unified set of definitions and formal concepts, <inline-formula id="inf31">
<mml:math id="m31">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> standardized analytical pipelines that minimize methodological variability, and <inline-formula id="inf32">
<mml:math id="m32">
<mml:mrow>
<mml:mo stretchy="false">(</mml:mo>
<mml:mrow>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
<mml:mi>i</mml:mi>
</mml:mrow>
<mml:mo stretchy="false">)</mml:mo>
</mml:mrow>
</mml:math>
</inline-formula> accessible open-source datasets and software packages that encourage reproducibility and methodological convergence. Together, these elements will harmonize computational practices, foster interdisciplinary collaboration, and ultimately improve the comparability and interpretability of findings across the diverse fields that rely on eye movement research.


It is important to stress, however, that the robustness of segmentation and derived metrics depends strongly on the hardware employed. High-speed laboratory-grade eye trackers — <inline-formula id="inf33">
<mml:math id="m33">
<mml:mrow>
<mml:mn>500</mml:mn>
<mml:mo>−</mml:mo>
<mml:mo>−</mml:mo>
<mml:mn>1000</mml:mn>
<mml:mspace width="0.3333em"/>
<mml:mi>H</mml:mi>
<mml:mi>z</mml:mi>
</mml:mrow>
</mml:math>
</inline-formula> — provide fine-grained temporal resolution, yielding reliable estimates of fixation stability, saccade dynamics, and pursuit gain. In these conditions, reproducibility is typically high for metrics such as RMSD or Cohen’s Kappa. By contrast, low-cost or mobile devices — <inline-formula id="inf34">
<mml:math id="m34">
<mml:mrow>
<mml:mn>30</mml:mn>
<mml:mo>−</mml:mo>
<mml:mo>−</mml:mo>
<mml:mn>120</mml:mn>
<mml:mspace width="0.3333em"/>
<mml:mi>H</mml:mi>
<mml:mi>z</mml:mi>
</mml:mrow>
</mml:math>
</inline-formula> — are more prone to noise and data loss, which introduces uncertainty in event boundaries. Fixations, being relatively long in duration, are somewhat resilient, although noise can still inflate false positives. Saccades, in turn, are especially vulnerable: low sampling rates may miss peak velocities or misestimate onset and offset times, leading to degraded temporal precision and event-level accuracy. These differences underscore the need for robust, hardware-agnostic metrics that remain interpretable across diverse research settings.


Looking ahead, several technological and methodological trends promise to reshape oculomotor research. The rapid adoption of VR platforms equipped with eye tracking enables exploration of gaze behavior in immersive, ecologically valid 3D contexts, where traditional eye movements interact with head and body dynamics (Adhanom et al., 2023[^B4]). The growing use of mobile eye tracking is similarly expanding research far beyond lab settings, though it raises significant challenges in data quality and reproducibility (Fu et al., 2024[^B61]). On the computational front, while AI and deep learning methods for event segmentation are emerging, the need for rigorous evaluation and privacy-aware implementations remains pressing—especially in VR contexts (Bozkir et al., 2023[^B20]). More broadly, as Extended Reality (XR) environments integrate eye tracking with multimodal sensors, methodologies must adapt to both technological possibilities and ethical considerations (Kourtesis, 2024[^B109]). Together, these advances point toward richer, more scalable, and context-sensitive analyses of oculomotor behavior.


Finally, we reviewed emerging approaches that challenge the traditional paradigm of segmentation into discrete events—. Advanced signal processing methods, such as topological data analysis (TDA), enable the study of the intrinsic structure of eye movement signals without imposing predefined categories. By focusing on patterns such as connectivity, loops, or voids in gaze trajectories, TDA captures structural properties that may be overlooked by conventional event-based frameworks. This represents a promising step toward more naturalistic analyses, particularly in contexts where boundaries between fixations, saccades, and pursuits are ambiguous or functionally irrelevant. As these methods mature, they are likely to complement existing frameworks and enrich our understanding of oculomotor control in real-world visual behavior.

[^B1]: Abel L .   A . , Friedman L . , Jesberger J . , Malki A . , Meltzer H .  1991  Quantitative assessment of smooth pursuit gain and catch-up saccades in schizophrenia and affective disorders  Biol. psychiatry  1063  1072  29 
[^B2]: Accardo A . , Pensiero S . , Da Pozzo S . , Perissutti P .  1995  Characteristics of horizontal smooth pursuit eye movements to sinusoidal stimulation in children of primary school age  Vis. Res.  539  548  35 
[^B3]: Acker W . , Toone B .  1978  Attention, eye tracking and schizophrenia  Br. J. Soc. Clin. Psychol.  173  181  17 
[^B4]: Adhanom I .   B . , MacNeilage P . , Folmer E .  2023  Eye tracking in virtual reality: a broad review of applications and challenges  Virtual Real.  1481  1505  27 
[^B5]: Andersson R . , Larsson L . , Holmqvist K . , Stridh M . , Nyström M .  2016  One algorithm to rule them all? an evaluation and discussion of ten eye movement event-detection algorithms  Behav. Res. Methods  616  637  49 
[^B6]: Bahill A .   T . , Clark M .   R . , Stark L .  1975  The main sequence, a tool for studying human eye movements  Math. Biosci.  191  204  24 
[^B7]: Bahill A . , Brockenbrough A . , Troost B .  1981  Variability and development of a normative data base for saccadic eye movements  Investigative Ophthalmol. and Vis. Sci.  116  125  21 
[^B8]: Baloh R .   W . , Sills A .   W . , Kumley W .   E . , Honrubia V .  1975  Quantitative measurement of saccade amplitude, duration, and velocity  Neurology  1065  1070  25 
[^B9]: Barmack N .  1970  Modification of eye movements by instantaneous changes in the velocity of visual targets  Vis. Res.  1431  1441  10 
[^B10]: Becker W .  1991  Saccades  Vis. Vis. Dysfunct.  95  137  8 
[^B11]: Bendat J . , Piersol A .  1986  Random data: analysis and measurement procedures  wiley-interscience publication 
[^B12]: Berges A .   J . , Vedula S .   S . , Chara A . , Hager G .   D . , Ishii M . , Malpani A .  2023  Eye tracking and motion data predict endoscopic sinus surgery skill  Laryngoscope  500  505  133 
[^B13]: Bhoir S .   A . , Hasanzadeh S . , Esmaeili B . , Dodd M .   D . , Fardhosseini M .   S .  2015  Measuring construction workers attention using eye-tracking technology 
[^B14]: Bijvank J .   N . , van Rijn L . , Kamminga M . , Tan H . , Uitdehaag B . , Petzold A .  2019  Saccadic fatigability in the oculomotor system  J. neurological Sci.  167  174  402 
[^B15]: Bilmes J .   A .  1998  A gentle tutorial of the em algorithm and its application to parameter estimation for Gaussian mixture and hidden markov models  Int. Comput. Sci. Inst.  126  4 
[^B16]: Binkley D . , Davis M . , Lawrie D . , Maletic J .   I . , Morrell C . , Sharif B .  2013  The impact of identifier style on effort and comprehension  Empir. Softw. Eng.  219  276  18 
[^B17]: Birawo B . , Kasprowski P .  2022  Review and evaluation of eye movement event detection algorithms  Sensors  8810  22 
[^B18]: Blignaut P .  2009  Fixation identification: the optimum threshold for a dispersion algorithm  Atten. Percept. and Psychophys.  881  895  71 
[^B19]: Boman D .   K . , Hotson J .   R .  1992  Predictive smooth pursuit eye movements near abrupt changes in motion direction  Vis. Res.  675  689  32 
[^B20]: Bozkir E . , Özdel S . , Wang M . , David-John B . , Gao H . , Butler K .  2023  Eye-tracked virtual reality: a comprehensive survey on methods and privacy challenges  arXiv Prepr. arXiv:2305.14080 
[^B21]: Březinová V . , Kendell R .  1977  Smooth pursuit eye movements of schizophrenics and normal people under stress  Br. J. Psychiatry  59  63  130 
[^B22]: Brunyé T .   T . , Drew T . , Weaver D .   L . , Elmore J .   G .  2019  A review of eye tracking for understanding and improving diagnostic interpretation  Cognitive Res. Princ. Implic.  7  16  4 
[^B23]: Brysbaert M . , Vitu F . , Schroyens W .  1996  The right visual field advantage and the optimal viewing position effect: on the relation between foveal and parafoveal word recognition  Neuropsychology  385  395  10 
[^B24]: Burke M . , Barnes G .  2006  Quantitative differences in smooth pursuit and saccadic eye movements  Exp. brain Res.  596  608  175 
[^B25]: Camerini P . , Galbiati G . , Maffioli F .  1988  Algorithms for finding optimum trees: description, use and evaluation  Ann. Operations Res.  263  397  13 
[^B26]: Carl J . , Gellman R .  1987  Human smooth pursuit: stimulus-dependent responses  J. Neurophysiology  1446  1463  57 
[^B27]: Carlsson G .  2009  Topology and data  Bull. Am. Math. Soc.  255  308  46 
[^B28]: Cerquera-Jaramillo M .   A . , Nava-Mesa M .   O . , González-Reyes R .   E . , Tellez-Conti C . , de-la Torre A .  2018  Visual features in alzheimer’s disease: from basic mechanisms to clinical overview  Neural plast.  2941783  2018 
[^B29]: Chen Y . - F . , Lin H . - H . , Chen T . , Tsai T . - T . , Shee I . - F .  2002  The peak velocity and skewness relationship for the reflexive saccades  Biomed. Eng. Appl. Basis Commun.  71  80  14 
[^B30]: Chen-Harris H . , Joiner W .   M . , Ethier V . , Zee D .   S . , Shadmehr R .  2008  Adaptive control of saccades via internal feedback  J. Neurosci.  2804  2813  28 
[^B31]: Chung Y . - M . , Hu C . - S . , Lo Y . - L . , Wu H . - T .  2021  A persistent homology approach to heart rate variability analysis with an application to sleep-wake classification  Front. physiology  637684  12 
[^B32]: Churchland A .   K . , Lisberger S .   G .  2002  Gain control in human smooth-pursuit eye movements  J. Neurophysiology  2936  2945  87 
[^B33]: Collewijn H . , Tamminga E .   P .  1984  Human smooth and saccadic eye movements during voluntary pursuit of different target motions on different backgrounds  J. physiology  217  250  351 
[^B34]: Collins T . , Doré-Mazars K .  2006  Eye movement signals influence perception: evidence from the adaptation of reactive and volitional saccades  Vis. Res.  3659  3673  46 
[^B35]: Collins T . , Semroud A . , Orriols E . , Doré-Mazars K .  2008  Saccade dynamics before, during, and after saccadic adaptation in humans  Investigative Ophthalmol. and Vis. Sci.  604  612  49 
[^B36]: Crossland M . , Sims M . , Galbraith R . , Rubin G .  2004  Evaluation of a new quantitative technique to assess the number and extent of preferred retinal loci in macular disease  Vis. Res.  1537  1546  44 
[^B37]: Crossland M .   D . , Dunbar H .   M . , Rubin G .   S .  2009  Fixation stability measurement using the mp1 microperimeter  Retina  651  656  29 
[^B38]: De Brouwer S . , Yuksel D . , Blohm G . , Missal M . , Lefèvre P .  2002  What triggers catch-up saccades during visual tracking?  J. neurophysiology  1646  1650  87 
[^B39]: De Hemptinne C . , Lefevre P . , Missal M .  2006  Influence of cognitive expectation on the initiation of anticipatory and visual pursuit eye movements in the rhesus monkey  J. neurophysiology  3770  3782  95 
[^B40]: Di Stasi L .   L . , Catena A . , Canas J .   J . , Macknik S .   L . , Martinez-Conde S .  2013  Saccadic velocity as an arousal index in naturalistic tasks  Neurosci. and Biobehav. Rev.  968  975  37 
[^B41]: Dodge R . , Benedict F .   G .  1915  Psychological effects of alcohol: an experimental investigation of the effects of moderate doses of ethyl alcohol on a related group of neuro-muscular processes in man  232  Carnegie institution of Washington 
[^B42]: Dodge R . , Cline T .   S .  1901  The angle velocity of eye movements  Psychol. Rev.  145  157  8 
[^B43]: Donkelaar P .   v . , Siu K . - C . , Walterschied J .  2004  Saccadic output is influenced by limb kinetics during eye—hand coordination  J. Mot. Behav.  245  252  36 
[^B44]: Doyle M . , Walker R .  2001  Curved saccade trajectories: voluntary and reflexive saccades curve away from irrelevant distractors  Exp. brain Res.  333  344  139 
[^B45]: Duchowski A .   T . , Duchowski A .   T .  2017  Eye tracking methodology: theory and practice  Springer 
[^B46]: Duchowski A .   T . , Jörg S . , Allen T .   N . , Giannopoulos I . , Krejtz K .  2016  Eye movement synthesis  Proceedings of the ninth biennial ACM symposium on eye tracking research and applications  147  154 
[^B47]: Edelsbrunner H . , Harer J .   L .  2022  Computational topology: an introduction  American Mathematical Society 
[^B48]: Engbert R . , Kliegl R .  2004  Microsaccades keep the eyes’ balance during fixation  Psychol. Sci.  431  436  15 
[^B49]: Ester M . , Kriegel H . - P . , Sander J . , Xu X .  1996  A density-based algorithm for discovering clusters in large spatial databases with noise  kdd  226  231  96 
[^B50]: Ettinger U . , Kumari V . , Crawford T .   J . , Davis R .   E . , Sharma T . , Corr P .   J .  2003  Reliability of smooth pursuit, fixation, and saccadic eye movements  Psychophysiology  620  628  40 
[^B51]: Federighi P . , Cevenini G . , Dotti M .   T . , Rosini F . , Pretegiani E . , Federico A .  2011  Differences in saccade dynamics between spinocerebellar ataxia 2 and late-onset cerebellar ataxias  Brain  879  891  134 
[^B52]: Federighi P . , Ramat S . , Rosini F . , Pretegiani E . , Federico A . , Rufa A .  2017  Characteristic eye movements in ataxia-telangiectasia-like disorder: an explanatory hypothesis  Front. neurology  596  8 
[^B53]: Fernández G . , Mandolesi P . , Rotstein N .   P . , Colombo O . , Agamennoni O . , Politi L .   E .  2013  Eye movement alterations during reading in patients with early alzheimer disease  Investigative Ophthalmol. and Vis. Sci.  8345  8352  54 
[^B54]: Findlay J .   M . , Gilchrist I .   D .  2003  Active vision: the psychology of looking and seeing  37  Oxford University Press 
[^B55]: Fischer B . , Weber H .  1993  Express saccades and visual attention  Behav. Brain Sci.  553  567  16 
[^B56]: Fletcher W .   A . , Sharpe J .   A .  1986  Saccadic eye movement dysfunction in alzheimer’s disease  Ann. Neurology Official J. Am. Neurological Assoc. Child Neurology Soc.  464  471  20 
[^B57]: Foulsham T . , Kingstone A . , Underwood G .  2008  Turning the world around: patterns in saccade direction vary with picture orientation  Vis. Res.  1777  1790  48 
[^B58]: Franco J . , De Pablo J . , Gaviria A . , Sepúlveda E . , Vilella E .  2014  Smooth pursuit eye movements and schizophrenia: literature review  Arch. la Soc. Española Oftalmol. English Ed.  361  367  89 
[^B59]: Fried M . , Tsitsiashvili E . , Bonneh Y .   S . , Sterkin A . , Wygnanski-Jaffe T . , Epstein T .  2014  Adhd subjects fail to suppress eye blinks and microsaccades while anticipating visual stimuli but recover with medication  Vis. Res.  62  72  101 
[^B60]: Frohman E . , Frohman T . , O’suilleabhain P . , Zhang H . , Hawker K . , Racke M .  2002  Quantitative oculographic characterisation of internuclear ophthalmoparesis in multiple sclerosis: the versional dysconjugacy index z score  J. Neurology, Neurosurg. and Psychiatry  51  55  73 
[^B61]: Fu X . , Franchak J .   M . , MacNeill L .   A . , Gunther K .   E . , Borjon J .   I . , Yurkovic-Harding J .  2024  Implementing mobile eye tracking in psychological research: a practical guide  Behav. Res. Methods  8269  8288  56 
[^B62]: Fuhl W . , Castner N . , Kasneci E .  2018  Histogram of oriented velocities for eye movement detection  Proceedings of the workshop on modeling cognitive processes from multimodal data  1  6 
[^B63]: Fuhl W . , Rong Y . , Kasneci E .  2021  Fully convolutional neural networks for raw eye tracking data segmentation, generation, and reconstruction  
2020 25th international Conference on pattern recognition (ICPR) (IEEE)  142  149 
[^B64]: Fukushima K . , Fukushima J . , Warabi T . , Barnes G .   R .  2013  Cognitive processes involved in smooth pursuit eye movements: behavioral evidence, neural substrate and clinical correlation  Front. Syst. Neurosci.  4  7 
[^B65]: Galley N .  1989  Saccadic eye movement velocity as an indicator of (de) activation. a revievv and some speculations 
[^B66]: Garbutt S . , Harwood M .   R . , Kumar A .   N . , Han Y .   H . , Leigh R .   J .  2003  Evaluating small eye movements in patients with saccadic palsies  Ann. N. Y. Acad. Sci.  337  346  1004 
[^B67]: Ghasia F . , Wang J .  2022  Amblyopia and fixation eye movements  J. Neurological Sci.  120373  441 
[^B68]: Gibaldi A . , Sabatini S .   P .  2021  The saccade main sequence revised: a fast and repeatable tool for oculomotor analysis  Behav. Res. Methods  167  187  53 
[^B69]: Goldberg J .   H . , Kotval X .   P .  1999  Computer interface evaluation using eye movements: methods and constructs  Int. J. industrial ergonomics  631  645  24 
[^B70]: Goldberg J .   H . , Schryver J .   C .  1995  Eye-gaze-contingent control of the computer interface: Methodology and example for zoom detection  Behav. Res. methods, Instrum. and Comput.  338  350  27 
[^B71]: Golla H . , Tziridis K . , Haarmeier T . , Catz N . , Barash S . , Thier P .  2008  Reduced saccadic resilience and impaired saccadic adaptation due to cerebellar disease  Eur. J. Neurosci.  132  144  27 
[^B72]: Griffith H . , Lohr D . , Abdulin E . , Komogortsev O .  2021  Gazebase, a large-scale, multi-stimulus, longitudinal eye movement dataset  Sci. Data  184  8 
[^B73]: Grönqvist H . , Gredebäck G . , von Hofsten C .  2006  Developmental asymmetries between horizontal and vertical tracking  Vis. Res.  1754  1761  46 
[^B74]: Guadron L . , Titchener S .   A . , Abbott C .   J . , Ayton L .   N . , van Opstal J . , Petoe M .   A .  2023  The saccade main sequence in patients with retinitis pigmentosa and advanced age-related macular degeneration  Investigative Ophthalmol. and Vis. Sci.  1  64 
[^B75]: Gupta S . , Routray A .  2012  Estimation of saccadic ratio from eye image sequences to detect human alertness  2012 4th international conference on intelligent human computer interaction (IHCI)  1  6  IEEE 
[^B76]: Güzel İ . , Kaygun A .  2023  Classification of stochastic processes with topological data analysis  Concurrency Comput. Pract. Exp.  e7732  35 
[^B77]: Harris C .   M . , Wolpert D .   M .  2006  The main sequence of saccades optimizes speed-accuracy trade-off  Biol. Cybern.  21  29  95 
[^B78]: Harwood M .   R . , Mezey L .   E . , Harris C .   M .  1999  The spectral main sequence of human saccades  J. Neurosci.  9098  9106  19 
[^B79]: Hayhoe M . , Ballard D .  2005  Eye movements in natural behavior  Trends cognitive Sci.  188  194  9 
[^B80]: He D . , Wang S . , Ogmen H .  2025  Spatial-temporal topological features in eye tracking data are informative for neural disorder screening  Investigative Ophthalmol. and Vis. Sci.  3893  66 
[^B81]: Heinen S .   J . , Potapchuk E . , Watamaniuk S .   N .  2016  A foveal target increases catch-up saccade frequency during smooth pursuit  J. neurophysiology  1220  1227  115 
[^B82]: Henderson J .   M .  2003  Human gaze control during real-world scene perception  Trends cognitive Sci.  498  504  7 
[^B83]: Herrmann C .   J . , Metzler R . , Engbert R .  2017  A self-avoiding walk with neural delays as a model of fixational eye movements  Sci. Rep.  12958  17  7 
[^B84]: Hessels R .   S . , Niehorster D .   C . , Kemner C . , Hooge I .   T .  2017  Noise-robust fixation detection in eye movement data: identification by two-means clustering (i2mc)  Behav. Res. methods  1802  1823  49 
[^B85]: Hessels R .   S . , Niehorster D .   C . , Nyström M . , Andersson R . , Hooge I .   T .  2018  Is the eye-movement field confused about fixations and saccades? a survey among 124 researchers  R. Soc. open Sci.  180502  5 
[^B86]: Holmqvist K . , Nystrom M . , Andersson R . , Dewhurst R . , Jarodzka H . , Van de Weijer J .  2011  Eye tracking: a comprehensive guide to methods and measures  Oxford University Press 
[^B87]: Hoppe S . , Bulling A .  2016  End-to-end eye movement detection using convolutional neural networks  arXiv Prepr. arXiv:1609.02452 
[^B88]: Hsiao J .   H . - w . , Cottrell G .  2008  Two fixations suffice in face recognition  Psychol. Sci.  998  1006  19 
[^B89]: Inchingolo P . , Spanio M .  1985  On the identification and analysis of saccadic eye movements-a quantitative study of the processing procedures  IEEE Trans. Biomed. Eng.  683  695  32 
[^B90]: Ingster-Moati I . , Vaivre-Douret L . , Quoc E .   B . , Albuisson E . , Dufier J . - L . , Golse B .  2009  Vertical and horizontal smooth pursuit eye movements in children: a neuro-developmental study  Eur. J. Paediatr. neurology  362  366  13 
[^B91]: Inhoff A .   W . , Radach R .  1998  Definition and computation of oculomotor measures in the study of cognitive processes  Eye Guid. Read. scene Percept.  29  53 
[^B92]: Inhoff A .   W . , Radach R . , Starr M . , Greenberg S .  2000  Allocation of visuo-spatial attention and saccade programming during reading  Reading as a perceptual process  221  246  Elsevier 
[^B93]: Jacob R .   J . , Karn K .   S .  2003  Eye tracking in human-computer interaction and usability research: ready to deliver the promises  The mind’s eye  573  605  Elsevier 
[^B94]: Jensen K . , Beylergil S .   B . , Shaikh A .   G .  2019  Slow saccades in cerebellar disease  Cerebellum and Ataxias  1  9  6 
[^B95]: Kachan O . , Onuchin A .  2021  Topological data analysis of eye movements  IEEE 18th international symposium on biomedical imaging 
[^B96]: Kao G .   W . , Morrow M .   J .  1994  The relationship of anticipatory smooth eye movement to smooth pursuit initiation  Vis. Res.  3027  3036  34 
[^B97]: Kasneci E . , Kasneci G . , Kübler T .   C . , Rosenstiel W .  2015  Online recognition of fixations, saccades, and smooth pursuits for automated analysis of traffic hazard perception  Artificial neural networks: methods and applications in bio-/neuroinformatics  411  434  Springer 
[^B98]: Kathmann N . , Hochrein A . , Uwer R .  1999  Effects of dual task demands on the accuracy of smooth pursuit eye movements  Psychophysiology  158  163  36 
[^B99]: Katsanis J . , Iacono W .   G . , Harris M .  1998  Development of oculomotor functioning in preadolescence, adolescence, and adulthood  Psychophysiology  64  72  35 
[^B100]: Kerzel D . , Born S . , Souto D .  2009  Smooth pursuit eye movements and perception share target selection, but only some central resources  Behav. brain Res.  66  73  201 
[^B101]: Khurana B . , Kowler E .  1987  Shared attentional control of smooth eye movement and perception  Vis. Res.  1603  1618  27 
[^B102]: Klein C . , Ettinger U .  2019  Eye movement research: an introduction to its scientific foundations and applications  Springer Nature 
[^B103]: Klin A . , Jones W . , Schultz R . , Volkmar F . , Cohen D .  2002  Visual fixation patterns during viewing of naturalistic social situations as predictors of social competence in individuals with autism  Archives general psychiatry  809  816  59 
[^B104]: Ko H . - k . , Snodderly D .   M . , Poletti M .  2016  Eye movements between saccades: measuring ocular drift and tremor  Vis. Res.  93  104  122 
[^B105]: Komogortsev O .   V . , Karpov A .  2013  Automated classification and scoring of smooth pursuit eye movements in the presence of fixations and saccades  Behav. Res. methods  203  215  45 
[^B106]: Komogortsev O .   V . , Khan J .   I .  2007  Kalman filtering in the design of eye-gaze-guided computer interfaces  International conference on human-computer interaction  679  689  Springer 
[^B107]: Komogortsev O .   V . , Gobert D .   V . , Jayarathna S . , Gowda S .   M .  2010a  Standardization of automated analyses of oculomotor fixation and saccadic behaviors  IEEE Trans. Biomed. Eng.  2635  2645  57 
[^B108]: Komogortsev O .   V . , Jayarathna S . , Koh D .   H . , Gowda S .   M .  2010b  Qualitative and quantitative scoring and evaluation of the eye movement classification algorithms  Proceedings of the 2010 Symposium on eye-tracking research and applications  65  68 
[^B109]: Kourtesis P .  2024  A comprehensive review of multimodal xr applications, risks, and ethical challenges in the metaverse  Multimodal Technol. Interact.  98  8 
[^B110]: Krauzlis R .   J .  2004  Recasting the smooth pursuit eye movement system  J. neurophysiology  591  603  91 
[^B111]: Krauzlis R . , Miles F .  1996  Decreases in the latency of smooth pursuit and saccadic eye movements produced by the “gap paradigm” in the monkey  Vis. Res.  1973  1985  36 
[^B112]: Krejtz K . , Duchowski A . , Krejtz I . , Szarkowska A . , Kopacz A .  2016a  Discerning ambient/focal attention with coefficient k  ACM Trans. Appl. Percept. (TAP)  1  20  13 
[^B113]: Krejtz K . , Duchowski A .   T . , Krejtz I . , Kopacz A . , Chrzastowski-Wachtel P .  2016b  Gaze transitions when learning with multimedia  J. Eye Mov. Res.  9 
[^B114]: Krejtz K . , Çöltekin A . , Duchowski A . , Niedzielska A .  2017  Using coefficient K to distinguish ambient/focal visual attention during map viewing  J. eye Mov. Res.  10 
[^B115]: Laborde Q . , Roques A . , Robert M .   P . , Armougum A . , Vayatis N . , Bargiotas I .  2025  Vision toolkit part 1. neurophysiological foundations and experimental paradigms in eye-tracking research: a review  Front. Physiology  1571534  1572025  16 
[^B116]: Lacquaniti F . , Terzuolo C . , Viviani P .  1983  The law relating the kinematic and figural aspects of drawing movements  Acta Psychol.  115  130  54 
[^B117]: Ladda J . , Eggert T . , Glasauer S . , Straube A .  2007  Velocity scaling of cue-induced smooth pursuit acceleration obeys constraints of natural motion  Exp. brain Res.  343  356  182 
[^B118]: Land M .   F .  2009  Vision, eye movements, and natural behavior  Vis. Neurosci.  51  62  26 
[^B119]: Lebedev S . , Van Gelder P . , Tsui W .   H .  1996  Square-root relations between main saccadic parameters  Investigative Ophthalmol. and Vis. Sci.  2750  2758  37 
[^B120]: Leech J . , Gresty M . , Hess K . , Rudge P .  1977  Gaze failure, drifting eye movements, and centripetal nystagmus in cerebellar disease  Br. J. Ophthalmol.  774  781  61 
[^B121]: Leigh R .   J . , Zee D .   S .  2015  The neurology of eye movements 
[^B122]: Lencer R . , Trillenberg P .  2008  Neurophysiology and neuroanatomy of smooth pursuit in humans  Brain cognition  219  228  68 
[^B123]: Leonard B . , Zhang M . , Snyder V . , Holland C . , Bensinger E . , Sheehy C .   K .  2021  Fixational eye movements following concussion  Investigative Ophthalmol. and Vis. Sci.  1035  60 
[^B124]: Li B . , Wang Q . , Barney E . , Hart L . , Wall C . , Chawarska K .  2016  Modified dbscan algorithm on oculomotor fixation identification  337  338  Association for Computing Machinery 
[^B125]: Lin H . - H . , Chen Y . - F . , Chen T . , Tsai T . - T . , Huang K . - H .  2004  Temporal analysis of the acceleration and deceleration phases for visual saccades  Biomed. Eng. Appl. Basis Commun.  355  362  16 
[^B126]: Liu P . - L .  2014  Using eye tracking to understand learners’ reading process through the concept-mapping learning strategy  Comput. and Educ.  237  249  78 
[^B127]: Liu H . - C . , Chuang H . - H .  2011  An examination of cognitive processing of multimedia information based on viewers’ eye movements  Interact. Learn. Environ.  503  517  19 
[^B128]: Liu J . - C . , Li K . - A . , Yeh S . - L . , Chien S . - Y .  2022  Assessing perceptual load and cognitive load by fixation-related information of eye movements  Sensors  1187  22 
[^B129]: Liversedge S . , Gilchrist I . , Everling S .  2011  The Oxford handbook of eye movements  OUP 
[^B130]: Lopez J .   S .   A .  2009  Off-the-shelf gaze interaction  Ph.D. thesis 
[^B131]: Ludwig C .   J . , Gilchrist I .   D .  2002  Measuring saccade curvature: a curve-fitting approach  Behav. Res. Methods, Instrum. and Comput.  618  624  34 
[^B132]: MacAskill M .   R . , Anderson T .   J .  2016  Eye movements in neurodegenerative diseases  Curr. Opin. neurology  61  68  29 
[^B133]: MacAskill M .   R . , Anderson T .   J . , Jones R .   D .  2002  Adaptive modification of saccade amplitude in Parkinson’s disease  Brain  1570  1582  125 
[^B134]: Mahanama B . , Jayawardana Y . , Rengarajan S . , Jayawardena G . , Chukoskie L . , Snider J .  2022a  Eye movement and pupil measures: a review  Front. Comput. Sci.  733531  3 
[^B135]: Mahanama B . , Jayawardana Y . , Rengarajan S . , Jayawardena G . , Chukoskie L . , Snider J .  2022b  Eye movement and pupil measures: a review  Front. Comput. Sci.  733531  3 
[^B136]: Martinez-Conde S . , Macknik S .   L . , Troncoso X .   G . , Hubel D .   H .  2009  Microsaccades: a neurophysiological analysis  Trends Neurosci.  463  475  32 
[^B137]: May J .   G . , Kennedy R .   S . , Williams M .   C . , Dunlap W .   P . , Brannan J .   R .  1990  Eye movement indices of mental workload  Acta Psychol.  75  89  75 
[^B138]: McGillem C .   D . , Cooper G .   R .  1991  Continuous and discrete signal and system analysis 
[^B139]: McPeek R .   M . , Han J .   H . , Keller E .   L .  2003  Competition between saccade goals in the superior colliculus produces saccade curvature  J. neurophysiology  2577  2590  89 
[^B140]: Megaw E .   D . , Richardson J .  1979  Eye movements and industrial inspection  Appl. Ergon.  145  154  10 
[^B141]: Metz H .   S . , Scott A .   B . , O’Meara D . , Stewart H .   L .  1970  Ocular saccades in lateral rectus palsy  Archives Ophthalmol.  453  460  84 
[^B142]: Meyer C .   H . , Lasker A .   G . , Robinson D .   A .  1985  The upper limit of human smooth pursuit velocity  Vis. Res.  561  563  25 
[^B143]: Michell A . , Xu Z . , Fritz D . , Lewis S . , Foltynie T . , Williams-Gray C .  2006  Saccadic latency distributions in Parkinson’s disease and the effects of l-dopa  Exp. brain Res.  7  18  174 
[^B144]: Miles W .   R .  1929  Horizontal eye movements at the onset of sleep  Psychol. Rev.  122  141  36 
[^B145]: Montesano G . , Crabb D .   P . , Jones P .   R . , Fogagnolo P . , Digiuni M . , Rossetti L .   M .  2018  Evidence for alterations in fixational eye movements in glaucoma  BMC Ophthalmol.  191  198  18 
[^B146]: Moschovakis A .   K .  1996  The superior colliculus and eye movement control  Curr. Opin. Neurobiol.  811  816  6 
[^B147]: Moshel S . , Zivotofsky A .   Z . , Liang J .   R . , Engbert R . , Kurths J . , Kliegl R .   e .   a .  2008  Persistence and phase synchronisation properties of fixational eye movements  Eur. Phys. J. Special Top.  207  223  161 
[^B148]: Mozaffari S . , Al-Naser M . , Klein P . , Küchemann S . , Kuhn J . , Widmann T .  2020  Classification of visual strategies in physics vector field problem-solving  ICAART  257  267  2020 
[^B149]: Murray N .   G . , Szekely B . , Islas A . , Munkasy B . , Gore R . , Berryhill M .  2020  Smooth pursuit and saccades after sport-related concussion  J. neurotrauma  340  346  37 
[^B150]: Nakayama M . , Shimizu Y .  2004  Frequency analysis of task evoked pupillary response and eye-movement  Proceedings of the 2004 symposium on Eye tracking research and applications  71  76 
[^B151]: Nakayama M . , Takahashi K . , Shimizu Y .  2002  The act of task difficulty and eye-movement frequency for the oculo-motor indices  Proceedings of the 2002 symposium on Eye tracking research and applications  37  42 
[^B152]: Nazir T .   A . , Jacobs A .   M . , O’Regan J .   K .  1998  Letter legibility and visual word recognition  Mem. and cognition  810  821  26 
[^B153]: Nyström M . , Holmqvist K .  2010  An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking data  Behav. Res. Methods  188  204  42 
[^B154]: Onuchin A . , Kachan O .  2023  Individual topology structure of eye movement trajectories  International conference on neuroinformatics  45  55  Springer 
[^B155]: O’Driscoll G .   A . , Callahan B .   L .  2008  Smooth pursuit in schizophrenia: a meta-analytic review of research since 1993  Brain cognition  359  370  68 
[^B156]: O’Regan J .   K . , Jacobs A .   M .  1992  Optimal viewing position effect in word recognition: a challenge to current theory  J. Exp. Psychol. Hum. Percept. Perform.  185  197  18 
[^B157]: Park B . , Korbach A . , Brünken R .  2015  Do learner characteristics moderate the seductive-details-effect? a cognitive-load-study using eye-tracking  J. Educ. Technol. and Soc.  24  36  18 
[^B158]: Peterson M .   F . , Eckstein M .   P .  2012  Looking just below the eyes is optimal across face recognition tasks  Proc. Natl. Acad. Sci.  E3314  E3323  109 
[^B159]: Phillips M .   H . , Edelman J .   A .  2008  The dependence of visual scanning performance on search direction and difficulty  Vis. Res.  2184  2192  48 
[^B160]: Pincus S . , Gladstone I . , Ehrenkranz R .  1991  A regularity statistic for medical data analysis  J. Clin. Monit. Comput.  335  345  7 
[^B161]: Rabiner L .   R .  1978  Digital processing of speech signals  Pearson Education India 
[^B162]: Ramat S . , Leigh R .   J . , Zee D .   S . , Optican L .   M .  2007  What clinical disorders tell us about the neural control of saccadic eye movements  Brain  10  35  130 
[^B163]: Raney G .   E . , Campbell S .   J . , Bovee J .   C .  2014  Using eye movements to evaluate the cognitive processes involved in text comprehension  JoVE J. Vis. Exp.  e50780 
[^B164]: Rashbass C .  1961  The relationship between saccadic and smooth tracking eye movements  J. physiology  326  338  159 
[^B165]: Rayner K .  1998  Eye movements in reading and information processing: 20 years of research  Psychol. Bull.  372  422  124 
[^B167]: Rayner K . , Pollatsek A . , Ashby J . , Clifton Jr C .  2012  Psychology of reading  Psychology Press 
[^B168]: Rigas I . , Komogortsev O . , Shadmehr R .  2016  Biometric recognition via eye movements: saccadic vigor and acceleration cues  ACM Trans. Appl. Percept.  1  21  13 
[^B169]: Rigas I . , Friedman L . , Komogortsev O .  2018  Study of an extensive set of eye movement features: extraction methods and statistical analysis  J. Eye Mov. Res.  11 
[^B170]: Ritchie L .  1976  Effects of cerebellar lesions on saccadic eye movements  J. neurophysiology  1246  1256  39 
[^B171]: Robert M .   P . , Ingster-Moati I . , Albuisson E . , Cabrol D . , Golse B . , Vaivre-Douret L .  2014  Vertical and horizontal smooth pursuit eye movements in children with developmental coordination disorder  Dev. Med. and Child Neurology  595  600  56 
[^B172]: Robinson D .   A .  1965  The mechanics of human smooth pursuit eye movement  J. Physiology  569  591  180 
[^B173]: Robinson D .   A . , Gordon J . , Gordon S .  1986  A model of the smooth pursuit eye movement system  Biol. Cybern.  43  57  55 
[^B174]: Rottach K .   G . , Zivotofsky A .   Z . , Das V .   E . , Averbuch-Heller L . , Discenna A .   O . , Poonyathalang A .  1996  Comparison of horizontal, vertical and diagonal smooth pursuit eye movements in normal human subjects  Vis. Res.  2189  2195  36 
[^B175]: Saeb S . , Weber C . , Triesch J .  2011  Learning the optimal control of coordinated eye and head movements  PLoS Comput. Biol.  e1002253  7 
[^B176]: Salvucci D .   D . , Goldberg J .   H .  2000  Identifying fixations and saccades in eye-tracking protocols  Proceedings of the 2000 symposium on eye tracking research and applications  71  78  Association for Computing Machinery 
[^B177]: Santini T . , Fuhl W . , Kübler T . , Kasneci E .  2016  Bayesian identification of fixations, saccades, and smooth pursuits  Proceedings of the ninth biennial ACM symposium on eye tracking research and applications  163  170 
[^B178]: Sauter D . , Martin B . , Di Renzo N . , Vomscheid C .  1991  Analysis of eye tracking movements using innovations generated by a kalman filter  Med. Biol. Eng. Comput.  63  69  29 
[^B179]: Scheiter K . , Eitel A .  2017  The use of eye tracking as a research and instructional tool in multimedia learning  
Eye-tracking technology applications in educational research (IGI Global)  143  164 
[^B180]: Schmidt D . , Abel L . , DellOsso L . , Daroff R .  1979  Saccadic velocity characteristics-intrinsic variability and fatigue  Aviat. space, Environ. Med.  393  395  50 
[^B181]: Schmitt L .   M . , Cook E .   H . , Sweeney J .   A . , Mosconi M .   W .  2014  Saccadic eye movement abnormalities in autism spectrum disorder indicate dysfunctions in cerebellum and brainstem  Mol. autism  47  13  5 
[^B182]: Schoonahd J .   W . , Gould J .   D . , Miller L .   A .  1973  Studies of visual inspection  Ergonomics  365  379  16 
[^B183]: Schor C .   M . , Westall C .  1984  Visual and vestibular sources of fixation instability in amblyopia  Investigative Ophthalmol. and Vis. Sci.  729  738  25 
[^B184]: Schütz A .   C . , Braun D .   I . , Gegenfurtner K .   R .  2011  Eye movements and perception: a selective review  J. Vis.  9  11 
[^B185]: Selhorst J .   B . , Stark L . , Ochs A .   L . , Hoyt W .   F .  1976  Disorders in cerebellar ocular motor control. i. saccadic overshoot dysmetria. an oculographic, control system and clinico-anatomical analysis  Brain a J. neurology  497  508  99 
[^B186]: Shaikh A .   G . , Otero-Millan J . , Kumar P . , Ghasia F .   F .  2016  Abnormal fixational eye movements in amblyopia  PloS one  e0149953  11 
[^B187]: Sharafi Z . , Shaffer T . , Sharif B . , Guéhéneuc Y . - G .  2015  Eye-tracking metrics in software engineering  2015 asia-pacific software engineering conference (APSEC)  96  103  IEEE 
[^B188]: Sharif B . , Falcone M . , Maletic J .   I .  2012  An eye-tracking study on the role of scan time in finding source code defects  Proceedings of the symposium on eye tracking research and applications  381  384 
[^B189]: Sheliga B .   M . , Riggio L . , Craighero L . , Rizzolatti G .  1995  Spatial attention-determined modifications in saccade trajectories  Neuroreport An Int. J. Rapid Commun. Res. Neurosci.  585  588  6 
[^B190]: Sheliga B . , Craighero L . , Riggio L . , Rizzolatti G .  1997  Effects of spatial attention on directional manual and ocular responses  Exp. brain Res.  339  351  114 
[^B191]: Shic F . , Scassellati B . , Chawarska K .  2008  The incomplete fixation measure  Proceedings of the 2008 symposium on Eye tracking research and applications  111  114 
[^B192]: Shirama A . , Kanai C . , Kato N . , Kashino M .  2016  Ocular fixation abnormality in patients with autism spectrum disorder  J. Autism Dev. Disord.  1613  1622  46 
[^B193]: Skaramagkas V . , Giannakakis G . , Ktistakis E . , Manousos D . , Karatzanis I . , Tachos N .   S .  2021  Review of eye tracking metrics involved in emotional and cognitive processes  IEEE Rev. Biomed. Eng.  260  277  16 
[^B194]: Smit A . , Van Gisbergen J .  1990  An analysis of curvature in fast and slow human saccades  Exp. Brain Res.  335  345  81 
[^B195]: Smit A . , Van Gisbergen J . , Cools A .  1987  A parametric analysis of human saccades in different experimental paradigms  Vis. Res.  1745  1762  27 
[^B196]: Snyder L .   H . , Calton J .   L . , Dickinson A .   R . , Lawrence B .   M .  2002  Eye-hand coordination: saccades are faster when accompanied by a coordinated arm movement  J. neurophysiology  2279  2286  87 
[^B197]: Souto D . , Kerzel D .  2014  Ocular tracking responses to background motion gated by feature-based attention  J. neurophysiology  1074  1081  112 
[^B198]: Sparks D .   L .  1986  Translation of sensory signals into commands for control of saccadic eye movements: role of primate superior colliculus  Physiol. Rev.  118  171  66 
[^B199]: Spering M .  2022  Eye movements as a window into decision-making  Annu. Rev. Vis. Sci.  427  448  8 
[^B200]: Spering M . , Gegenfurtner K .   R .  2007  Contextual effects on smooth-pursuit eye movements  J. Neurophysiology  1353  1367  97 
[^B201]: Startsev M . , Zemblys R .  2023  Evaluating eye movement event detection: a review of the state of the art  Behav. Res. Methods  1653  1714  55 
[^B202]: Steinman R .   M .  1965  Effect of target size, luminance, and color on monocular fixation  JOSA  1158  1164  55 
[^B203]: Steinman R .   M . , Kowler E . , Collewijn H .  1990  New directions for oculomotor research  Vis. Res.  1845  1864  30 
[^B204]: Stoica P . , Moses R .   L .  2005  Spectral analysis of signals  452  Pearson Prentice Hall 
[^B205]: Stolte M . , Kraus L . , Ansorge U .  2023  Visual attentional guidance during smooth pursuit eye movements: distractor interference is independent of distractor-target similarity  Psychophysiology  e14384  60 
[^B206]: Stubbs J .   L . , Corrow S .   L . , Kiang B . , Panenka W .   J . , Barton J .   J .  2018  The effects of enhanced attention and working memory on smooth pursuit eye movement  Exp. brain Res.  485  495  236 
[^B207]: Troost B .   T . , Daroff R .   B .  1977  The ocular motor defects in progressive supranuclear palsy  Ann. Neurology Official J. Am. Neurological Assoc. Child Neurology Soc.  397  403  2 
[^B208]: Tychsen L . , Lisberger S .   G .  1986  Visual motion processing for the initiation of smooth-pursuit eye movements in humans  J. neurophysiology  953  968  56 
[^B209]: Underwood G . , Binns A . , Walker S .  2000  Attentional demands on the processing of neighbouring words  Reading as a perceptual process  247  268  Elsevier 
[^B210]: Van Donkelaar P . , Drew A .   S .  2002  The allocation of attention during smooth pursuit eye movements  Prog. brain Res.  267  277  140 
[^B211]: Van Gelder P . , Lebedev S . , Liu P .   M . , Tsui W .   H .  1995  Anticipatory saccades in smooth pursuit: task effects and pursuit vector after saccades  Vis. Res.  667  678  35 
[^B212]: Van Gisbergen J . , Van Opstal A . , Roebroek J .  1987  Stimulus-induced midflight modification of saccade trajectories  Eye movements from physiology to cognition  27  36  Elsevier 
[^B213]: van Opstal A .   J . , Goossens H .  2008  Linear ensemble-coding in midbrain superior colliculus specifies the saccade kinematics  Biol. Cybern.  561  577  98 
[^B214]: Van Opstal A . , Van Gisbergen J .  1987  Skewness of saccadic velocity profiles: a unifying parameter for normal and slow saccades  Vis. Res.  731  745  27 
[^B215]: Van Orden K .   F . , Jung T .   P . , Makeig S .  2000  Combined eye activity measures accurately estimate changes in sustained visual task performance  Biol. Psychol.  221  240  52 
[^B216]: Viviani P .  1977  The curvature of oblique saccades  curvature oblique saccades. Vis. Res  661  664  17 
[^B217]: Viviani P . , Swensson R .   G .  1982  Saccadic eye movements to peripherally discriminated visual targets  J. Exp. Psychol. Hum. Percept. Perform.  113  126  8 
[^B218]: von Wartburg R . , Wurtz P . , Pflugshaupt T . , Nyffeler T . , Lüthi M . , Müri R .   M .  2007  Size matters: saccades during scene perception  Perception  355  365  36 
[^B219]: Vullings C .  2018  Saccadic latencies depend on functional relations with the environment 
[^B220]: Walker R . , McSorley E . , Haggard P .  2006  The control of saccade trajectories: direction of curvature depends on prior knowledge of target location and saccade latency  Percept. and Psychophys.  129  138  68 
[^B221]: Wang Y . , Cong W .  2015  Statistical analysis of air traffic controllers’ eye movements  In conference: air traffic management R&D Seminar 
[^B222]: Welch P .  1967  The use of fast fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms  IEEE Trans. Audio Electroacoustics  70  73  15 
[^B223]: Wetzel P .   A . , Gitchel G .   T . , Baron M .   S .  2011  Effect of Parkinson’s disease on eye movements during reading  Investigative Ophthalmol. and Vis. Sci.  4697  52 
[^B224]: Whelan R .  2008  Effective analysis of reaction time data  Psychol. Rec.  475  482  58 
[^B225]: Xu-McGregor D .   K . , Stern J .   A .  1996  Time on task and blink effects on saccade duration  Ergonomics  649  660  39 
[^B226]: Xu-Wilson M . , Zee D .   S . , Shadmehr R .  2009  The intrinsic value of visual information affects saccade velocities  Exp. Brain Res.  475  481  196 
[^B227]: Yarbus A .   L . , Yarbus A .   L .  1967  Eye movements during perception of complex objects  Eye movements Vis.  171  211 
[^B228]: Yee R .   D . , Cogan D .   G . , Zee D .   S . , Baloh R .   W . , Honrubia V .  1976  Rapid eye movements in myasthenia gravis: ii. electro-oculographic analysis  Archives Ophthalmol.  1465  1472  94 
[^B229]: Zackon D .   H . , Sharpe J .   A .  1987  Smooth pursuit in senescence: effects of target acceleration and velocity  Acta oto-laryngologica  290  297  104 
[^B230]: Zemblys R . , Niehorster D .   C . , Komogortsev O . , Holmqvist K .  2018  Using machine learning to detect events in eye-tracking data  Behav. Res. methods  160  181  50 
[^B231]: Zuber B . , Semmlow J . , Stark L .  1968  Frequency characteristics of the saccadic eye movement  Biophysical J.  1288  1298  8 

