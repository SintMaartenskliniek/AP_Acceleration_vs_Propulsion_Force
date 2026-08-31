# Lumbar Acceleration IMU
Code used for the study Lumbar Acceleration IMU. This work was supported by Interreg Deutschland-Nederland grant for LifeHelper: Real-time movement monitoring - anywhere and anytime (project number 13153).

<br>

**FULL ARTICLE UNDER SUBMISSION**: Lars van Rengs, Katrijn Smulders, Brenda E. Groen, and Noël L. W. Keijsers: Can velocity increment derived from a single, lumbar-mounted inertial measurement unit serve as a surrogate for propulsion force impulse in healthy and post-stroke individuals?

<br>

This repository contains algorithms and datasets for gait analysis using both IMU (Inertial Measurement Unit) and OMCS (Optical Motion Capture System) data.
It includes validated algorithms, walking trial datasets from healthy individuals and post-stroke individuals, as well as scripts for data analysis.

<br>

**📂 IMU_GaitAnalysis [1]**

Contains the IMU-based gait analysis algorithm developed and validated by C. Ensink.
This folder also includes datasets from various walking trials with 20 healthy individuals and 10 post-stroke individuals.

<br>

**📂 MovingReality [2]**

Contains datasets from walking trials with 12 post-stroke individuals.

<br>

**📂 OMCS_GaitAnalysis**

Contains the algorithm for OMCS-based gait analysis.

<br>

**📂 Statistics**

Contains RStudio-script and dataset used for data analysis of this study.

<br>

**📄 OMCS_StepByStepData.py**

Creates a dataframe with OMCS gait data for the left and right leg (df_left.pkl and df_right.pkl) and saves them to the working directory.
* Each row represents a step of a subject.
* Spatiotemporal metrics are organized in columns.
* Steps are linked using initial contact (IC) and terminal contact (TC) events, enabling step-by-step comparison between legs.

<br>

**📄 main_LumbarAccelerationIMU.py**

Extends the OMCS step-by-step dataframe by adding anterior-posterior acceleration measures derived from OMCS and IMU data.

<br>

**📄 helpers_LumbarAccelerationIMU.py**

Contains helper functions that are used in main_LumbarAcceleration.py.

<br>

**References**

[1] Ensink, C., Smulders, K., Warnar, J., & Keijsers, N. (2023). Validation of an algorithm to assess regular and irregular gait using inertial sensors in healthy and stroke individuals. PeerJ, 11, e16641. https://doi.org/10.7717/peerj.16641

[2] Ensink, C., Hofstad, C., Ee, R., & Keijsers, N. (2025, 01/24). Effect of Feedback on Foot Strike Angle and Forward Propulsion in People With Stroke. IEEE Transactions on Neural Systems and Rehabilitation Engineering, PP, 1-1. https://doi.org/10.1109/TNSRE.2025.3533748
