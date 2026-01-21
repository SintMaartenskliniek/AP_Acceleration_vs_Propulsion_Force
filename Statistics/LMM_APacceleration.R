## ---------------------------
##
## Script name:   LMM_APacceleration
##
## Purpose:       Linear mixed model to evaluate the association between propulsion force impulse (OMCS) and anterior-posterior
##                acceleration impulse (IMU) in healthy and post-stroke individuals (paretic and non-paretic leg).
##
## Author:        Lars van Rengs - l.vanrengs@maartenskliniek.nl
##
## Created:       2025
## 
## ---------------------------


## Load required packages
library(tidyverse)
library(readxl)
library(jtools)
library(lme4)
library(rstatix)
library(MuMIn)
library(patchwork)
library(Polychrome)
library(ggpubr)


## ---------------------------
## Load & Prepare Data
## ---------------------------

# Load dataset
filename = "R_dataset_APacceleration.xlsx"
Data <- read_excel(filename, sheet = "Data")

# Define step conditions
step_conditions <- c(1, 3, 5)

# Define groups
groups <- list(
  "Healthy" = "Healthy",
  "Paretic" = "Paretic",
  "NonParetic" = "NonParetic"
)

# Filter dataset to include only the regular walking trials
regular_walking_trials <- c(
  "900_V_pp01_SP01.c3d",
  "900_V_pp03_SP01.c3d",
  "900_V_pp04_SP01.c3d",
  "900_V_pp05_SP01.c3d",
  "900_V_pp06_SP01.c3d",
  "900_V_pp07_SP01.c3d",
  "900_V_pp08_SP02.c3d",
  "900_V_pp09_SP01.c3d",
  "900_V_pp10_SP01.c3d",
  "900_V_pp11_SP01.c3d",
  "900_V_pp12_SP01.c3d",
  "900_V_pp13_SP01.c3d",
  "900_V_pp14_SP01.c3d",
  "900_V_pp15_SP01.c3d",
  "900_V_pp16_SP01.c3d",
  "900_V_pp18_SP01.c3d",
  "900_V_pp19_SP01.c3d",
  "900_V_pp20_SP01.c3d",
  "900_V_pp21_SP01.c3d",
  "900_V_pp22_SP01.c3d",
  "900_CVA_01_SP01.c3d",
  "900_CVA_pp02_SP01.c3d",
  "900_CVA_03_FS01.c3d",
  "900_CVA_04_SP01.c3d",
  "900_CVA_05_SP01.c3d",
  "900_CVA_06_SP01.c3d",
  "900_CVA_07_SP01.c3d",
  "900_CVA_08_SP01.c3d",
  "900_CVA_09_FS01.c3d",
  "900_CVA_10_SP01.c3d",
  "1019_MR001_1Reg.c3d",
  "1019_MR002_Reg.c3d",
  "1019_MR003_1Reg02.c3d",
  "1019_MR004_1Reg.c3d",
  "1019_MR005_1Reg01.c3d",
  "1019_MR006_1Reg.c3d",
  "1019_MR007_1Reg02.c3d",
  "1019_MR008_1Reg02.c3d",
  "1019_MR009_1Reg.c3d",
  "1019_MR010_1Reg.c3d",
  "1019_MR011_1Reg.c3d",
  "1019_MR012_1Reg.c3d"
)
Data <- Data %>%
  filter(basename(trial) %in% regular_walking_trials)

# Add a 'Group' column
Data <- Data %>%
  mutate(
    Group = case_when(
      Data$group_name == "Healthy_controls" ~ "Healthy",
      
      Data$group_name == "CVA" & Data$affected_leg == "left" & Data$leg == "left" ~ "Paretic",
      Data$group_name == "CVA" & Data$affected_leg == "left" & Data$leg == "right" ~ "NonParetic",
      
      Data$group_name == "CVA" & Data$affected_leg == "right" & Data$leg == "right" ~ "Paretic",
      Data$group_name == "CVA" & Data$affected_leg == "right" & Data$leg == "left" ~ "NonParetic",

      Data$group_name == "CVA_feedback" & Data$affected_leg == "left" & Data$leg == "left" ~ "Paretic",
      Data$group_name == "CVA_feedback" & Data$affected_leg == "left" & Data$leg == "right" ~ "NonParetic",

      Data$group_name == "CVA_feedback" & Data$affected_leg == "right" & Data$leg == "right" ~ "Paretic",
      Data$group_name == "CVA_feedback" & Data$affected_leg == "right" & Data$leg == "left" ~ "NonParetic",
      
      TRUE ~ NA_character_
    )
  )

# Filter out problematic or irrelevant trials
exclude_trials <- c(
  "900_CVA_03_FS01.c3d",        # Fixed-speed instead of self-paced
  "900_CVA_03_FS02.c3d",        # 900_CVA_03 was unable to perform stepping stones trial --> This person performed 2 regular walking trials, remove one for further analysis ('900_CVA_03_FS02.c3d')
  "900_CVA_04_SP01.c3d",        # 900_CVA_04_SP01.c3d >> walking mostly on one of the treadmill bands, not viable for gait event detection.
  "900_CVA_09_FS01.c3d",        # 900_CVA_09 had to perform regular walking condition at a fixed treadmill speed ('900_CVA_09_FS01.c3d'); all other participants performed regular walking trial in self-paced mode
  "900_V_pp01_SP03.c3d",        # '900_V_pp01_SP03.c3d': Fixed speed trial, accidentally wrongly named
  "900_V_pp07_SP01.c3d",        # 900_V_pp07_SP01.c3d --> OMCS data is missing
  "1019_MR003_FBIC.c3d",        # 1019_MR003_FBIC.c3d --> No Xsens data available; recording error
  "1019_MR006_FBIC.c3d",        # 1019_MR006_FBIC was removed from further analysis due to poor marker visibility
  "1019_MR006_2Reg02.c3d"       # 1019_MR006_2Reg02 --> Xsens data; recording error
)
Data <- Data %>%
  filter(!(basename(trial) %in% exclude_trials))

# Exclude double-included subjects
exclude_subjects <- c(
  "900_CVA_08"                  # 900_CVA_08 is the same person as 1019_pp07
)
Data <- Data %>%
  filter(!(subject_name %in% exclude_subjects))

# Filter healthy individuals --> only include one leg per participant to ensure consistent representation
healthy_left_leg <- c("900_V_01", "900_V_03", "900_V_05", "900_V_07", "900_V_09", "900_V_11", "900_V_13", "900_V_15", "900_V_17", "900_V_19", "900_V_21")
healthy_right_leg  <- c("900_V_02", "900_V_04", "900_V_06", "900_V_08", "900_V_10", "900_V_12", "900_V_14", "900_V_16", "900_V_18", "900_V_20", "900_V_22")
Data <- Data %>%
  filter(
    !(group_name == "Healthy_controls" & (
      (subject_name %in% healthy_left_leg & leg != "left") |
        (subject_name %in% healthy_right_leg & leg != "right")
    ))
  )

# Select and rename relevant columns for analysis
Data_selected <- Data %>%
  select(subject_name, trial, Group, leg, IMU_APAccelerationSacrumImpulse_Value, PropulsionImpulse_Value) %>%
  rename(
    SubjectID = subject_name,
    Trial = trial,
    Group = Group,
    Leg = leg,
    APacceleration = IMU_APAccelerationSacrumImpulse_Value,
    PropulsionForce = PropulsionImpulse_Value
  ) %>%
  filter(!is.na(APacceleration) & !is.na(PropulsionForce))

# Get all combinations before filtering by step count
All_combinations <- Data_selected %>%
  group_by(SubjectID, Trial, Group, Leg) %>%
  summarise(StepCount = n(), .groups = "drop")

# Filter data to only include combinations with at least 50 steps (>= 10 steps in 5-step condition)
Data_selected <- Data_selected %>%
  group_by(SubjectID, Trial, Group, Leg) %>%
  filter(n() >= 50) %>%
  ungroup()

# Get combinations with >= 50 steps
Kept_combinations <- Data_selected %>%
  group_by(SubjectID, Trial, Group, Leg) %>%
  summarise(StepCount = n(), .groups = "drop")

# Get combinations with < 50 steps (i.e., removed)
Removed_combinations <- anti_join(All_combinations, Kept_combinations,
                                  by = c("SubjectID", "Trial", "Group", "Leg"))

# Count available number of steps
Number_of_Steps <- Data_selected %>%
  group_by(SubjectID, Trial, Group, Leg) %>%
  summarise(n = n(), .groups = "drop")

Number_of_Steps_summary <- Number_of_Steps %>%
  group_by(Group) %>%
  summarise(
    N_individuals = n(),
    median_steps = round(median(n), 0),
    Q1 = round(quantile(n, 0.25), 0),
    Q3 = round(quantile(n, 0.75), 0),
    IQR = round(IQR(n), 0)
  )

# Extract additional parameters from included steps
All_Data_selected <- Data %>%
  semi_join(Kept_combinations, 
            by = c("subject_name" = "SubjectID", 
                   "trial" = "Trial", 
                   "Group" = "Group", 
                   "leg" = "Leg")) %>%
  filter(!is.na(IMU_APAccelerationSacrumImpulse_Value) & 
           !is.na(PropulsionImpulse_Value))

Summary_parameters_per_subject <- All_Data_selected %>%
  group_by(Group, subject_name, trial) %>%
  summarise(
    StanceTime_mean = mean(StanceTime_Value, na.rm = TRUE),
    StanceTime_sd   = sd(StanceTime_Value, na.rm = TRUE),
    
    SwingTime_mean  = mean(SwingTime_Value, na.rm = TRUE),
    SwingTime_sd    = sd(SwingTime_Value, na.rm = TRUE),
    
    GaitCycleDuration_mean  = mean(GaitCycleDuration_Value, na.rm = TRUE),
    GaitCycleDuration_sd    = sd(GaitCycleDuration_Value, na.rm = TRUE),
    
    GaitSpeedStride_mean  = mean(GaitSpeedStride_Value, na.rm = TRUE),
    GaitSpeedStride_sd    = sd(GaitSpeedStride_Value, na.rm = TRUE),
    
    StepLength_mean = mean(StepLength_Value, na.rm = TRUE),
    StepLength_sd   = sd(StepLength_Value, na.rm = TRUE),
    
    StrideLength_mean = mean(StrideLength_Value, na.rm = TRUE),
    StrideLength_sd   = sd(StrideLength_Value, na.rm = TRUE),
    
    StepWidth_mean = mean(StepWidth_Value, na.rm = TRUE),
    StepWidth_sd   = sd(StepWidth_Value, na.rm = TRUE),
    
    BrakingImpulse_mean = mean(BrakingImpulse_Value, na.rm = TRUE),
    BrakingImpulse_sd   = sd(BrakingImpulse_Value, na.rm = TRUE),
    
    BrakingPeak_mean = mean(BrakingPeak_Value, na.rm = TRUE),
    BrakingPeak_sd   = sd(BrakingPeak_Value, na.rm = TRUE),
    
    PropulsionImpulse_mean = mean(PropulsionImpulse_Value, na.rm = TRUE),
    PropulsionImpulse_sd   = sd(PropulsionImpulse_Value, na.rm = TRUE),
    
    PropulsionPeak_mean = mean(PropulsionPeak_Value, na.rm = TRUE),
    PropulsionPeak_sd   = sd(PropulsionPeak_Value, na.rm = TRUE),
    
    aTLA_mean = mean(aTLA_Value, na.rm = TRUE),
    aTLA_sd   = sd(aTLA_Value, na.rm = TRUE),
    
    tcTLA_mean = mean(tcTLA_Value, na.rm = TRUE),
    tcTLA_sd   = sd(tcTLA_Value, na.rm = TRUE),
    
    maxTLA_mean = mean(maxTLA_Value, na.rm = TRUE),
    maxTLA_sd   = sd(maxTLA_Value, na.rm = TRUE),
    
    OMCS_APDecelerationSacrumImpulse_mean = mean(OMCS_APDecelerationSacrumImpulse_Value, na.rm = TRUE),
    OMCS_APDecelerationSacrumImpulse_sd   = sd(OMCS_APDecelerationSacrumImpulse_Value, na.rm = TRUE),
    
    OMCS_APDecelerationSacrumPeak_mean = mean(OMCS_APDecelerationSacrumPeak_Value, na.rm = TRUE),
    OMCS_APDecelerationSacrumPeak_sd   = sd(OMCS_APDecelerationSacrumPeak_Value, na.rm = TRUE),
    
    OMCS_APAccelerationSacrumImpulse_mean = mean(OMCS_APAccelerationSacrumImpulse_Value, na.rm = TRUE),
    OMCS_APAccelerationSacrumImpulse_sd   = sd(OMCS_APAccelerationSacrumImpulse_Value, na.rm = TRUE),
    
    OMCS_APAccelerationSacrumPeak_mean = mean(OMCS_APAccelerationSacrumPeak_Value, na.rm = TRUE),
    OMCS_APAccelerationSacrumPeak_sd   = sd(OMCS_APAccelerationSacrumPeak_Value, na.rm = TRUE),
    
    IMU_APDecelerationSacrumImpulse_mean = mean(IMU_APDecelerationSacrumImpulse_Value, na.rm = TRUE),
    IMU_APDecelerationSacrumImpulse_sd   = sd(IMU_APDecelerationSacrumImpulse_Value, na.rm = TRUE),
    
    IMU_APDecelerationSacrumPeak_mean = mean(IMU_APDecelerationSacrumPeak_Value, na.rm = TRUE),
    IMU_APDecelerationSacrumPeak_sd   = sd(IMU_APDecelerationSacrumPeak_Value, na.rm = TRUE),
    
    IMU_APAccelerationSacrumImpulse_mean = mean(IMU_APAccelerationSacrumImpulse_Value, na.rm = TRUE),
    IMU_APAccelerationSacrumImpulse_sd   = sd(IMU_APAccelerationSacrumImpulse_Value, na.rm = TRUE),
    
    IMU_APAccelerationSacrumPeak_mean = mean(IMU_APAccelerationSacrumPeak_Value, na.rm = TRUE),
    IMU_APAccelerationSacrumPeak_sd   = sd(IMU_APAccelerationSacrumPeak_Value, na.rm = TRUE),

    .groups = "drop"
  )

Summary_parameters_per_group <- Summary_parameters_per_subject %>%
  group_by(Group) %>%
  summarise(
    across(
      .cols = ends_with("_mean"),
      .fns = list(mean = ~mean(.x, na.rm = TRUE),
                  sd = ~sd(.x, na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )



## -----------------------------------------------------------------------------
## Generate datasets for 1-, 3-, and 5-step conditions
## -----------------------------------------------------------------------------

step_condition_data <- list()

for (w in step_conditions) {
  
  if (w == 1) {
    temp <- Data_selected %>%
      group_by(SubjectID, Trial, Group, Leg) %>%
      mutate(StrideNr_block = row_number()) %>%
      ungroup() %>%
      mutate(StepCondition = w) %>%
      select(SubjectID, Trial, Group, Leg, APacceleration, PropulsionForce, StepCondition, StrideNr_block)
    
  } else {
    temp <- Data_selected %>%
      group_by(SubjectID, Trial, Group, Leg) %>%
      mutate(StrideNr_block = row_number(),
             ConditionID = ceiling(StrideNr_block / w)) %>%
      group_by(SubjectID, Trial, Group, Leg, ConditionID) %>%
      filter(n() == w) %>%
      summarise(
        APacceleration = median(APacceleration, na.rm = TRUE),
        PropulsionForce = median(PropulsionForce, na.rm = TRUE),
        .groups = "drop_last"
      ) %>%
      mutate(
        StrideNr_block = row_number(),
        StepCondition = w
      ) %>%
      ungroup() %>%
      select(SubjectID, Trial, Group, Leg, APacceleration, PropulsionForce, StepCondition, StrideNr_block)
  }
  
  step_condition_data[[as.character(w)]] <- temp
}

# Combine all step condition datasets
Data_stepconditions <- bind_rows(step_condition_data) %>%
  mutate(StepCondition = factor(StepCondition))

# Create subsets for all step conditions
Dataset_full <- Data_stepconditions
Dataset_1StepCondition <- filter(Data_stepconditions, StepCondition == 1)
Dataset_3StepCondition <- filter(Data_stepconditions, StepCondition == 3)
Dataset_5StepCondition <- filter(Data_stepconditions, StepCondition == 5)



## -----------------------------------------------------------------------------
## Group-level and individual-level associations
## -----------------------------------------------------------------------------

cat("===== Group-level and individual-level associations =====\n")

Group_level_results <- list()
Individual_level_results <- list()

for (step in step_conditions) {
  
  dataset_name <- paste0("Dataset_", step, "StepCondition")
  data_all <- get(dataset_name)

  for (group in names(groups)) {
    cat("\n=== Step Condition:", step, "===\n")
    cat("\n--- Group:", group, "---\n")
    
    data_group <- filter(data_all, Group == groups[[group]])

    
    # === Group-level associations ===
    
    # Fit linear mixed model with random intercept and slope
    model <- lmer(PropulsionForce ~ APacceleration + (1 + APacceleration | SubjectID), data = data_group)
    
    # Residual diagnostics
    residuals_model <- resid(model)
    fitted_model <- fitted(model)
    
    par(mfrow = c(1, 3))
    
    ## Histogram of residuals
    hist(residuals_model,
         breaks = 20,
         main = paste("Residuals Histogram\n", group, "-", step, "step condition"),
         xlab = "Residuals", col = "lightblue", border = "white")
    
    ## Q-Q plot of residuals
    qqnorm(residuals_model,
           main = paste("Q-Q Plot\n", group, "-", step, "step condition"),
           pch = 20)
    qqline(residuals_model, col = "red", lwd = 2)
    
    ## Boxplot of residuals
    boxplot(residuals_model,
            horizontal = TRUE,
            main = paste("Boxplot of Residuals\n", group, "-", step, "step condition"),
            col = "lightgreen")    
    
    # Print model summary
    model_summary <- summ(model, confint = TRUE, digits = 3)
    print(model_summary)
    
    # Model info
    n_obs <- as.numeric(nobs(model))
    model_aic <- AIC(model)
    model_bic <- BIC(model)
    
    # Calculate marginal and conditional R²
    r2_vals <- r.squaredGLMM(model)

    # Fixed effects
    fixed_effects <- as.data.frame(model_summary$coeftable)
    colnames(fixed_effects) <- c("Estimate", "2.5%", "97.5%", "t", "df", "p")
    
    # Random effects
    re_varcorr <- VarCorr(model)
    re_stddevs <- as.data.frame(re_varcorr)[, c("grp", "var1", "sdcor")]
    sd_intercept <- re_stddevs$sdcor[re_stddevs$var1 == "(Intercept)"]
    sd_intercept <- sd_intercept[1]
    sd_slope <- re_stddevs$sdcor[re_stddevs$var1 == "APacceleration"]
    sd_slope <- sd_slope[1]
    sd_residual <- sigma(model)
    
    # ICC
    icc_numerator <- sd_intercept^2
    icc_denominator <- icc_numerator + sd_residual^2
    icc_value <- icc_numerator / icc_denominator

    # RMSE fixed-effect
    fixed_intercept <- fixef(model)["(Intercept)"]
    fixed_slope <- fixef(model)["APacceleration"]
    data_group <- data_group %>%
      mutate(
        FixedEffect_Predicted = fixed_intercept + fixed_slope * APacceleration,
        FixedEffect_Residual = PropulsionForce - FixedEffect_Predicted
      )
    group_level_RMSE <- data_group %>%
      summarise(
        N = n(),
        RSS = sum(FixedEffect_Residual^2, na.rm = TRUE),
        MSE = RSS / N,
        RMSE = sqrt(MSE),
        .groups = "drop"
      ) %>%
      pull(RMSE)  # Extract the scalar RMSE value
    
    # Group count
    n_subjects <- as.numeric(length(unique(data_group$SubjectID)))

    # Combine and save group-level results
    Group_level_results[[paste0(step, "_", group)]] <- tibble(
      StepCondition = step,
      Group = group,
      Observations = n_obs,
      AIC = model_aic,
      BIC = model_bic,
      R2_marginal = r2_vals[1],
      R2_conditional = r2_vals[2],
      Intercept_Est = fixed_effects["(Intercept)", "Estimate"],
      Intercept_CI_low = fixed_effects["(Intercept)", "2.5%"],
      Intercept_CI_high = fixed_effects["(Intercept)", "97.5%"],
      Intercept_t = fixed_effects["(Intercept)", "t"],
      Intercept_df = fixed_effects["(Intercept)", "df"],
      Intercept_p = fixed_effects["(Intercept)", "p"],
      Slope_Est = fixed_effects["APacceleration", "Estimate"],
      Slope_CI_low = fixed_effects["APacceleration", "2.5%"],
      Slope_CI_high = fixed_effects["APacceleration", "97.5%"],
      Slope_t = fixed_effects["APacceleration", "t"],
      Slope_df = fixed_effects["APacceleration", "df"],
      Slope_p = fixed_effects["APacceleration", "p"],
      Random_SD_Intercept = sd_intercept,
      Random_SD_Slope = sd_slope,
      Residual_SD = sd_residual,
      N_SubjectID = n_subjects,
      ICC_SubjectID = icc_value,
      Fixed_effect_RMSE = group_level_RMSE
    )
    
    
    # === Individual-level associations ===
    
    # Predict values from model
    data_group$Predicted <- predict(model)
    data_group$Residual <- data_group$PropulsionForce - data_group$Predicted
    
    # subject-specific R²
    subject_specific_r2 <- data_group %>%
      group_by(SubjectID) %>%
      summarise(
        LMM_TSS = sum((PropulsionForce - mean(PropulsionForce))^2, na.rm = TRUE),
        LMM_RSS = sum(Residual^2, na.rm = TRUE),
        LMM_R2_subject = 1 - (LMM_RSS / LMM_TSS),
        .groups = "drop"
      )
    
    # subject-specific RMSE
    subject_specific_RMSE <- data_group %>%
      group_by(SubjectID) %>%
      summarise(
        LMM_N = n(),
        LMM_RSS = sum(Residual^2, na.rm = TRUE),
        LMM_MSE = LMM_RSS / LMM_N,
        LMM_RMSE = sqrt(LMM_MSE),
        .groups = "drop"
      )
    
    # subject-specific slopes
    subject_specific_slopes <- coef(model)$SubjectID %>%
      rownames_to_column("SubjectID") %>%
      select(SubjectID, LMM_Intercept = `(Intercept)`, LMM_Slope = APacceleration)
    
    # Combine and save individual-level metrics
    subject_results <- subject_specific_r2 %>%
      left_join(subject_specific_RMSE, by = "SubjectID") %>%
      left_join(subject_specific_slopes, by = "SubjectID") %>%
      mutate(Group = group, StepCondition = step) %>%
      select(StepCondition, Group, SubjectID, everything())
    
    Individual_level_results[[paste0(step, "_", group)]] <- subject_results
  }
}

# Bind all rows together
output_Group_level_association <- bind_rows(Group_level_results)
output_Individual_level_association <- bind_rows(Individual_level_results)



## -----------------------------------------------------------------------------
## Effect of step aggregation
## -----------------------------------------------------------------------------

cat("\n===== Effect of step aggregation =====\n")

metrics <- c("LMM_R2_subject", "LMM_RMSE", "LMM_Slope")
friedman_results <- list()
posthoc_results <- list()

for (group in names(groups)) {
  cat("\n--- Group:", group, "---\n")
  
  # Subset for current group
  group_data <- output_Individual_level_association %>%
    filter(Group == groups[[group]]) %>%
    mutate(StepCondition = factor(StepCondition, levels = c("1", "3", "5")))
  
  for (metric in metrics) {
    cat("\nMetric:", metric, "\n")
    
    metric_data <- group_data %>%
      select(SubjectID, StepCondition, !!sym(metric)) %>%
      rename(Value = !!sym(metric))
    
    # Friedman test
    friedman_test_result <- metric_data %>%
      friedman_test(Value ~ StepCondition | SubjectID)
    
    print(friedman_test_result)
    friedman_results[[paste0(group, "_", metric)]] <- friedman_test_result %>%
      mutate(group = group, metric = metric)
    
    if (friedman_test_result$p < 0.05) {
      metric_data <- group_data %>%
        select(SubjectID, StepCondition, !!sym(metric)) %>%
        rename(Value = !!sym(metric)) %>%
        mutate(StepCondition = factor(StepCondition, levels = c("1", "3", "5"))) %>%
        ungroup()
      
      # Post-hoc Wilcoxon signed-rank test with Bonferroni correction
      posthoc <- pairwise_wilcox_test(
        data = metric_data,
        formula = Value ~ StepCondition,
        paired = TRUE,
        p.adjust.method = "bonferroni"
      )
      
      posthoc <- posthoc %>%
        mutate(group = group, metric = metric)
      
      print(posthoc)
      posthoc_results[[paste0(group, "_", metric)]] <- posthoc
    } else {
      cat("Friedman test not significant; skipping post-hoc tests.\n")
    }
  }
}

# Bind all rows together
output_Friedman_results <- bind_rows(friedman_results)
output_Wilcoxon_results <- bind_rows(posthoc_results)



## -----------------------------------------------------------------------------
## Table 1
## -----------------------------------------------------------------------------

ParticipantCharacteristics <- read_excel(filename, sheet = "Participant Characteristics")

# Filter ParticipantCharacteristics
subject_ids <- unique(Data_selected$SubjectID)

ParticipantCharacteristics <- ParticipantCharacteristics %>%
  filter(Group %in% c("Healthy", "CVA")) %>%
  filter(Subject %in% subject_ids)

# Create summary table with participant characteristics
ParticipantCharacteristics_summary_table <- ParticipantCharacteristics %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    
    `Gender (male/female)` = paste0(sum(`Gender [M/F]` == "M", na.rm = TRUE), "/", 
                    sum(`Gender [M/F]` == "F", na.rm = TRUE)),
    
    `Age (mean ± SD, years)` = paste0(round(mean(`Age [years]`, na.rm = TRUE), 1), 
                 " ± ", round(sd(`Age [years]`, na.rm = TRUE), 1)),
    
    `Height (mean ± SD, cm)` = paste0(round(mean(`Height [cm]`, na.rm = TRUE), 1), 
                    " ± ", round(sd(`Height [cm]`, na.rm = TRUE), 1)),
    
    `Weight (mean ± SD, kg)` = paste0(round(mean(`Weight [kg]`, na.rm = TRUE), 1), 
                    " ± ", round(sd(`Weight [kg]`, na.rm = TRUE), 1)),
    
    `Affected side (left/right)` = if ("Affected leg [left/right/none]" %in% names(.)) {
      leg_data <- `Affected leg [left/right/none]`
      leg_data <- leg_data[!is.na(leg_data)]
      
      if (length(leg_data) == 0 || all(leg_data == "none")) {
        "-"
      } else {
        paste0(
          sum(leg_data == "left"), "/", 
          sum(leg_data == "right")
        )
      }
    } else {
      "-"
    },
    
    `Stroke type (ischemic/hemorrhagic/unknown)` = if ("Stroke type (ischemic/hemorrhagic/unknown)" %in% names(.)) {
      if (all(is.na(`Stroke type (ischemic/hemorrhagic/unknown)`))) {
        "-"
      } else {
        paste0(sum(`Stroke type (ischemic/hemorrhagic/unknown)` == "ischemic", na.rm = TRUE), "/", 
             sum(`Stroke type (ischemic/hemorrhagic/unknown)` == "hemorrhagic", na.rm = TRUE), "/", 
             sum(`Stroke type (ischemic/hemorrhagic/unknown)` == "unknown", na.rm = TRUE))
      }
    } else {"-"},

    `Time since stroke onset (median (IQR), months)` = if ("Time since stroke [months]" %in% names(.)) {
      if (all(is.na(`Time since stroke [months]`))) {
        "-"
      } else {
        q1 <- quantile(`Time since stroke [months]`, 0.25, na.rm = TRUE)
        q3 <- quantile(`Time since stroke [months]`, 0.75, na.rm = TRUE)
        med <- median(`Time since stroke [months]`, na.rm = TRUE)
        paste0(
          round(med, 1), " (", round(q1, 1), "–", round(q3, 1), ")"
        )
      }
    } else {"-"},
    
    `Comfortable walking speed (mean ± SD, m/s)` = if ("Comfortable walking speed [m/s]" %in% names(.)) {
      paste0(
        round(mean(`Comfortable walking speed [m/s]`, na.rm = TRUE), 1), " ± ",
        round(sd(`Comfortable walking speed [m/s]`, na.rm = TRUE), 1)
      )
    } else {"-"}
  )

# Print the final table
print(ParticipantCharacteristics_summary_table)


# Run statistical tests

## Gender (Categorical - chi-squared or Fisher's test)
gender_table <- table(ParticipantCharacteristics$Group,
                      ParticipantCharacteristics$`Gender [M/F]`)
gender_test <- chisq.test(gender_table)

## Age (Continuous - t-test or Mann–Whitney)
age_test <- wilcox.test(`Age [years]` ~ Group,
                        data = ParticipantCharacteristics, conf.int = TRUE)

## Height (Continuous - t-test or Mann–Whitney)
height_test <- wilcox.test(`Height [cm]` ~ Group,
                           data = ParticipantCharacteristics, conf.int = TRUE)

## Weight (Continuous - t-test or Mann–Whitney)
weight_test <- wilcox.test(`Weight [kg]` ~ Group,
                           data = ParticipantCharacteristics, conf.int = TRUE)

## Comfortable walking speed (Continuous - t-test or Mann–Whitney)
speed_test <- wilcox.test(`Comfortable walking speed [m/s]` ~ Group,
                          data = ParticipantCharacteristics, conf.int = TRUE)

# Display test results
list(
  Gender_test = gender_test,
  Age_test = age_test,
  Height_test = height_test,
  Weight_test = weight_test,
  Comfortable_walking_speed_test = speed_test
)



## -----------------------------------------------------------------------------
## Figure 3
## -----------------------------------------------------------------------------

# Helper to extract LMM labels
summaryStats_Individual_level_association <- output_Individual_level_association %>%
  group_by(Group, StepCondition) %>%
  summarise(
    R2_median = median(LMM_R2_subject, na.rm = TRUE),
    R2_lower = quantile(LMM_R2_subject, probs = 0.25, na.rm = TRUE),
    R2_upper = quantile(LMM_R2_subject, probs = 0.75, na.rm = TRUE),
    .groups = "drop"
  )

get_lmm_label <- function(group_name, step_condition_label) {
  row1 <- output_Group_level_association %>%
    filter(Group == group_name, StepCondition == as.numeric(step_condition_label))
  row2 <- summaryStats_Individual_level_association %>%
    filter(Group == group_name, StepCondition == as.numeric(step_condition_label))
  
  # Format numeric values
  r2_marginal <- format(round(row1$R2_marginal, 2), nsmall = 2)
  r2_within_median <- format(round(row2$R2_median, 2), nsmall = 2)
  r2_within_ci_low <- format(round(row2$R2_lower, 2), nsmall = 2)
  r2_within_ci_high <- format(round(row2$R2_upper, 2), nsmall = 2)
  
  # Compose labels
  label1 <- paste0(
    "R[marginal]^2 == ", r2_marginal
  )
  label2 <- paste0(
    "R[subject]^2 == ", r2_within_median,
    " ~ '[' ~ ", r2_within_ci_low,
    " ~ '-' ~ ", r2_within_ci_high,
    " ~ ']'"
  )
  return(list(label1 = label1, label2 = label2))
}

# Create color palette
subject_ids <- sort(unique(Dataset_full$SubjectID))
color_palette <- createPalette(length(subject_ids), seedcolors = c("#000000"))
subject_color_map <- setNames(color_palette, subject_ids)

# Plot generator
make_plot <- function(group_name, step_condition_label) {
  # Filter dataset
  filtered_data <- Dataset_full %>%
    filter(Group == group_name, StepCondition == as.numeric(step_condition_label))
  
  # Extract labels from LMM
  lmm_labels <- get_lmm_label(group_name, step_condition_label)
  lmm_label1 <- lmm_labels$label1
  lmm_label2 <- lmm_labels$label2
  
  # Extract intercept and slope from LMM
  row <- output_Group_level_association %>%
    filter(Group == group_name, StepCondition == as.numeric(step_condition_label))
  
  intercept <- row$Intercept_Est
  slope <- row$Slope_Est

  # Create plot titles
  n_val <- output_Group_level_association %>%
    filter(Group == group_name, StepCondition == as.numeric(step_condition_label)) %>%
    pull(N_SubjectID)
  
  title_text <- case_when(
    group_name == "Healthy"    ~ paste0("Healthy individuals\n(N = ", n_val, ")\n"),
    group_name == "Paretic"    ~ paste0("Post-stroke individuals\nparetic leg\n(N = ", n_val, ")\n"),
    group_name == "NonParetic" ~ paste0("Post-stroke individuals\nnon-paretic leg\n(N = ", n_val, ")\n"),
    TRUE ~ paste0(group_name, "\n(N = ", n_val, ")\n")
  )
  
  # Generate group-level prediction line
  x_range <- range(filtered_data$APacceleration, na.rm = TRUE)
  x_vals <- seq(x_range[1], x_range[2], length.out = 100)
  line_data <- data.frame(
    APacceleration = x_vals,
    PropulsionForce = intercept + slope * x_vals
  )
  filtered_data <- filtered_data %>%
    mutate(
      group_pred = intercept + slope * APacceleration,
      group_resid = PropulsionForce - group_pred
    )
  
  # Generate individual-level prediction lines
  individual_models <- filtered_data %>%
    group_by(SubjectID) %>%
    do(model = lm(PropulsionForce ~ APacceleration, data = .))
  
  filtered_data <- filtered_data %>%
    group_by(SubjectID) %>%
    mutate(
      indiv_pred = predict(individual_models$model[[cur_group_id()]]),
      indiv_resid = PropulsionForce - indiv_pred
    ) %>%
    ungroup()
  
  
  # Create plot
  ggplot(filtered_data, aes(x = APacceleration, y = PropulsionForce, color = SubjectID)) +
  # Data points
    geom_point(alpha = 0.5, size = 1.5) +
    scale_color_manual(values = subject_color_map) +
  # Individual-level R2 + RMSE
    geom_smooth(method = "lm", se = FALSE, aes(group = SubjectID), size = 1.0) +
  # Group-level R2 + RMSE
    geom_line(data = line_data, aes(x = APacceleration, y = PropulsionForce), color = "black", size = 1.5, inherit.aes = FALSE) +
  # R2 labels
    annotate("text", x = 0.05, y = 0.80, label = lmm_label1, hjust = 0, vjust = 1, parse = TRUE, size = 8.0) +
    annotate("text", x = 0.05, y = 0.70, label = lmm_label2, hjust = 0, vjust = 1, parse = TRUE, size = 8.0) +
  # Plot theme
    theme_minimal(base_size = 25) +
    theme(panel.grid = element_line(color = "gray90")) +
    xlim(0, 0.8) +
    ylim(0, 0.8) +
    coord_fixed(ratio = 1) +
    labs(title = title_text, 
         x = "AP Acceleration Impulse [m/s]", 
         y = "Propulsion Force Impulse [N·s/kg]") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, vjust = 0.5),
      legend.position = "none")
}

# Generate plots
p_A <- make_plot("Healthy", "1")
p_B <- make_plot("Paretic", "1")
p_C <- make_plot("NonParetic", "1")

# Remove Y-labels of figure B and C for full plot
p_B <- p_B + theme(axis.title.y = element_blank())
p_C <- p_C + theme(axis.title.y = element_blank())

# Assemble full plot layout
full_plot <- (p_A + p_B + p_C) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag = element_text(face = "bold"))

# Print final plot
full_plot



## -----------------------------------------------------------------------------
## Figure 4
## -----------------------------------------------------------------------------


# Combine all R² data
individual_level_data <- bind_rows(output_Individual_level_association)

# Make StepCondition an ordered factor for plotting
individual_level_data$StepCondition <- factor(individual_level_data$StepCondition,
                                              levels = c(1, 3, 5),
                                              labels = c("1-step",
                                                         "3-step",
                                                         "5-step"))

# Order Groups for facet labels
individual_level_data$Group <- factor(individual_level_data$Group,
                                      levels = c("Healthy", "Paretic", "NonParetic"),
                                      labels = c(
                                        paste0("Healthy individuals\n(N = ", Number_of_Steps_summary$N_individuals[Number_of_Steps_summary$Group=="Healthy"],")"),
                                        paste0("Post-stroke individuals - paretic leg\n(N = ", Number_of_Steps_summary$N_individuals[Number_of_Steps_summary$Group=="Paretic"],")"),
                                        paste0("Post-stroke individuals - non-paretic leg\n(N = ", Number_of_Steps_summary$N_individuals[Number_of_Steps_summary$Group=="NonParetic"],")")
                                      ))

labels_vec <- levels(individual_level_data$Group)

# Prepare significance data
prepare_sig_data <- function(df, metric, labels_vec) {
  df %>%
    filter(metric == !!metric) %>%
    rename(Group = group) %>%
    mutate(
      group1 = recode(group1, "1" = "1-step", "3" = "3-step", "5" = "5-step"),
      group2 = recode(group2, "1" = "1-step", "3" = "3-step", "5" = "5-step"),
      Group  = recode(Group, "Healthy" = labels_vec[1], "Paretic" = labels_vec[2], "NonParetic" = labels_vec[3])
    )
}

sig_LMM_R2_subject <- prepare_sig_data(output_Wilcoxon_results, "LMM_R2_subject", labels_vec)
sig_LMM_RMSE       <- prepare_sig_data(output_Wilcoxon_results, "LMM_RMSE", labels_vec)

# Boxplot function
make_boxplot <- function(data, yvar, ylab, sig_data, ylim, breaks) {
  
  # dynamic y offsets based on ylim
  y_min <- ylim[1]; y_max <- ylim[2]
  height_unit <- (y_max - y_min) * 0.025
  
  height_map <- c(
    "1-step vs 3-step" = y_max,
    "1-step vs 5-step" = y_max - 3*height_unit,
    "3-step vs 5-step" = y_max - 6*height_unit
  )
  
  # Prepare sig data
  sig_data <- sig_data %>%
    filter(p.adj.signif != "ns") %>%
    mutate(
      comparison = paste(group1, "vs", group2),
      y_position = unname(height_map[comparison]),
      group1 = factor(group1, levels = levels(data$StepCondition)),
      group2 = factor(group2, levels = levels(data$StepCondition)),
      Group  = factor(Group, levels = levels(data$Group))
    ) %>%
    drop_na(y_position)
  
  # Base plot
  p <- ggplot(data, aes(x = StepCondition, y = !!sym(yvar))) +
    geom_boxplot(outlier.shape = NA, fill = "white", color = "black", linewidth = 0.8) +
    geom_jitter(width = 0.2, size = 2.0, alpha = 0.5) +
    facet_wrap(~Group, nrow = 1) +
    scale_y_continuous(limits = ylim, breaks = breaks) +
    labs(y = ylab, x = NULL) +
    theme_minimal(base_size = 25) +
    theme(
      strip.text = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )

  # Add significance brackets if any
  if (nrow(sig_data) > 0) {
    for (grp in unique(sig_data$Group)) {
      sig_sub <- sig_data %>% filter(Group == grp)
      if (nrow(sig_sub) > 0) {
        p <- p + geom_signif(
          data = sig_sub,
          aes(xmin = group1, xmax = group2, annotations = p.adj.signif, y_position = y_position),
          inherit.aes = FALSE,
          manual = TRUE,
          tip_length = 0.02,
          textsize = 8,
          vjust = 0.5
        )
      }
    }
  }
  
  p
}

# Create individual panels
pA <- make_boxplot(
  data = individual_level_data,
  yvar = "LMM_R2_subject",
  ylab = "R²",
  sig_data = sig_LMM_R2_subject,
  ylim = c(-0.5, 1.1),
  breaks = seq(-0.4, 1.0, 0.2)
) + ggtitle("A") +
  theme(
    plot.title = element_text(
      face = "bold"
    )
  )

pB <- make_boxplot(
  data = individual_level_data,
  yvar = "LMM_RMSE",
  ylab = "RMSE [N·s/kg]",
  sig_data = sig_LMM_RMSE,
  ylim = c(0, 0.09),
  breaks = seq(0, 0.09, 0.015)
) + ggtitle("B") +
  theme(
    plot.title = element_text(
      face = "bold"
    )
  )

# Add column headers
col1_label <- wrap_elements(grid::textGrob(
  paste0("Healthy individuals\n(N = ", Number_of_Steps_summary$N_individuals[Number_of_Steps_summary$Group=="Healthy"],")"),
  gp = grid::gpar(fontsize = 25, fontface = "bold"), hjust = 0.5, x = 0.5
))
col2_label <- wrap_elements(grid::textGrob(
  paste0("Post-stroke individuals\nparetic leg\n(N = ", Number_of_Steps_summary$N_individuals[Number_of_Steps_summary$Group=="Paretic"],")"),
  gp = grid::gpar(fontsize = 25, fontface = "bold"), hjust = 0.5, x = 0.5
))
col3_label <- wrap_elements(grid::textGrob(
  paste0("Post-stroke individuals\nnon-paretic leg\n(N = ", Number_of_Steps_summary$N_individuals[Number_of_Steps_summary$Group=="NonParetic"],")"),
  gp = grid::gpar(fontsize = 25, fontface = "bold"), hjust = 0.5, x = 0.5
))

# Combine panels into final layout
final_plot <-
  (col1_label | col2_label | col3_label) /
  (pA) /
  (pB) +
  plot_layout(heights = c(0.35, 1, 1)) +
  plot_annotation(tag_levels = NULL)

# Print final plot
final_plot


# Print summary stats
individual_level_summary_stats <- individual_level_data %>%
  group_by(Group, StepCondition) %>%
  summarise(
    median_R2_within = median(LMM_R2_subject, na.rm = TRUE),
    Q1_R2_within = quantile(LMM_R2_subject, 0.25, na.rm = TRUE),
    Q3_R2_within = quantile(LMM_R2_subject, 0.75, na.rm = TRUE),
    IQR_R2_within = IQR(LMM_R2_subject, na.rm = TRUE),
    median_RMSE = median(LMM_RMSE, na.rm = TRUE),
    Q1_RMSE = quantile(LMM_RMSE, 0.25, na.rm = TRUE),
    Q3_RMSE = quantile(LMM_RMSE, 0.75, na.rm = TRUE),
    IQR_RMSE = IQR(LMM_RMSE, na.rm = TRUE),
    median_slope = median(LMM_Slope, na.rm = TRUE),
    Q1_slope = quantile(LMM_Slope, 0.25, na.rm = TRUE),
    Q3_slope = quantile(LMM_Slope, 0.75, na.rm = TRUE),
    IQR_slope = IQR(LMM_Slope, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(Group, StepCondition)

print(individual_level_summary_stats)



## -----------------------------------------------------------------------------
## Output dataframes
## -----------------------------------------------------------------------------

View(ParticipantCharacteristics_summary_table)

View(Number_of_Steps_summary)

View(Summary_parameters_per_subject)

View(Summary_parameters_per_group)

View(Dataset_full)

View(output_Group_level_association)

View(output_Individual_level_association)

View(individual_level_summary_stats)

View(output_Friedman_results)

View(output_Wilcoxon_results)


