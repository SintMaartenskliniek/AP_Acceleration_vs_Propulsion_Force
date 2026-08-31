"""
Creates a dataframe with OMCS gait data for the left and right leg (df_left.pkl and df_right.pkl) and saves them to the working directory.
* Each row represents a step of a subject.
* Spatiotemporal metrics are organized in columns.
* Steps are linked using initial contact (IC) and terminal contact (TC) events, enabling step-by-step comparison between legs.

Version - Author:
    2026: Lars van Rengs - l.vanrengs@maartenskliniek.nl
"""

# Import dependencies
import pandas as pd
import numpy as np
import warnings
import pickle

from OMCS_GaitAnalysis.readmarkerdata import readmarkerdata
from OMCS_GaitAnalysis.gaiteventdetection import gaiteventdetection
from OMCS_GaitAnalysis.gaitcharacteristics import spatiotemporals, propulsion

# Define groups, subjects, and trials
group_names = [
    'Healthy_controls',
    'CVA',
    'CVA_feedback'
]
subject_names = {
    'Healthy_controls': ['900_V_01', '900_V_03', '900_V_04', '900_V_05', '900_V_06', '900_V_07', '900_V_08', '900_V_09', '900_V_10', '900_V_11', '900_V_12', '900_V_13', '900_V_14', '900_V_15', '900_V_16', '900_V_18', '900_V_19', '900_V_20', '900_V_21', '900_V_22'],
    'CVA': ['900_CVA_01', '900_CVA_02', '900_CVA_03', '900_CVA_04', '900_CVA_05', '900_CVA_06', '900_CVA_07', '900_CVA_08', '900_CVA_09', '900_CVA_10'],
    'CVA_feedback': ['1019_pp01', '1019_pp02', '1019_pp03', '1019_pp04', '1019_pp05', '1019_pp06', '1019_pp07', '1019_pp08', '1019_pp09', '1019_pp10', '1019_pp11', '1019_pp12']
}
trial_names = {
    'Healthy_controls': {
        '900_V_01': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_01/Vicon/GRAIL/900_V_pp01_SP01.c3d'],
        '900_V_03': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_03/Vicon/GRAIL/900_V_pp03_SP01.c3d'],
        '900_V_04': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_04/Vicon/GRAIL/900_V_pp04_SP01.c3d'],
        '900_V_05': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_05/Vicon/GRAIL/900_V_pp05_SP01.c3d'],
        '900_V_06': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_06/Vicon/GRAIL/900_V_pp06_SP01.c3d'],
        '900_V_07': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_07/Vicon/GRAIL/900_V_pp07_SP01.c3d'],
        '900_V_08': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_08/Vicon/GRAIL/900_V_pp08_SP02.c3d'],
        '900_V_09': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_09/Vicon/GRAIL/900_V_pp09_SP01.c3d'],
        '900_V_10': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_10/Vicon/GRAIL/900_V_pp10_SP01.c3d'],
        '900_V_11': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_11/Vicon/GRAIL/900_V_pp11_SP01.c3d'],
        '900_V_12': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_12/Vicon/GRAIL/900_V_pp12_SP01.c3d'],
        '900_V_13': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_13/Vicon/GRAIL/900_V_pp13_SP01.c3d'],
        '900_V_14': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_14/Vicon/GRAIL/900_V_pp14_SP01.c3d'],
        '900_V_15': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_15/Vicon/GRAIL/900_V_pp15_SP01.c3d'],
        '900_V_16': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_16/Vicon/GRAIL/900_V_pp16_SP01.c3d'],
        '900_V_18': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_18/Vicon/GRAIL/900_V_pp18_SP01.c3d'],
        '900_V_19': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_19/Vicon/GRAIL/900_V_pp19_SP01.c3d'],
        '900_V_20': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_20/Vicon/GRAIL/900_V_pp20_SP01.c3d'],
        '900_V_21': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_21/Vicon/GRAIL/900_V_pp21_SP01.c3d'],
        '900_V_22': ['IMU_GaitAnalysis/data/Healthy_controls/900_V_22/Vicon/GRAIL/900_V_pp22_SP01.c3d'],
    },
    'CVA': {
        '900_CVA_01': ['IMU_GaitAnalysis/data/CVA/900_CVA_01/Vicon/GRAIL/900_CVA_01_SP01.c3d'],
        '900_CVA_02': ['IMU_GaitAnalysis/data/CVA/900_CVA_02/Vicon/GRAIL/900_CVA_pp02_SP01.c3d'],
        '900_CVA_03': [],
        '900_CVA_04': ['IMU_GaitAnalysis/data/CVA/900_CVA_04/Vicon/GRAIL/900_CVA_04_SP01.c3d'],
        '900_CVA_05': ['IMU_GaitAnalysis/data/CVA/900_CVA_05/Vicon/GRAIL/900_CVA_05_SP01.c3d'],
        '900_CVA_06': ['IMU_GaitAnalysis/data/CVA/900_CVA_06/Vicon/GRAIL/900_CVA_06_SP01.c3d'],
        '900_CVA_07': ['IMU_GaitAnalysis/data/CVA/900_CVA_07/Vicon/GRAIL/900_CVA_07_SP01.c3d'],
        '900_CVA_08': ['IMU_GaitAnalysis/data/CVA/900_CVA_08/Vicon/GRAIL/900_CVA_08_SP01.c3d'],
        '900_CVA_09': [],
        '900_CVA_10': ['IMU_GaitAnalysis/data/CVA/900_CVA_10/Vicon/GRAIL/900_CVA_10_SP01.c3d'],
    },
    'CVA_feedback': {
        '1019_pp01': ['MovingReality/data/1019_pp01/Vicon/1019_MR001_1Reg.c3d'],
        '1019_pp02': ['MovingReality/data/1019_pp02/Vicon/1019_MR002_Reg.c3d'],
        '1019_pp03': ['MovingReality/data/1019_pp03/Vicon/1019_MR003_1Reg02.c3d'],
        '1019_pp04': ['MovingReality/data/1019_pp04/Vicon/1019_MR004_1Reg.c3d'],
        '1019_pp05': ['MovingReality/data/1019_pp05/Vicon/1019_MR005_1Reg01.c3d'],
        '1019_pp06': ['MovingReality/data/1019_pp06/Vicon/1019_MR006_1Reg.c3d'],
        '1019_pp07': ['MovingReality/data/1019_pp07/Vicon/1019_MR007_1Reg02.c3d'],
        '1019_pp08': ['MovingReality/data/1019_pp08/Vicon/1019_MR008_1Reg02.c3d'],
        '1019_pp09': ['MovingReality/data/1019_pp09/Vicon/1019_MR009_1Reg.c3d'],
        '1019_pp10': ['MovingReality/data/1019_pp10/Vicon/1019_MR010_1Reg.c3d'],
        '1019_pp11': ['MovingReality/data/1019_pp11/Vicon/1019_MR011_1Reg.c3d'],
        '1019_pp12': ['MovingReality/data/1019_pp12/Vicon/1019_MR012_1Reg.c3d'],
    }
}               

# Subject-specific information
subject_info = {
    '900_V_01': {'gender': 'F', 'body_mass': 72.0, 'height': 168, 'affected_leg': 'none'},
    '900_V_03': {'gender': 'F', 'body_mass': 74.8, 'height': 164, 'affected_leg': 'none'},
    '900_V_04': {'gender': 'M', 'body_mass': 76.8, 'height': 166, 'affected_leg': 'none'},
    '900_V_05': {'gender': 'F', 'body_mass': 67.8, 'height': 165, 'affected_leg': 'none'},
    '900_V_06': {'gender': 'M', 'body_mass': 77.2, 'height': 183, 'affected_leg': 'none'},
    '900_V_07': {'gender': 'F', 'body_mass': 62.4, 'height': 173, 'affected_leg': 'none'},
    '900_V_08': {'gender': 'F', 'body_mass': 63.6, 'height': 168, 'affected_leg': 'none'},
    '900_V_09': {'gender': 'M', 'body_mass': 69.0, 'height': 179, 'affected_leg': 'none'},
    '900_V_10': {'gender': 'M', 'body_mass': 93.0, 'height': 186, 'affected_leg': 'none'},
    '900_V_11': {'gender': 'M', 'body_mass': 77.6, 'height': 181, 'affected_leg': 'none'},
    '900_V_12': {'gender': 'F', 'body_mass': 78.2, 'height': 180, 'affected_leg': 'none'},
    '900_V_13': {'gender': 'M', 'body_mass': 88.6, 'height': 180, 'affected_leg': 'none'},
    '900_V_14': {'gender': 'F', 'body_mass': 68.4, 'height': 170, 'affected_leg': 'none'},
    '900_V_15': {'gender': 'F', 'body_mass': 66.2, 'height': 162, 'affected_leg': 'none'},
    '900_V_16': {'gender': 'F', 'body_mass': 70.4, 'height': 166, 'affected_leg': 'none'},
    '900_V_18': {'gender': 'M', 'body_mass': 77.0, 'height': 182, 'affected_leg': 'none'},
    '900_V_19': {'gender': 'F', 'body_mass': 70.0, 'height': 174, 'affected_leg': 'none'},
    '900_V_20': {'gender': 'M', 'body_mass': 76.8, 'height': 180, 'affected_leg': 'none'},
    '900_V_21': {'gender': 'M', 'body_mass': 89.2, 'height': 179, 'affected_leg': 'none'},
    '900_V_22': {'gender': 'M', 'body_mass': 73.4, 'height': 176, 'affected_leg': 'none'},

    '900_CVA_01': {'gender': 'F', 'body_mass': 70.0, 'height': 162, 'affected_leg': 'right'},
    '900_CVA_02': {'gender': 'M', 'body_mass': 82.0, 'height': 183, 'affected_leg': 'left'},
    '900_CVA_03': {'gender': 'M', 'body_mass': 71.0, 'height': 178, 'affected_leg': 'left'},
    '900_CVA_04': {'gender': 'M', 'body_mass': 93.0, 'height': 181, 'affected_leg': 'left'},
    '900_CVA_05': {'gender': 'M', 'body_mass': 91.0, 'height': 171, 'affected_leg': 'left'},
    '900_CVA_06': {'gender': 'F', 'body_mass': 71.0, 'height': 176, 'affected_leg': 'right'},
    '900_CVA_07': {'gender': 'M', 'body_mass': 95.4, 'height': 184, 'affected_leg': 'right'},
    '900_CVA_08': {'gender': 'M', 'body_mass': 85.0, 'height': 184, 'affected_leg': 'right'},   # Same person as 1019_pp07
    '900_CVA_09': {'gender': 'M', 'body_mass': 76.0, 'height': 180, 'affected_leg': 'right'},
    '900_CVA_10': {'gender': 'F', 'body_mass': 77.0, 'height': 165, 'affected_leg': 'right'},

    '1019_pp01': {'gender': 'M', 'body_mass': 122.0, 'height': 177.5, 'affected_leg': 'left'},
    '1019_pp02': {'gender': 'F', 'body_mass': 68.0, 'height': 171, 'affected_leg': 'right'},
    '1019_pp03': {'gender': 'M', 'body_mass': 75.0, 'height': 172, 'affected_leg': 'right'},
    '1019_pp04': {'gender': 'F', 'body_mass': 70.0, 'height': 163, 'affected_leg': 'left'},
    '1019_pp05': {'gender': 'M', 'body_mass': 80.0, 'height': 183, 'affected_leg': 'left'},
    '1019_pp06': {'gender': 'M', 'body_mass': 85.0, 'height': 194, 'affected_leg': 'left'},
    '1019_pp07': {'gender': 'M', 'body_mass': 90.0, 'height': 183, 'affected_leg': 'right'},    # Same person as 900_CVA_08
    '1019_pp08': {'gender': 'F', 'body_mass': 91.0, 'height': 172, 'affected_leg': 'right'},
    '1019_pp09': {'gender': 'M', 'body_mass': 74.0, 'height': 171, 'affected_leg': 'right'},
    '1019_pp10': {'gender': 'M', 'body_mass': 82.0, 'height': 188, 'affected_leg': 'right'},
    '1019_pp11': {'gender': 'F', 'body_mass': 104.0, 'height': 170, 'affected_leg': 'left'},
    '1019_pp12': {'gender': 'F', 'body_mass': 79.0, 'height': 172, 'affected_leg': 'left'}
}

# Notes:
#   * 900_CVA_04_SP01.c3d --> walking mostly on one of the treadmill bands, not viable for gait event detection
#   * 900_V_pp07_SP01.c3d --> OMCS data is missing


"""
-------------------------------------------------------------------------------
                                Left leg
-------------------------------------------------------------------------------
"""
# Initialize storage for df_left
df_left = {}

# Function to process the left leg for a single trial
def process_trial_left(trial, body_mass):  
    n_columns_df = 40
    debugplot_indicator = False

    # Marker data
    markerdata, fs_markerdata, analogdata, fs_analogdata = readmarkerdata(trial, analogdata=True)
    markerdatafilt = {}
    for key in markerdata:
        if 'LASI' in key:
            markerdatafilt['LASI'] = markerdata[key]
        elif 'RASI' in key:
            markerdatafilt['RASI'] = markerdata[key]
        elif 'LPSI' in key:
            markerdatafilt['LPSI'] = markerdata[key]
        elif 'RPSI' in key:
            markerdatafilt['RPSI'] = markerdata[key]
        elif 'LTHI' in key:
            markerdatafilt['LTHI'] = markerdata[key]
        elif 'LKNE' in key:
            markerdatafilt['LKNE'] = markerdata[key]
        elif 'LTIB' in key:
            markerdatafilt['LTIB'] = markerdata[key]
        elif 'LANK' in key:
            markerdatafilt['LANK'] = markerdata[key]
        elif 'LHEE' in key:
            markerdatafilt['LHEE'] = markerdata[key]
        elif 'LTOE' in key:
            markerdatafilt['LTOE'] = markerdata[key]
        elif 'RTHI' in key:
            markerdatafilt['RTHI'] = markerdata[key]
        elif 'RKNE' in key:
            markerdatafilt['RKNE'] = markerdata[key]
        elif 'RTIB' in key:
            markerdatafilt['RTIB'] = markerdata[key]
        elif 'RANK' in key:
            markerdatafilt['RANK'] = markerdata[key]
        elif 'RHEE' in key:
            markerdatafilt['RHEE'] = markerdata[key]
        elif 'RTOE' in key:
            markerdatafilt['RTOE'] = markerdata[key]
           
    # Interpolate missing values
    if trial == '900_V_pp08_SP02.c3d': # Gap fill (3 x 1 sample)
        for key in markerdatafilt:
            missingvalues = np.unique(np.where(markerdatafilt[key] == 0)[0])
            nonmissingvalues = (np.where(markerdatafilt[key] != 0)[0])
            markerdatafilt[key][missingvalues,0] = np.interp(missingvalues, nonmissingvalues, markerdatafilt[key][nonmissingvalues,0])
            markerdatafilt[key][missingvalues,1] = np.interp(missingvalues, nonmissingvalues, markerdatafilt[key][nonmissingvalues,1])
            markerdatafilt[key][missingvalues,2] = np.interp(missingvalues, nonmissingvalues, markerdatafilt[key][nonmissingvalues,2])
           
    markerdata = markerdatafilt
       
    # Gait events and characteristics
    gaitevents = gaiteventdetection(markerdata, fs_markerdata, algorithmtype='velocity', trialtype='treadmill', debugplot=debugplot_indicator)
    gaitcharacteristics = spatiotemporals(markerdata, gaitevents)
    gaitevents, gaitcharacteristics, analogdata = propulsion(gaitevents, gaitcharacteristics, analogdata, bodyweight=body_mass, debugplot=debugplot_indicator, plot_title=trial)


    """ Extract data """    
    InitialContactIndex_left = gaitevents['Index numbers initial contact left']                 # size(n,1) --> IC
    TerminalContactIndex_left = gaitevents['Index numbers terminal contact left']               # size(n,1) --> TC 
        
    StanceTime_left = gaitcharacteristics['Stance time left (s)']                               # size(n,1) --> StanceTime                  = IC(i) -> TC(i)
    SwingTime_left = gaitcharacteristics['Swing time left (s)']                                 # size(n,1) --> SwingTime                   = TC(i) -> IC(i+1)
    GaitCycleDuration_left = gaitcharacteristics['Gait Cycle duration left (s)']                # size(n,3) --> TC, IC, GaitCycleDuration   = IC(i) -> IC(i+1) // TC(i) -> TC(i+1)
    GaitSpeedStride_left = gaitcharacteristics['Gait speed left strides (m/s)']                 # size(n,3) --> TC, IC, GaitSpeedStride     = speed(TC(i) -> IC(i+1))
    StrideLength_left = gaitcharacteristics['Stridelength left (mm)']                           # size(n,3) --> TC, IC, StrideLength        = distance(TC(i) -> IC(i+1))
    Contralateral_StrideLength_right = gaitcharacteristics['Stridelength right (mm)']           # size(n,3) --> TC, IC, StrideLength        = distance(TC(i) -> IC(i+1))
    StepLength_left = gaitcharacteristics['Steplength left (mm)']                               # size(n,2) --> IC, StepLength
    Contralateral_StepLength_right = gaitcharacteristics['Steplength right (mm)']               # size(n,2) --> IC, StepLength
    StepWidth_left = gaitcharacteristics['Stepwidth left (mm)']                                 # size(n,2) --> IC, StepWidth
    
    BrakingImpulse_left = gaitcharacteristics['Braking left']                                   # size(n,3) --> StartIndex, StopIndex, ImpulseValue
    BrakingPeak_left = gaitcharacteristics['Peak braking left']                                 # size(n,2) --> PeakIndex, PeakValue
    PropulsionImpulse_left = gaitcharacteristics['Propulsion left']                             # size(n,3) --> StartIndex, StopIndex, ImpulseValue
    PropulsionPeak_left = gaitcharacteristics['Peak propulsion left']                           # size(n,2) --> PeakIndex, PeakValue

   
    """ Create dataframe for IC[i], TC[i] and IC[i+1] """
    # Match IC and TC based on the condition IC[i] < TC[i] < IC[i+1]
    IC = InitialContactIndex_left
    TC = TerminalContactIndex_left
    matched_data = []
    ic_len = len(IC)
    tc_len = len(TC)
    
    ic_i = 0
    tc_i = 0
    
    # Step through each pair of IC and next IC
    while ic_i < ic_len:
        ic = IC[ic_i]
        next_ic = IC[ic_i + 1] if ic_i + 1 < ic_len else np.nan
        
        matched_in_loop = False  # Track if we appended for this IC
    
        # Find TC within (ic, next_ic)
        while tc_i < tc_len:
            current_tc = TC[tc_i]
    
            if current_tc > ic and (np.isnan(next_ic) or current_tc < next_ic):
                matched_data.append([ic, current_tc, next_ic])
                tc_i += 1
                matched_in_loop = True
                break  # only match one TC per IC segment
            elif current_tc <= ic:
                # TC before IC, unmatched
                matched_data.append([np.nan, current_tc, ic])
                tc_i += 1
            else:
                # TC beyond next IC — no match found for current IC
                # matched_data.append([ic, np.nan, next_ic])
                break
    
        if not matched_in_loop:
            # Only add this if we didn't match the IC already
            matched_data.append([ic, np.nan, next_ic])
        ic_i += 1
    
    # Leftover TC values with no matching IC
    while tc_i < tc_len:
        matched_data.append([np.nan, TC[tc_i], np.nan])
        tc_i += 1        
    
    # Post-process: If column 0 of the current row is NaN, set column 2 of the previous row to NaN
    for i in range(1, len(matched_data)):  # Start from the second row (index 1)
        if np.isnan(matched_data[i][0]):  # If column 0 of the current row is NaN
            matched_data[i - 1][2] = np.nan  # Set column 2 of the previous row to NaN
    
    # Convert the matched data to a dataframe
    max_length = len(matched_data)
    matched_array = np.array(matched_data)
    final_df = pd.DataFrame(np.full((max_length, n_columns_df), np.nan))
    final_df.iloc[:, 7:10] = matched_array
    final_df.rename(columns={7: 'InitialContactIndex_left', 
                             8: 'TerminalContactIndex_left', 
                             9: 'NextInitialContactIndex_left'}, inplace=True)
    
    
    """ Add StanceTime_left to the final_df in column 10 based on IC and TC """
    stance_time_column = []
    
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from the current row
        if not np.isnan(ic) and not np.isnan(tc):
            # Find the index of IC to get the contralateral StanceTime_left value
            ic_index = np.where(InitialContactIndex_left == ic)[0][0]
            stance_time_column.append(StanceTime_left[ic_index])
        else:
            stance_time_column.append(np.nan)
    
    # Add the stance time column to the DataFrame
    final_df.iloc[:, 10] = stance_time_column
    
    # Set the column headers for the new columns
    final_df.rename(columns={10: 'StanceTime_Value_left'}, inplace=True)
    
    
    """ Add SwingTime_left to the final_df in column 11 based on TC and next IC """
    swing_time_column = []
    
    for row in final_df.iloc[:, 8:10].values:  # Check columns 8 (TC) and 9 (next IC) of final_df
        tc, next_ic = row  # Extract TC and next IC values from the current row
        if not np.isnan(tc) and not np.isnan(next_ic):
            # Find the index of TC to get the contralateral SwingTime_left value
            tc_index = np.where(TerminalContactIndex_left == tc)[0][0]
            swing_time_column.append(SwingTime_left[tc_index])
        else:
            swing_time_column.append(np.nan)
    
    # Add the swing time column to the DataFrame
    final_df.iloc[:, 11] = swing_time_column
    
    # Set the column headers for the new columns
    final_df.rename(columns={11: 'SwingTime_Value_left'}, inplace=True)
    
    
    """ Add GaitCycleDuration_left columns to final_df (columns 12, 13, and 14) """
    tc_column = []
    ic_column = []
    gait_cycle_duration_column = []
    
    for row in final_df.iloc[:, 8:9].values:  # Loop through column 8 (TC) of final_df
        tc = row[0]  # Extract TC value from final_df
        
        if not np.isnan(tc):
            # Find the contralateral index for TC in GaitCycleDuration_left
            matching_indices = np.where(GaitCycleDuration_left[:, 0] == tc)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract TC, IC, and GaitCycleDuration from GaitCycleDuration_left
                tc_column.append(GaitCycleDuration_left[matching_index, 0])  # TC
                ic_column.append(GaitCycleDuration_left[matching_index, 1])  # IC
                gait_cycle_duration_column.append(GaitCycleDuration_left[matching_index, 2])  # GaitCycleDuration
            else:
                # Append NaN if no match found
                tc_column.append(np.nan)
                ic_column.append(np.nan)
                gait_cycle_duration_column.append(np.nan)
        else:
            # Append NaN if TC is NaN
            tc_column.append(np.nan)
            ic_column.append(np.nan)
            gait_cycle_duration_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 12] = tc_column  # Add TC to column 12
    final_df.iloc[:, 13] = ic_column  # Add IC to column 13
    final_df.iloc[:, 14] = gait_cycle_duration_column  # Add GaitCycleDuration to column 14
    
    # Set the column headers for the new columns
    final_df.rename(columns={12: 'GaitCycleDuration_TerminalContactIndex_left',
                             13: 'GaitCycleDuration_InitialContactIndex_left',
                             14: 'GaitCycleDuration_Value_left'}, inplace=True)
        
    """ Add GaitSpeedStride_left columns to final_df (columns 15, 16, and 17) """
    gait_speed_stride_tc_column = []
    gait_speed_stride_ic_column = []
    gait_speed_stride_column = []
    
    for row in final_df.iloc[:, 8:9].values:  # Loop through column 8 (TC) of final_df
        tc = row[0]  # Extract TC value from final_df
        
        if not np.isnan(tc):
            # Find the contralateral index for TC in GaitSpeedStride_left
            matching_indices = np.where(GaitSpeedStride_left[:, 0] == tc)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract TC, IC, and GaitSpeedStride from GaitSpeedStride_left
                gait_speed_stride_tc_column.append(GaitSpeedStride_left[matching_index, 0])  # TC
                gait_speed_stride_ic_column.append(GaitSpeedStride_left[matching_index, 1])  # IC
                gait_speed_stride_column.append(GaitSpeedStride_left[matching_index, 2])  # GaitSpeedStride
            else:
                # Append NaN if no match found
                gait_speed_stride_tc_column.append(np.nan)
                gait_speed_stride_ic_column.append(np.nan)
                gait_speed_stride_column.append(np.nan)
        else:
            # Append NaN if TC is NaN
            gait_speed_stride_tc_column.append(np.nan)
            gait_speed_stride_ic_column.append(np.nan)
            gait_speed_stride_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 15] = gait_speed_stride_tc_column  # Add TC to column 15
    final_df.iloc[:, 16] = gait_speed_stride_ic_column  # Add IC to column 16
    final_df.iloc[:, 17] = gait_speed_stride_column  # Add GaitSpeedStride to column 17
       
    # Set the column headers for the new columns
    final_df.rename(columns={15: 'GaitSpeedStride_TerminalContactIndex_left',
                             16: 'GaitSpeedStride_InitialContactIndex_left',
                             17: 'GaitSpeedStride_Value_left'}, inplace=True)
    
    
    """ Add StepLength_left columns to final_df (columns 18 and 19) """
    step_length_ic_column = []
    step_length_column = []
    
    for row in final_df.iloc[:, 9:10].values:  # Loop through column 9 (next IC) of final_df
        next_ic = row[0]  # Extract next IC value from final_df (column 2)
        
        if not np.isnan(next_ic):
            # Find the contralateral index for next IC in StepLength_left
            matching_indices = np.where(StepLength_left[:, 0] == next_ic)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract IC and StepLength from StepLength_left
                step_length_ic_column.append(StepLength_left[matching_index, 0])  # IC
                step_length_column.append(StepLength_left[matching_index, 1])  # StepLength
            else:
                # Append NaN if no match found
                step_length_ic_column.append(np.nan)
                step_length_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            step_length_ic_column.append(np.nan)
            step_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 18] = step_length_ic_column  # Add IC to column 18
    final_df.iloc[:, 19] = step_length_column  # Add StepLength to column 19
    
    # Set the column headers for the new columns
    final_df.rename(columns={18: 'StepLength_InitialContactIndex_left',
                             19: 'StepLength_Value_left'}, inplace=True)
    
    
    """ Add Contralateral_StepLength_right columns to final_df (columns 20 and 21) """
    contralateral_step_length_ic_column = []
    contralateral_step_length_column = []

    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching Contralateral_StepLength_right rows where Index is between IC and TC
            for contralateral_step_length in Contralateral_StepLength_right:
                contralateral_step_length_index, contralateral_step_length_value = contralateral_step_length
                
                # Check if the Index is between IC and TC
                if ic < contralateral_step_length_index < tc:
                    contralateral_step_length_ic_column.append(contralateral_step_length_index)  # Add Index to the list
                    contralateral_step_length_column.append(contralateral_step_length_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                contralateral_step_length_ic_column.append(np.nan)
                contralateral_step_length_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            contralateral_step_length_ic_column.append(np.nan)
            contralateral_step_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 20] = contralateral_step_length_ic_column  # Add Index to column 20
    final_df.iloc[:, 21] = contralateral_step_length_column     # Add Value to column 21
    
    # Set the column headers for the new columns
    final_df.rename(columns={20: 'Contralateral_StepLength_InitialContactIndex_right',
                             21: 'Contralateral_StepLength_Value_right'}, inplace=True)


    """ Add StrideLength_left columns to final_df (columns 22, 23, and 24) """
    stride_length_tc_column = []
    stride_length_ic_column = []
    stride_length_column = []
    
    for row in final_df.iloc[:, 8:9].values:  # Loop through column 8 (TC) of final_df
        tc = row[0]  # Extract TC value from final_df
        
        if not np.isnan(tc):
            # Find the contralateral index for TC in StrideLength_left
            matching_indices = np.where(StrideLength_left[:, 0] == tc)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract TC, IC, and StrideLength from StrideLength_left
                stride_length_tc_column.append(StrideLength_left[matching_index, 0])  # TC
                stride_length_ic_column.append(StrideLength_left[matching_index, 1])  # IC
                stride_length_column.append(StrideLength_left[matching_index, 2])  # StrideLength
            else:
                # Append NaN if no match found
                stride_length_tc_column.append(np.nan)
                stride_length_ic_column.append(np.nan)
                stride_length_column.append(np.nan)
        else:
            # Append NaN if TC is NaN
            stride_length_tc_column.append(np.nan)
            stride_length_ic_column.append(np.nan)
            stride_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 22] = stride_length_tc_column  # Add TC to column 22
    final_df.iloc[:, 23] = stride_length_ic_column  # Add IC to column 23
    final_df.iloc[:, 24] = stride_length_column  # Add StrideLength to column 24
    
    # Set the column headers for the new columns
    final_df.rename(columns={22: 'StrideLength_TerminalContactIndex_left',
                             23: 'StrideLength_InitialContactIndex_left',
                             24: 'StrideLength_Value_left'}, inplace=True)


    """ Add Contralateral_StrideLength_left columns to final_df (columns 25, 26 and 27) """
    contralateral_stride_length_tc_column = []
    contralateral_stride_length_ic_column = []
    contralateral_stride_length_column = []

    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching Contralateral_StrideLength_right rows where Index is between IC and TC
            for contralateral_stride_length in Contralateral_StrideLength_right:
                contralateral_stride_length_tc_index, contralateral_stride_length_ic_index, contralateral_stride_length_value = contralateral_stride_length
                
                # Check if the Index is between IC and TC
                if ic < contralateral_stride_length_tc_index < tc:
                    contralateral_stride_length_tc_column.append(contralateral_stride_length_tc_index)  # Add Index to the list
                    contralateral_stride_length_ic_column.append(contralateral_stride_length_ic_index)  # Add Index to the list
                    contralateral_stride_length_column.append(contralateral_stride_length_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                contralateral_stride_length_tc_column.append(np.nan)
                contralateral_stride_length_ic_column.append(np.nan)
                contralateral_stride_length_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            contralateral_stride_length_tc_column.append(np.nan)
            contralateral_stride_length_ic_column.append(np.nan)
            contralateral_stride_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 25] = contralateral_stride_length_tc_column  # Add Index to column 25
    final_df.iloc[:, 26] = contralateral_stride_length_ic_column  # Add Value to column 26
    final_df.iloc[:, 27] = contralateral_stride_length_column     # Add Value to column 27
    
    # Set the column headers for the new columns
    final_df.rename(columns={25: 'Contralateral_StrideLength_TerminalContactIndex_right',
                             26: 'Contralateral_StrideLength_InitialContactIndex_right',
                             27: 'Contralateral_StrideLength_Value_right'}, inplace=True)    


    """ Add StepWidth_left columns to final_df (columns 28 and 29) """
    step_width_ic_column = []
    step_width_column = []
    
    for row in final_df.iloc[:, 9:10].values:  # Loop through column 9 (next IC) of final_df
        next_ic = row[0]  # Extract next IC value from final_df (column 2)
        
        if not np.isnan(next_ic):
            # Find the contralateral index for next IC in StepWidth_left
            matching_indices = np.where(StepWidth_left[:, 0] == next_ic)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract IC and StepWidth from StepWidth_left
                step_width_ic_column.append(StepWidth_left[matching_index, 0])  # IC
                step_width_column.append(StepWidth_left[matching_index, 1])  # StepWidth
            else:
                # Append NaN if no match found
                step_width_ic_column.append(np.nan)
                step_width_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            step_width_ic_column.append(np.nan)
            step_width_column.append(np.nan)
    
    # Add the columns to the DataFrame (columns 28 and 29)
    final_df.iloc[:, 28] = step_width_ic_column  # Add IC to column 28
    final_df.iloc[:, 29] = step_width_column  # Add StepWidth to column 29
    
    # Set the column headers for the new columns
    final_df.rename(columns={28: 'StepWidth_InitialContactIndex_left',
                             29: 'StepWidth_Value_left'}, inplace=True)
    
    
    """ Add BrakingImpulse_left columns to final_df (columns 30, 31 and 32) """
    start_index_column = []
    stop_index_column = []
    impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching BrakingImpulse_left rows where stop_index is between IC and TC
            for braking_impulse in BrakingImpulse_left:
                start_index, stop_index, impulse_value = braking_impulse
                
                # Check if the stop_index is between IC and TC
                if ic < stop_index < tc:
                    start_index_column.append(start_index)  # Add StartIndex to the list
                    stop_index_column.append(stop_index)  # Add StopIndex to the list
                    impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                start_index_column.append(np.nan)
                stop_index_column.append(np.nan)
                impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            start_index_column.append(np.nan)
            stop_index_column.append(np.nan)
            impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 30] = start_index_column  # Add StartIndex to column 30
    final_df.iloc[:, 31] = stop_index_column  # Add StopIndex to column 31
    final_df.iloc[:, 32] = impulse_value_column  # Add ImpulseValue to column 32
    
    # Set the column headers for the new columns
    final_df.rename(columns={30: 'BrakingImpulse_StartIndex_left',
                             31: 'BrakingImpulse_StopIndex_left',
                             32: 'BrakingImpulse_Value_left'}, inplace=True)
    
    
    """ Add BrakingPeak_left columns to final_df (columns 33 and 34) """
    braking_peak_index_column = []
    braking_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching BrakingPeak_left rows where PeakIndex is between IC and TC
            for peak in BrakingPeak_left:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    braking_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    braking_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                braking_peak_index_column.append(np.nan)
                braking_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            braking_peak_index_column.append(np.nan)
            braking_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 33] = braking_peak_index_column  # Add PeakIndex to column 33
    final_df.iloc[:, 34] = braking_peak_value_column  # Add PeakValue to column 34
    
    # Set the column headers for the new columns
    final_df.rename(columns={33: 'BrakingPeak_PeakIndex_left',
                             34: 'BrakingPeak_Value_left'}, inplace=True)
   
    
    """ Add PropulsionImpulse_left columns to final_df (columns 35, 36 and 37) """
    propulsion_start_index_column = []
    propulsion_stop_index_column = []
    propulsion_impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching PropulsionImpulse_left rows where start_index is between IC and TC
            for propulsion_impulse in PropulsionImpulse_left:
                start_index, stop_index, impulse_value = propulsion_impulse
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    propulsion_start_index_column.append(start_index)  # Add StartIndex to the list
                    propulsion_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    propulsion_impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                propulsion_start_index_column.append(np.nan)
                propulsion_stop_index_column.append(np.nan)
                propulsion_impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            propulsion_start_index_column.append(np.nan)
            propulsion_stop_index_column.append(np.nan)
            propulsion_impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 35] = propulsion_start_index_column  # Add StartIndex to column 35
    final_df.iloc[:, 36] = propulsion_stop_index_column  # Add StopIndex to column 36
    final_df.iloc[:, 37] = propulsion_impulse_value_column  # Add ImpulseValue to column 37
    
    # Set the column headers for the new columns
    final_df.rename(columns={35: 'PropulsionImpulse_StartIndex_left',
                             36: 'PropulsionImpulse_StopIndex_left',
                             37: 'PropulsionImpulse_Value_left'}, inplace=True)
   
    
    """ Add PropulsionPeak_left columns to final_df (columns 38 and 39) """
    propulsion_peak_index_column = []
    propulsion_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching PropulsionPeak_left rows where PeakIndex is between IC and TC
            for peak in PropulsionPeak_left:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    propulsion_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    propulsion_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                propulsion_peak_index_column.append(np.nan)
                propulsion_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            propulsion_peak_index_column.append(np.nan)
            propulsion_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 38] = propulsion_peak_index_column  # Add PeakIndex to column 38
    final_df.iloc[:, 39] = propulsion_peak_value_column  # Add PeakValue to column 39
    
    # Set the column headers for the new columns
    final_df.rename(columns={38: 'PropulsionPeak_PeakIndex_left',
                             39: 'PropulsionPeak_Value_left'}, inplace=True)
      

    """ Add trial information """
    # Suppress any warnings during the assignment of trial information to final_df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore all warnings
        # Update the first six columns in final_df
        final_df.iloc[:, 0] = [group] * len(final_df)           # Update group_name column
        final_df.iloc[:, 1] = [subject] * len(final_df)         # Update subject_name column
        final_df.iloc[:, 2] = [trial] * len(final_df)           # Update trial column
        final_df.iloc[:, 3] = [gender] * len(final_df)          # Update gender column
        final_df.iloc[:, 4] = [body_mass] * len(final_df)       # Update body_mass column
        final_df.iloc[:, 5] = [height] * len(final_df)          # Update height column
        final_df.iloc[:, 6] = [affected_leg] * len(final_df)    # Update affectedleg column
       
    # Set the column headers for the first six columns
    final_df.columns = [
        'group_name',                       # Column 0
        'subject_name',                     # Column 1
        'trial',                            # Column 2
        'gender',                           # Column 3
        'body_mass',                        # Column 4
        'height',                           # Column 5
        'affected_leg',                     # Column 6
        *final_df.columns[7:]               # Keep the remaining column headers as is
    ]


    return final_df

# Loop through groups and subjects
for group in group_names:
    for subject in subject_names[group]:
        subject_trials = trial_names[group][subject]
        gender = subject_info[subject]['gender']
        body_mass = subject_info[subject]['body_mass']
        height = subject_info[subject]['height']
        affected_leg = subject_info[subject]['affected_leg']

        for trial in subject_trials:
            try:
                trial_df_left = process_trial_left(trial, body_mass)
                df_left[(group, subject, trial)] = trial_df_left
                # Print when a trial is successfully finished
                print(f"Successfully processed left leg of trial: {trial} for subject: {subject} in group: {group}")
            except Exception as e:
                # If an error occurs, print a message and continue with the next trial
                print(f"Error processing left leg of trial: {trial} for subject: {subject} in group: {group}. Error: {e}")
                continue  # This ensures the loop continues with the next trial

# Save df_left to the working directory
with open('df_left.pkl', 'wb') as f:
    pickle.dump(df_left , f)
print("Successfully saved df_left to the working directory.")

"""
-------------------------------------------------------------------------------
                                Right leg
-------------------------------------------------------------------------------
"""
# Initialize storage for df_right
df_right = {}

# Function to process the right leg for a single trial
def process_trial_right(trial, body_mass):  
    n_columns_df = 40
    debugplot_indicator = False
    
    # Marker data
    markerdata, fs_markerdata, analogdata, fs_analogdata = readmarkerdata(trial, analogdata=True)
    markerdatafilt = {}
    for key in markerdata:
        if 'LASI' in key:
            markerdatafilt['LASI'] = markerdata[key]
        elif 'RASI' in key:
            markerdatafilt['RASI'] = markerdata[key]
        elif 'LPSI' in key:
            markerdatafilt['LPSI'] = markerdata[key]
        elif 'RPSI' in key:
            markerdatafilt['RPSI'] = markerdata[key]
        elif 'LTHI' in key:
            markerdatafilt['LTHI'] = markerdata[key]
        elif 'LKNE' in key:
            markerdatafilt['LKNE'] = markerdata[key]
        elif 'LTIB' in key:
            markerdatafilt['LTIB'] = markerdata[key]
        elif 'LANK' in key:
            markerdatafilt['LANK'] = markerdata[key]
        elif 'LHEE' in key:
            markerdatafilt['LHEE'] = markerdata[key]
        elif 'LTOE' in key:
            markerdatafilt['LTOE'] = markerdata[key]
        elif 'RTHI' in key:
            markerdatafilt['RTHI'] = markerdata[key]
        elif 'RKNE' in key:
            markerdatafilt['RKNE'] = markerdata[key]
        elif 'RTIB' in key:
            markerdatafilt['RTIB'] = markerdata[key]
        elif 'RANK' in key:
            markerdatafilt['RANK'] = markerdata[key]
        elif 'RHEE' in key:
            markerdatafilt['RHEE'] = markerdata[key]
        elif 'RTOE' in key:
            markerdatafilt['RTOE'] = markerdata[key]
           
    # Interpolate missing values
    if trial == '900_V_pp08_SP02.c3d': # Gap fill (3 x 1 sample)
        for key in markerdatafilt:
            missingvalues = np.unique(np.where(markerdatafilt[key] == 0)[0])
            nonmissingvalues = (np.where(markerdatafilt[key] != 0)[0])
            markerdatafilt[key][missingvalues,0] = np.interp(missingvalues, nonmissingvalues, markerdatafilt[key][nonmissingvalues,0])
            markerdatafilt[key][missingvalues,1] = np.interp(missingvalues, nonmissingvalues, markerdatafilt[key][nonmissingvalues,1])
            markerdatafilt[key][missingvalues,2] = np.interp(missingvalues, nonmissingvalues, markerdatafilt[key][nonmissingvalues,2])
            
    markerdata = markerdatafilt
       
    # Gait events and characteristics
    gaitevents = gaiteventdetection(markerdata, fs_markerdata, algorithmtype='velocity', trialtype='treadmill', debugplot=debugplot_indicator)
    gaitcharacteristics = spatiotemporals(markerdata, gaitevents)
    gaitevents, gaitcharacteristics, analogdata = propulsion(gaitevents, gaitcharacteristics, analogdata, bodyweight=body_mass, debugplot=debugplot_indicator, plot_title=trial)
     

    """ Extract data """    
    InitialContactIndex_right = gaitevents['Index numbers initial contact right']                 # size(n,1) --> IC
    TerminalContactIndex_right = gaitevents['Index numbers terminal contact right']               # size(n,1) --> TC 
        
    StanceTime_right = gaitcharacteristics['Stance time right (s)']                               # size(n,1) --> StanceTime                  = IC(i) -> TC(i)
    SwingTime_right = gaitcharacteristics['Swing time right (s)']                                 # size(n,1) --> SwingTime                   = TC(i) -> IC(i+1)
    GaitCycleDuration_right = gaitcharacteristics['Gait Cycle duration right (s)']                # size(n,3) --> TC, IC, GaitCycleDuration   = IC(i) -> IC(i+1) // TC(i) -> TC(i+1)
    GaitSpeedStride_right = gaitcharacteristics['Gait speed right strides (m/s)']                 # size(n,3) --> TC, IC, GaitSpeedStride     = speed(TC(i) -> IC(i+1))
    StrideLength_right = gaitcharacteristics['Stridelength right (mm)']                           # size(n,3) --> TC, IC, StrideLength        = distance(TC(i) -> IC(i+1))
    Contralateral_StrideLength_left = gaitcharacteristics['Stridelength left (mm)']               # size(n,3) --> TC, IC, StrideLength        = distance(TC(i) -> IC(i+1))
    StepLength_right = gaitcharacteristics['Steplength right (mm)']                               # size(n,2) --> IC, StepLength
    Contralateral_StepLength_left = gaitcharacteristics['Steplength left (mm)']                   # size(n,2) --> IC, StepLength
    StepWidth_right = gaitcharacteristics['Stepwidth right (mm)']                                  # size(n,2) --> IC, StepWidth
    
    BrakingImpulse_right = gaitcharacteristics['Braking right']                                   # size(n,3) --> StartIndex, StopIndex, ImpulseValue
    BrakingPeak_right = gaitcharacteristics['Peak braking right']                                 # size(n,2) --> PeakIndex, PeakValue
    PropulsionImpulse_right = gaitcharacteristics['Propulsion right']                             # size(n,3) --> StartIndex, StopIndex, ImpulseValue
    PropulsionPeak_right = gaitcharacteristics['Peak propulsion right']                           # size(n,2) --> PeakIndex, PeakValue
       
    
    """ Create dataframe for IC[i], TC[i] and IC[i+1] """
    # Match IC and TC based on the condition IC[i] < TC[i] < IC[i+1]
    IC = InitialContactIndex_right
    TC = TerminalContactIndex_right
    matched_data = []
    ic_len = len(IC)
    tc_len = len(TC)
    
    ic_i = 0
    tc_i = 0
    
    # Step through each pair of IC and next IC
    while ic_i < ic_len:
        ic = IC[ic_i]
        next_ic = IC[ic_i + 1] if ic_i + 1 < ic_len else np.nan
        
        matched_in_loop = False  # Track if we appended for this IC
    
        # Find TC within (ic, next_ic)
        while tc_i < tc_len:
            current_tc = TC[tc_i]
    
            if current_tc > ic and (np.isnan(next_ic) or current_tc < next_ic):
                matched_data.append([ic, current_tc, next_ic])
                tc_i += 1
                matched_in_loop = True
                break  # only match one TC per IC segment
            elif current_tc <= ic:
                # TC before IC, unmatched
                matched_data.append([np.nan, current_tc, ic])
                tc_i += 1
            else:
                # TC beyond next IC — no match found for current IC
                # matched_data.append([ic, np.nan, next_ic])
                break
    
        if not matched_in_loop:
            # Only add this if we didn't match the IC already
            matched_data.append([ic, np.nan, next_ic])
        ic_i += 1
    
    # Leftover TC values with no matching IC
    while tc_i < tc_len:
        matched_data.append([np.nan, TC[tc_i], np.nan])
        tc_i += 1        
    
    # Post-process: If column 0 of the current row is NaN, set column 2 of the previous row to NaN
    for i in range(1, len(matched_data)):  # Start from the second row (index 1)
        if np.isnan(matched_data[i][0]):  # If column 0 of the current row is NaN
            matched_data[i - 1][2] = np.nan  # Set column 2 of the previous row to NaN
    
    # Convert the matched data to a dataframe
    max_length = len(matched_data)
    matched_array = np.array(matched_data)
    final_df = pd.DataFrame(np.full((max_length, n_columns_df), np.nan))
    final_df.iloc[:, 7:10] = matched_array
    final_df.rename(columns={7: 'InitialContactIndex_right', 
                             8: 'TerminalContactIndex_right', 
                             9: 'NextInitialContactIndex_right'}, inplace=True)
    
    
    """ Add StanceTime_right to the final_df in column 10 based on IC and TC """
    stance_time_column = []
    
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from the current row
        if not np.isnan(ic) and not np.isnan(tc):
            # Find the index of IC to get the contralateral StanceTime_right value
            ic_index = np.where(InitialContactIndex_right == ic)[0][0]
            stance_time_column.append(StanceTime_right[ic_index])
        else:
            stance_time_column.append(np.nan)
    
    # Add the stance time column to the DataFrame
    final_df.iloc[:, 10] = stance_time_column
    
    # Set the column headers for the new columns
    final_df.rename(columns={10: 'StanceTime_Value_right'}, inplace=True)
    
    
    """ Add SwingTime_right to the final_df in column 11 based on TC and next IC """
    swing_time_column = []
    
    for row in final_df.iloc[:, 8:10].values:  # Check columns 8 (TC) and 9 (next IC) of final_df
        tc, next_ic = row  # Extract TC and next IC values from the current row
        if not np.isnan(tc) and not np.isnan(next_ic):
            # Find the index of TC to get the contralateral SwingTime_right value
            tc_index = np.where(TerminalContactIndex_right == tc)[0][0]
            swing_time_column.append(SwingTime_right[tc_index])
        else:
            swing_time_column.append(np.nan)
    
    # Add the swing time column to the DataFrame
    final_df.iloc[:, 11] = swing_time_column
    
    # Set the column headers for the new columns
    final_df.rename(columns={11: 'SwingTime_Value_right'}, inplace=True)
    
    
    """ Add GaitCycleDuration_right columns to final_df (columns 12, 13, and 14) """
    tc_column = []
    ic_column = []
    gait_cycle_duration_column = []
    
    for row in final_df.iloc[:, 8:9].values:  # Loop through column 8 (TC) of final_df
        tc = row[0]  # Extract TC value from final_df
        
        if not np.isnan(tc):
            # Find the contralateral index for TC in GaitCycleDuration_right
            matching_indices = np.where(GaitCycleDuration_right[:, 0] == tc)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract TC, IC, and GaitCycleDuration from GaitCycleDuration_right
                tc_column.append(GaitCycleDuration_right[matching_index, 0])  # TC
                ic_column.append(GaitCycleDuration_right[matching_index, 1])  # IC
                gait_cycle_duration_column.append(GaitCycleDuration_right[matching_index, 2])  # GaitCycleDuration
            else:
                # Append NaN if no match found
                tc_column.append(np.nan)
                ic_column.append(np.nan)
                gait_cycle_duration_column.append(np.nan)
        else:
            # Append NaN if TC is NaN
            tc_column.append(np.nan)
            ic_column.append(np.nan)
            gait_cycle_duration_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 12] = tc_column  # Add TC to column 12
    final_df.iloc[:, 13] = ic_column  # Add IC to column 13
    final_df.iloc[:, 14] = gait_cycle_duration_column  # Add GaitCycleDuration to column 14
    
    # Set the column headers for the new columns
    final_df.rename(columns={12: 'GaitCycleDuration_TerminalContactIndex_right',
                             13: 'GaitCycleDuration_InitialContactIndex_right',
                             14: 'GaitCycleDuration_Value_right'}, inplace=True)
        
    """ Add GaitSpeedStride_right columns to final_df (columns 15, 16, and 17) """
    gait_speed_stride_tc_column = []
    gait_speed_stride_ic_column = []
    gait_speed_stride_column = []
    
    for row in final_df.iloc[:, 8:9].values:  # Loop through column 8 (TC) of final_df
        tc = row[0]  # Extract TC value from final_df
        
        if not np.isnan(tc):
            # Find the contralateral index for TC in GaitSpeedStride_right
            matching_indices = np.where(GaitSpeedStride_right[:, 0] == tc)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract TC, IC, and GaitSpeedStride from GaitSpeedStride_right
                gait_speed_stride_tc_column.append(GaitSpeedStride_right[matching_index, 0])  # TC
                gait_speed_stride_ic_column.append(GaitSpeedStride_right[matching_index, 1])  # IC
                gait_speed_stride_column.append(GaitSpeedStride_right[matching_index, 2])  # GaitSpeedStride
            else:
                # Append NaN if no match found
                gait_speed_stride_tc_column.append(np.nan)
                gait_speed_stride_ic_column.append(np.nan)
                gait_speed_stride_column.append(np.nan)
        else:
            # Append NaN if TC is NaN
            gait_speed_stride_tc_column.append(np.nan)
            gait_speed_stride_ic_column.append(np.nan)
            gait_speed_stride_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 15] = gait_speed_stride_tc_column  # Add TC to column 15
    final_df.iloc[:, 16] = gait_speed_stride_ic_column  # Add IC to column 16
    final_df.iloc[:, 17] = gait_speed_stride_column  # Add GaitSpeedStride to column 17
       
    # Set the column headers for the new columns
    final_df.rename(columns={15: 'GaitSpeedStride_TerminalContactIndex_right',
                             16: 'GaitSpeedStride_InitialContactIndex_right',
                             17: 'GaitSpeedStride_Value_right'}, inplace=True)
   
    
    """ Add StepLength_right columns to final_df (columns 18 and 19) """
    step_length_ic_column = []
    step_length_column = []
    
    for row in final_df.iloc[:, 9:10].values:  # Loop through column 9 (next IC) of final_df
        next_ic = row[0]  # Extract next IC value from final_df (column 2)
        
        if not np.isnan(next_ic):
            # Find the contralateral index for next IC in StepLength_right
            matching_indices = np.where(StepLength_right[:, 0] == next_ic)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract IC and StepLength from StepLength_right
                step_length_ic_column.append(StepLength_right[matching_index, 0])  # IC
                step_length_column.append(StepLength_right[matching_index, 1])  # StepLength
            else:
                # Append NaN if no match found
                step_length_ic_column.append(np.nan)
                step_length_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            step_length_ic_column.append(np.nan)
            step_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 18] = step_length_ic_column  # Add IC to column 18
    final_df.iloc[:, 19] = step_length_column  # Add StepLength to column 19
    
    # Set the column headers for the new columns
    final_df.rename(columns={18: 'StepLength_InitialContactIndex_right',
                             19: 'StepLength_Value_right'}, inplace=True)
   

    """ Add Contralateral_StepLength_left columns to final_df (columns 20 and 21) """
    contralateral_step_length_ic_column = []
    contralateral_step_length_column = []

    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching Contralateral_StepLength_left rows where Index is between IC and TC
            for contralateral_step_length in Contralateral_StepLength_left:
                contralateral_step_length_index, contralateral_step_length_value = contralateral_step_length
                
                # Check if the Index is between IC and TC
                if ic < contralateral_step_length_index < tc:
                    contralateral_step_length_ic_column.append(contralateral_step_length_index)  # Add Index to the list
                    contralateral_step_length_column.append(contralateral_step_length_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                contralateral_step_length_ic_column.append(np.nan)
                contralateral_step_length_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            contralateral_step_length_ic_column.append(np.nan)
            contralateral_step_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 20] = contralateral_step_length_ic_column  # Add Index to column 20
    final_df.iloc[:, 21] = contralateral_step_length_column     # Add Value to column 21
    
    # Set the column headers for the new columns
    final_df.rename(columns={20: 'Contralateral_StepLength_InitialContactIndex_left',
                             21: 'Contralateral_StepLength_Value_left'}, inplace=True)   


    """ Add StrideLength_right columns to final_df (columns 22, 23, and 24) """
    stride_length_tc_column = []
    stride_length_ic_column = []
    stride_length_column = []
    
    for row in final_df.iloc[:, 8:9].values:  # Loop through column 8 (TC) of final_df
        tc = row[0]  # Extract TC value from final_df
        
        if not np.isnan(tc):
            # Find the contralateral index for TC in StrideLength_right
            matching_indices = np.where(StrideLength_right[:, 0] == tc)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract TC, IC, and StrideLength from StrideLength_right
                stride_length_tc_column.append(StrideLength_right[matching_index, 0])  # TC
                stride_length_ic_column.append(StrideLength_right[matching_index, 1])  # IC
                stride_length_column.append(StrideLength_right[matching_index, 2])  # StrideLength
            else:
                # Append NaN if no match found
                stride_length_tc_column.append(np.nan)
                stride_length_ic_column.append(np.nan)
                stride_length_column.append(np.nan)
        else:
            # Append NaN if TC is NaN
            stride_length_tc_column.append(np.nan)
            stride_length_ic_column.append(np.nan)
            stride_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 22] = stride_length_tc_column  # Add TC to column 22
    final_df.iloc[:, 23] = stride_length_ic_column  # Add IC to column 23
    final_df.iloc[:, 24] = stride_length_column  # Add StrideLength to column 24
    
    # Set the column headers for the new columns
    final_df.rename(columns={22: 'StrideLength_TerminalContactIndex_right',
                             23: 'StrideLength_InitialContactIndex_right',
                             24: 'StrideLength_Value_right'}, inplace=True)


    """ Add Contralateral_StrideLength_left columns to final_df (columns 25, 26 and 27) """
    contralateral_stride_length_tc_column = []
    contralateral_stride_length_ic_column = []
    contralateral_stride_length_column = []

    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching Contralateral_StrideLength_left rows where Index is between IC and TC
            for contralateral_stride_length in Contralateral_StrideLength_left:
                contralateral_stride_length_tc_index, contralateral_stride_length_ic_index, contralateral_stride_length_value = contralateral_stride_length
                
                # Check if the Index is between IC and TC
                if ic < contralateral_stride_length_tc_index < tc:
                    contralateral_stride_length_tc_column.append(contralateral_stride_length_tc_index)  # Add Index to the list
                    contralateral_stride_length_ic_column.append(contralateral_stride_length_ic_index)  # Add Index to the list
                    contralateral_stride_length_column.append(contralateral_stride_length_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                contralateral_stride_length_tc_column.append(np.nan)
                contralateral_stride_length_ic_column.append(np.nan)
                contralateral_stride_length_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            contralateral_stride_length_tc_column.append(np.nan)
            contralateral_stride_length_ic_column.append(np.nan)
            contralateral_stride_length_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 25] = contralateral_stride_length_tc_column  # Add Index to column 25
    final_df.iloc[:, 26] = contralateral_stride_length_ic_column  # Add Value to column 26
    final_df.iloc[:, 27] = contralateral_stride_length_column     # Add Value to column 27
    
    # Set the column headers for the new columns
    final_df.rename(columns={25: 'Contralateral_StrideLength_TerminalContactIndex_left',
                             26: 'Contralateral_StrideLength_InitialContactIndex_left',
                             27: 'Contralateral_StrideLength_Value_left'}, inplace=True)      
    
    
    """ Add StepWidth_right columns to final_df (columns 28 and 29) """
    step_width_ic_column = []
    step_width_column = []
    
    for row in final_df.iloc[:, 9:10].values:  # Loop through column 9 (next IC) of final_df
        next_ic = row[0]  # Extract next IC value from final_df (column 2)
        
        if not np.isnan(next_ic):
            # Find the contralateral index for next IC in StepWidth_right
            matching_indices = np.where(StepWidth_right[:, 0] == next_ic)[0]
            
            if len(matching_indices) > 0:
                # Get the first matching index
                matching_index = matching_indices[0]
                
                # Extract IC and StepWidth from StepWidth_right
                step_width_ic_column.append(StepWidth_right[matching_index, 0])  # IC
                step_width_column.append(StepWidth_right[matching_index, 1])  # StepWidth
            else:
                # Append NaN if no match found
                step_width_ic_column.append(np.nan)
                step_width_column.append(np.nan)
        else:
            # Append NaN if next IC is NaN
            step_width_ic_column.append(np.nan)
            step_width_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 28] = step_width_ic_column  # Add IC to column 28
    final_df.iloc[:, 29] = step_width_column  # Add StepWidth to column 29
    
    # Set the column headers for the new columns
    final_df.rename(columns={28: 'StepWidth_InitialContactIndex_right',
                             29: 'StepWidth_Value_right'}, inplace=True)
    
    
    """ Add BrakingImpulse_right columns to final_df (columns 30, 31 and 32) """
    # Add BrakingImpulse_right columns to final_df
    start_index_column = []
    stop_index_column = []
    impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching BrakingImpulse_right rows where stop_index is between IC and TC
            for braking_impulse in BrakingImpulse_right:
                start_index, stop_index, impulse_value = braking_impulse
                
                # Check if the stop_index is between IC and TC
                if ic < stop_index < tc:
                    start_index_column.append(start_index)  # Add StartIndex to the list
                    stop_index_column.append(stop_index)  # Add StopIndex to the list
                    impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                start_index_column.append(np.nan)
                stop_index_column.append(np.nan)
                impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            start_index_column.append(np.nan)
            stop_index_column.append(np.nan)
            impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 30] = start_index_column  # Add StartIndex to column 30
    final_df.iloc[:, 31] = stop_index_column  # Add StopIndex to column 31
    final_df.iloc[:, 32] = impulse_value_column  # Add ImpulseValue to column 32
    
    # Set the column headers for the new columns
    final_df.rename(columns={30: 'BrakingImpulse_StartIndex_right',
                             31: 'BrakingImpulse_StopIndex_right',
                             32: 'BrakingImpulse_Value_right'}, inplace=True)
    
    
    """ Add BrakingPeak_right columns to final_df (columns 33 and 34) """
    braking_peak_index_column = []
    braking_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching BrakingPeak_right rows where PeakIndex is between IC and TC
            for peak in BrakingPeak_right:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    braking_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    braking_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                braking_peak_index_column.append(np.nan)
                braking_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            braking_peak_index_column.append(np.nan)
            braking_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 33] = braking_peak_index_column  # Add PeakIndex to column 33
    final_df.iloc[:, 34] = braking_peak_value_column  # Add PeakValue to column 34
    
    # Set the column headers for the new columns
    final_df.rename(columns={33: 'BrakingPeak_PeakIndex_right',
                             34: 'BrakingPeak_Value_right'}, inplace=True)
   
    
    """ Add PropulsionImpulse_right columns to final_df (columns 35, 36 and 37) """
    propulsion_start_index_column = []
    propulsion_stop_index_column = []
    propulsion_impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching PropulsionImpulse_right rows where start_index is between IC and TC
            for propulsion_impulse in PropulsionImpulse_right:
                start_index, stop_index, impulse_value = propulsion_impulse
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    propulsion_start_index_column.append(start_index)  # Add StartIndex to the list
                    propulsion_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    propulsion_impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                propulsion_start_index_column.append(np.nan)
                propulsion_stop_index_column.append(np.nan)
                propulsion_impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            propulsion_start_index_column.append(np.nan)
            propulsion_stop_index_column.append(np.nan)
            propulsion_impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 35] = propulsion_start_index_column  # Add StartIndex to column 35
    final_df.iloc[:, 36] = propulsion_stop_index_column  # Add StopIndex to column 36
    final_df.iloc[:, 37] = propulsion_impulse_value_column  # Add ImpulseValue to column 37
    
    # Set the column headers for the new columns
    final_df.rename(columns={35: 'PropulsionImpulse_StartIndex_right',
                             36: 'PropulsionImpulse_StopIndex_right',
                             37: 'PropulsionImpulse_Value_right'}, inplace=True)
   
    
    """ Add PropulsionPeak_right columns to final_df (columns 38 and 39) """
    propulsion_peak_index_column = []
    propulsion_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching PropulsionPeak_right rows where PeakIndex is between IC and TC
            for peak in PropulsionPeak_right:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    propulsion_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    propulsion_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                propulsion_peak_index_column.append(np.nan)
                propulsion_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            propulsion_peak_index_column.append(np.nan)
            propulsion_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.iloc[:, 38] = propulsion_peak_index_column  # Add PeakIndex to column 38
    final_df.iloc[:, 39] = propulsion_peak_value_column  # Add PeakValue to column 39
    
    # Set the column headers for the new columns
    final_df.rename(columns={38: 'PropulsionPeak_PeakIndex_right',
                             39: 'PropulsionPeak_Value_right'}, inplace=True)
 

    """ Add trial information """
    # Suppress any warnings during the assignment of trial information to final_df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore all warnings
        # Update the first six columns in final_df
        final_df.iloc[:, 0] = [group] * len(final_df)           # Update group_name column
        final_df.iloc[:, 1] = [subject] * len(final_df)         # Update subject_name column
        final_df.iloc[:, 2] = [trial] * len(final_df)           # Update trial column
        final_df.iloc[:, 3] = [gender] * len(final_df)          # Update gender column
        final_df.iloc[:, 4] = [body_mass] * len(final_df)       # Update body_mass column
        final_df.iloc[:, 5] = [height] * len(final_df)          # Update height column
        final_df.iloc[:, 6] = [affected_leg] * len(final_df)    # Update affectedleg column
       
    # Set the column headers for the first six columns
    final_df.columns = [
        'group_name',                       # Column 0
        'subject_name',                     # Column 1
        'trial',                            # Column 2
        'gender',                           # Column 3
        'body_mass',                        # Column 4
        'height',                           # Column 5
        'affected_leg',                     # Column 6
        *final_df.columns[7:]               # Keep the remaining column headers as is
    ]


    return final_df

# Loop through groups and subjects
for group in group_names:
    for subject in subject_names[group]:
        subject_trials = trial_names[group][subject]
        gender = subject_info[subject]['gender']
        body_mass = subject_info[subject]['body_mass']
        height = subject_info[subject]['height']
        affected_leg = subject_info[subject]['affected_leg']

        for trial in subject_trials:
            try:
                trial_df_right = process_trial_right(trial, body_mass)
                df_right[(group, subject, trial)] = trial_df_right
                # Print when a trial is successfully finished
                print(f"Successfully processed right leg of trial: {trial} for subject: {subject} in group: {group}")
            except Exception as e:
                # If an error occurs, print a message and continue with the next trial
                print(f"Error processing right leg of trial: {trial} for subject: {subject} in group: {group}. Error: {e}")
                continue  # This ensures the loop continues with the next trial

# Save df_right to the working directory
with open('df_right.pkl', 'wb') as f:
    pickle.dump(df_right, f)
print("Successfully saved df_right to the working directory.")

