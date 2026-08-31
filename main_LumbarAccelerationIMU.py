"""
Extends the OMCS step-by-step dataframe by adding anterior-posterior acceleration measures derived from OMCS and IMU data.

Version - Author:
    2026: Lars van Rengs - l.vanrengs@maartenskliniek.nl
"""

# Import dependencies
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
import copy

from helpers_LumbarAccelerationIMU import dataimport, analyze_OMCS, OMCS_calculate_sacrum_acceleration, filter_data, APaccelerationLumbar


# %% Import data
# Set wether or not you want to create plot of the data (debugplot = True / False)
debugplot = False

# Set trialtype you wish to analyze to 'True'
analyze_trialtypes = dict()
analyze_trialtypes['Healthy_controls'] = True
analyze_trialtypes['CVA'] = True
analyze_trialtypes['CVA_feedback'] = True

# Set wether or not a saved .pkl file can be found in your directory (storedfile = True / False)
storedfile = False

if storedfile == True:
    filename = 'dataset_LumbarAccelerationIMU.pkl'
elif storedfile == False:
    # Define filepaths for vicon and xsens data
    datafolder_validation_study = os.path.abspath('IMU_GaitAnalysis/data')
    datafolder_feedback_study = os.path.abspath('MovingReality/data')
    # Set name for file to be saved
    save_as = 'dataset_LumbarAccelerationIMU.pkl' 

# If there is no .pkl file of the data available: analyze from raw data
if storedfile == False:
        
    # Data import
    corresponding_files, trialnames, OMCS, IMU, errors = dataimport(datafolder_validation_study, datafolder_feedback_study, analyze_trialtypes)
    
    # Save file inbetween (in case of error, at least all raw data is stored and does not have to be loaded again)
    f = open(save_as,"wb")
    a = {'OMCS':OMCS, 'IMU':IMU, 'corresponding_files':corresponding_files, 'trialnames':trialnames, 'analyze_trialtypes':analyze_trialtypes}
    pickle.dump(a,f)
    f.close()
    
# If there is a .pkl file of the data available: analyze from .pkl file
elif storedfile == True:
    # Open data file with analyzed gait data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        IMU = data['IMU']
        OMCS = data['OMCS']
        trialnames = data['trialnames']
        corresponding_files = data['corresponding_files']
    f.close()


# %% Exclude invalid trials
exclude_trials = [
    '900_CVA_04_SP01.c3d',
    '900_V_pp07_SP01.c3d'
]
trialnames = [t for t in trialnames if t not in exclude_trials]

# Notes:
#   * 900_CVA_04_SP01.c3d --> walking mostly on one of the treadmill bands, not viable for gait event detection
#   * 900_V_pp07_SP01.c3d --> OMCS data is missing


# %% Extract parameters from OMCS-data
OMCS, OMCS_gait_events, OMCS_gait_characteristics = analyze_OMCS(OMCS, IMU, trialnames)


# %% Extract OMCS AP-GRF
OMCS_AP_GRF_left = dict()
OMCS_AP_GRF_right = dict()
for f in OMCS:
    try:
        OMCS_AP_GRF_left[f] = OMCS[f]['Analog data']['Force Y left filtered']
        OMCS_AP_GRF_right[f] = OMCS[f]['Analog data']['Force Y right filtered']       
    except:
        print('Cannot extract AP-GRF for trial ', f) 

        
# %% Calculate OMCS acceleration
OMCS_POS_Sacrum, OMCS_VEL_Sacrum, OMCS_ACC_Sacrum = OMCS_calculate_sacrum_acceleration(OMCS)


# %% Extract IMU acceleration
IMU_ACC_SF_Lumbar = dict()
for f in IMU:
    try:
        IMU_ACC_SF_Lumbar[f] = IMU[f]['Lumbar']['raw']['Accelerometer Sensor Frame']
        
        IMU_ACCx_SF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_SF_Lumbar[f][:,0])    # filter signal: order, fcut, fs, signal
        IMU_ACCy_SF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_SF_Lumbar[f][:,1])    # filter signal: order, fcut, fs, signal
        IMU_ACCz_SF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_SF_Lumbar[f][:,2])    # filter signal: order, fcut, fs, signal
        IMU_ACC_SF_Lumbar[f] = np.column_stack((IMU_ACCx_SF, IMU_ACCy_SF, IMU_ACCz_SF))       
    except:
        print('Cannot extract IMU based acceleration (Sensor Frame) for trial ', f) 

IMU_ACC_EF_Lumbar = dict()
for f in IMU:
    try:
        IMU_ACC_EF_Lumbar[f] = IMU[f]['Lumbar']['raw']['Accelerometer Earth Frame']
        
        IMU_ACCx_EF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_EF_Lumbar[f][:,0])    # filter signal: order, fcut, fs, signal
        IMU_ACCy_EF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_EF_Lumbar[f][:,1])    # filter signal: order, fcut, fs, signal
        IMU_ACCz_EF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_EF_Lumbar[f][:,2])    # filter signal: order, fcut, fs, signal
        IMU_ACC_EF_Lumbar[f] = np.column_stack((IMU_ACCx_EF, IMU_ACCy_EF, IMU_ACCz_EF))
    except:
        print('Cannot extract IMU based acceleration (Earth Frame) for trial ', f) 


# %% Calculate IMU acceleration in body frame
# https://www.researchgate.net/publication/224593080_Which_Way_Am_I_Facing_Inferring_Horizontal_Device_Orientation_from_an_Accelerometer_Signal
def process_IMU_to_body_frame(IMU_ACC_EF_Lumbar, IMU, f):
    try:
        # Extract horizontal accelerations from Earth Frame acceleration
        acc_EF_horizontal = IMU_ACC_EF_Lumbar[f][:, :2]

        # Center for PCA
        acc_centered = acc_EF_horizontal - np.mean(acc_EF_horizontal, axis=0)

        # Apply PCA to horizontal plane
        pca = PCA(n_components=2)
        pca.fit(acc_centered)
        walking_vector = pca.components_[0]  # first principal component
        
        # Flip vector if necessary to point forward (positive X component)
        if np.mean(acc_EF_horizontal @ walking_vector) < 0:
            walking_vector *= -1
            
        # Calculate walking direction angle
        walking_direction_rad = np.arctan2(walking_vector[1], walking_vector[0])
        walking_direction_deg = np.degrees(walking_direction_rad)
        
        # Normalize angle to [-180, 180]
        if walking_direction_deg > 180:
            walking_direction_deg -= 360

        # print(f"{f}: Walking direction (deg): {walking_direction_deg:.2f}")

        # Build rotation matrix to body frame (rotate horizontal plane; around Z-axis)
        R_z = np.array([
            [np.cos(-walking_direction_rad), -np.sin(-walking_direction_rad), 0],
            [np.sin(-walking_direction_rad),  np.cos(-walking_direction_rad), 0],
            [0, 0, 1]
        ])

        # Rotate full 3D acceleration into body frame
        acc_EF_full = IMU_ACC_EF_Lumbar[f]  # Nx3
        acc_BF = (R_z @ acc_EF_full.T).T
               
        # Apply flipping if angle exceeds 90°
        if abs(walking_direction_deg) > 90:
            acc_BF[:, [0, 1]] = -acc_BF[:, [0, 1]]
           
        # Exception for trial 900_CVA_pp02_SP01.c3d
        if f == '900_CVA_pp02_SP01.c3d':
            acc_BF[:, [0, 1]] = -acc_BF[:, [0, 1]]
            
        # Apply swapping if walking angle suggests lateral walking     
        if 45 < abs(walking_direction_deg) <= 135:
            acc_BF[:, [0, 1]] = acc_BF[:, [1, 0]]
            
        # Exception for trial 900_V_pp18_SP01.c3d, 1019_MR007_1Reg02.c3d, and 1019_MR009_1Reg.c3d
        if f == '900_V_pp18_SP01.c3d' or f == '1019_MR007_1Reg02.c3d' or f == '1019_MR009_1Reg.c3d':
            acc_BF[:, [0, 1]] = acc_BF[:, [1, 0]]
    
        # Apply filter on body frame acceleration signal
        acc_BF[:, 0] = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], acc_BF[:, 0])
        acc_BF[:, 1] = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], acc_BF[:, 1])
        acc_BF[:, 2] = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], acc_BF[:, 2])

        return acc_BF, walking_direction_deg

    except Exception as e:
        print(f"Error processing trial {f}: {str(e)}")
        return None, None

IMU_ACC_BF_Lumbar = dict()
walking_directions = dict()
for f in IMU:
    acc_BF, walking_dir = process_IMU_to_body_frame(IMU_ACC_EF_Lumbar, IMU, f)
    if acc_BF is not None:
        IMU_ACC_BF_Lumbar[f] = acc_BF
        walking_directions[f] = walking_dir
        

# %% Extract AP acceleration for OMCS and IMU
OMCS_AP_ACC_Sacrum = dict()
for f in OMCS:
    try:
        OMCS_AP_ACC_Sacrum[f] = OMCS_ACC_Sacrum[f][:, 1]
    except:
        print('Cannot extract AP acceleration of OMCS for trial ', f) 

IMU_AP_ACC_Lumbar_BF = dict()
for f in IMU:
    try:
        IMU_AP_ACC_Lumbar_BF[f] = -IMU_ACC_BF_Lumbar[f][:, 0]                   # Set IMU_AP_ACC_Lumbar to the same format as OMCS_AP_ACC_Sacrum
    except:
        print('Cannot extract AP acceleration (Body Frame) of IMU for trial ', f) 
  
    
# %% Extract acceleration measures for OMCS and IMU
FinalData = {}
for f in trialnames:
    FinalData[f] = {}

    FinalData[f]['OMCS'] = {}
    FinalData[f]['IMU'] = {}

    FinalData[f]['OMCS']['Braking left - Impulse'] = {}
    FinalData[f]['OMCS']['Braking left - Peak'] = {}
    FinalData[f]['OMCS']['Propulsion left - Impulse'] = {}
    FinalData[f]['OMCS']['Propulsion left - Peak'] = {}
    FinalData[f]['OMCS']['Braking right - Impulse'] = {}
    FinalData[f]['OMCS']['Braking right - Peak'] = {}
    FinalData[f]['OMCS']['Propulsion right - Impulse'] = {}
    FinalData[f]['OMCS']['Propulsion right - Peak'] = {}

    FinalData[f]['OMCS']['AP Deceleration Sacrum left - VelocityIncrement'] = {}
    FinalData[f]['OMCS']['AP Deceleration Sacrum left - Peak'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum left - VelocityIncrement'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum left - Peak'] = {}
    FinalData[f]['OMCS']['AP Deceleration Sacrum right - VelocityIncrement'] = {}
    FinalData[f]['OMCS']['AP Deceleration Sacrum right - Peak'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum right - VelocityIncrement'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum right - Peak'] = {}

    FinalData[f]['IMU']['AP Deceleration Lumbar left - VelocityIncrement'] = {}
    FinalData[f]['IMU']['AP Deceleration Lumbar left - Peak'] = {}
    FinalData[f]['IMU']['AP Acceleration Lumbar left - VelocityIncrement'] = {}
    FinalData[f]['IMU']['AP Acceleration Lumbar left - Peak'] = {}
    FinalData[f]['IMU']['AP Deceleration Lumbar right - VelocityIncrement'] = {}
    FinalData[f]['IMU']['AP Deceleration Lumbar right - Peak'] = {}
    FinalData[f]['IMU']['AP Acceleration Lumbar right - VelocityIncrement'] = {}
    FinalData[f]['IMU']['AP Acceleration Lumbar right - Peak'] = {}


IMU_gait_events = {}
IMU_gait_characteristics = {}
for f in trialnames:
    IMU_gait_events[f] = {}
    IMU_gait_events[f]['Index numbers initial contact left'] = {}
    IMU_gait_events[f]['Index numbers terminal contact left'] = {}
    IMU_gait_events[f]['Index numbers initial contact right'] = {}
    IMU_gait_events[f]['Index numbers terminal contact right'] = {}
   
    IMU_gait_characteristics[f] = {}

        
# OMCS - Propulsion force
for f in trialnames:
    try:       
        FinalData[f]['OMCS']['Braking left - Impulse'] = OMCS_gait_characteristics[f]['Braking left']                   # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['Braking left - Peak'] = OMCS_gait_characteristics[f]['Peak braking left']                 # size(n,2) --> Index, Value
        FinalData[f]['OMCS']['Propulsion left - Impulse'] = OMCS_gait_characteristics[f]['Propulsion left']             # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['Propulsion left - Peak'] = OMCS_gait_characteristics[f]['Peak propulsion left']           # size(n,2) --> Index, Value
        
        FinalData[f]['OMCS']['Braking right - Impulse'] = OMCS_gait_characteristics[f]['Braking right']                 # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['Braking right - Peak'] = OMCS_gait_characteristics[f]['Peak braking right']               # size(n,2) --> Index, Value
        FinalData[f]['OMCS']['Propulsion right - Impulse'] = OMCS_gait_characteristics[f]['Propulsion right']           # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['Propulsion right - Peak'] = OMCS_gait_characteristics[f]['Peak propulsion right']         # size(n,2) --> Index, Value
    except:
        print('Cannot extract OMCS based propulsion force parameters for trial ', f) 


   
# OMCS - AP acceleration sacrum
for f in trialnames:
    try:
        # Prepare OMCS gait events for APaccelerationLumbar function
        OMCS_gait_events[f]['AP Acceleration Lumbar left start'] = {}
        OMCS_gait_events[f]['AP Acceleration Lumbar left stop'] = {}
        OMCS_gait_events[f]['AP Acceleration Lumbar right start'] = {}
        OMCS_gait_events[f]['AP Acceleration Lumbar right stop'] = {}
        OMCS_gait_events[f]['AP Deceleration Lumbar left start'] = {}
        OMCS_gait_events[f]['AP Deceleration Lumbar left stop'] = {}
        OMCS_gait_events[f]['AP Deceleration Lumbar right start'] = {}
        OMCS_gait_events[f]['AP Deceleration Lumbar right stop'] = {}
        OMCS_gait_events[f]['Peak AP Acceleration Lumbar left'] = {}
        OMCS_gait_events[f]['Peak AP Acceleration Lumbar right'] = {}
        OMCS_gait_events[f]['Peak AP Deceleration Lumbar left'] = {}
        OMCS_gait_events[f]['Peak AP Deceleration Lumbar right'] = {}
    
        # Prepare OMCS gait characteristics for APaccelerationLumbar function
        OMCS_gait_characteristics[f]['AP Acceleration Lumbar left'] = {}
        OMCS_gait_characteristics[f]['AP Acceleration Lumbar right'] = {}
        OMCS_gait_characteristics[f]['AP Deceleration Lumbar left'] = {}
        OMCS_gait_characteristics[f]['AP Deceleration Lumbar right'] = {}
        OMCS_gait_characteristics[f]['Peak AP Acceleration Lumbar left'] = {}
        OMCS_gait_characteristics[f]['Peak AP Acceleration Lumbar right'] = {}
        OMCS_gait_characteristics[f]['Peak AP Deceleration Lumbar left'] = {}
        OMCS_gait_characteristics[f]['Peak AP Deceleration Lumbar right'] = {}
    
        # Calculate values for OMCS data
        OMCS_gait_events[f], OMCS_gait_characteristics[f], OMCS_AP_ACC_Sacrum[f] = APaccelerationLumbar(OMCS_gait_events[f], OMCS_gait_characteristics[f], OMCS_AP_ACC_Sacrum[f], sample_frequency = OMCS[f]['Sample frequency marker data'], debugplot = False, plot_title = f)

        # Store OMCS data in FinalData
        FinalData[f]['OMCS']['AP Deceleration Lumbar left - VelocityIncrement'] = OMCS_gait_characteristics[f]['AP Deceleration Lumbar left']       # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['AP Deceleration Lumbar left - Peak'] = OMCS_gait_characteristics[f]['Peak AP Deceleration Lumbar left']               # size(n,2) --> Index, Value
        FinalData[f]['OMCS']['AP Acceleration Lumbar left - VelocityIncrement'] = OMCS_gait_characteristics[f]['AP Acceleration Lumbar left']       # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['AP Acceleration Lumbar left - Peak'] = OMCS_gait_characteristics[f]['Peak AP Acceleration Lumbar left']               # size(n,2) --> Index, Value
        
        FinalData[f]['OMCS']['AP Deceleration Lumbar right - VelocityIncrement'] = OMCS_gait_characteristics[f]['AP Deceleration Lumbar right']     # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['AP Deceleration Lumbar right - Peak'] = OMCS_gait_characteristics[f]['Peak AP Deceleration Lumbar right']             # size(n,2) --> Index, Value
        FinalData[f]['OMCS']['AP Acceleration Lumbar right - VelocityIncrement'] = OMCS_gait_characteristics[f]['AP Acceleration Lumbar right']     # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['OMCS']['AP Acceleration Lumbar right - Peak'] = OMCS_gait_characteristics[f]['Peak AP Acceleration Lumbar right']             # size(n,2) --> Index, Value
    except:
        print('Cannot extract OMCS based AP acceleration parameters for trial ', f)
        


# IMU - AP acceleration lumbar
for f in trialnames:
    try:
        # Store IMU gait events
        IMU_gait_events[f]['Index numbers initial contact left'] = {}
        IMU_gait_events[f]['Index numbers terminal contact left'] = {}
        IMU_gait_events[f]['Index numbers initial contact right'] = {}
        IMU_gait_events[f]['Index numbers terminal contact right'] = {}

        IMU_gait_events[f]['Index numbers initial contact left'] = IMU[f]['Left foot']['Gait Events']['Initial Contact']
        IMU_gait_events[f]['Index numbers terminal contact left'] = IMU[f]['Left foot']['Gait Events']['Terminal Contact']
        IMU_gait_events[f]['Index numbers initial contact right'] = IMU[f]['Right foot']['Gait Events']['Initial Contact']
        IMU_gait_events[f]['Index numbers terminal contact right'] = IMU[f]['Right foot']['Gait Events']['Terminal Contact']

        # Prepare IMU gait events for APaccelerationLumbar function
        IMU_gait_events[f]['AP Acceleration Lumbar left start'] = {}
        IMU_gait_events[f]['AP Acceleration Lumbar left stop'] = {}
        IMU_gait_events[f]['AP Acceleration Lumbar right start'] = {}
        IMU_gait_events[f]['AP Acceleration Lumbar right stop'] = {}
        IMU_gait_events[f]['AP Deceleration Lumbar left start'] = {}
        IMU_gait_events[f]['AP Deceleration Lumbar left stop'] = {}
        IMU_gait_events[f]['AP Deceleration Lumbar right start'] = {}
        IMU_gait_events[f]['AP Deceleration Lumbar right stop'] = {}
        IMU_gait_events[f]['Peak AP Acceleration Lumbar left'] = {}
        IMU_gait_events[f]['Peak AP Acceleration Lumbar right'] = {}
        IMU_gait_events[f]['Peak AP Deceleration Lumbar left'] = {}
        IMU_gait_events[f]['Peak AP Deceleration Lumbar right'] = {}
           
        # Prepare IMU gait characteristics for APaccelerationLumbar function
        IMU_gait_characteristics[f] = IMU[f]['Spatiotemporals']

        IMU_gait_characteristics[f]['AP Acceleration Lumbar left'] = {}
        IMU_gait_characteristics[f]['AP Acceleration Lumbar right'] = {}
        IMU_gait_characteristics[f]['AP Deceleration Lumbar left'] = {}
        IMU_gait_characteristics[f]['AP Deceleration Lumbar right'] = {}
        IMU_gait_characteristics[f]['Peak AP Acceleration Lumbar left'] = {}
        IMU_gait_characteristics[f]['Peak AP Acceleration Lumbar right'] = {}
        IMU_gait_characteristics[f]['Peak AP Deceleration Lumbar left'] = {}
        IMU_gait_characteristics[f]['Peak AP Deceleration Lumbar right'] = {}
        
        # Calculate values for IMU data
        IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_BF[f] = APaccelerationLumbar(IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_BF[f], sample_frequency = IMU[f]['Sample Frequency (Hz)'], debugplot = False, plot_title = f)

        # Store IMU data in FinalData
        FinalData[f]['IMU']['AP Deceleration Lumbar left - VelocityIncrement'] = IMU_gait_characteristics[f]['AP Deceleration Lumbar left']         # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['IMU']['AP Deceleration Lumbar left - Peak'] = IMU_gait_characteristics[f]['Peak AP Deceleration Lumbar left']                 # size(n,2) --> Index, Value
        FinalData[f]['IMU']['AP Acceleration Lumbar left - VelocityIncrement'] = IMU_gait_characteristics[f]['AP Acceleration Lumbar left']         # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['IMU']['AP Acceleration Lumbar left - Peak'] = IMU_gait_characteristics[f]['Peak AP Acceleration Lumbar left']                 # size(n,2) --> Index, Value
        
        FinalData[f]['IMU']['AP Deceleration Lumbar right - VelocityIncrement'] = IMU_gait_characteristics[f]['AP Deceleration Lumbar right']       # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['IMU']['AP Deceleration Lumbar right - Peak'] = IMU_gait_characteristics[f]['Peak AP Deceleration Lumbar right']               # size(n,2) --> Index, Value
        FinalData[f]['IMU']['AP Acceleration Lumbar right - VelocityIncrement'] = IMU_gait_characteristics[f]['AP Acceleration Lumbar right']       # size(n,3) --> StartIndex, StopIndex, Value
        FinalData[f]['IMU']['AP Acceleration Lumbar right - Peak'] = IMU_gait_characteristics[f]['Peak AP Acceleration Lumbar right']               # size(n,2) --> Index, Value
    except:
        print('Cannot extract IMU based AP acceleration parameters for trial ', f) 
        
    
# %% Add IMU and OMCS acceleration measures to dataframes df_left and df_right
with open('df_right.pkl', 'rb') as f:
    df_right = pickle.load(f)
with open('df_left.pkl', 'rb') as f:
    df_left = pickle.load(f)
print("Successfully loaded df_left and df_right from the working directory.")

df_left_with_IMU_data = copy.deepcopy(df_left)
df_right_with_IMU_data = copy.deepcopy(df_right)


"""
Left leg
"""
matching_dfs = {}
for key, df in df_left_with_IMU_data.items():
    filename = key[-1].split('/')[-1]  # Extract filename from the key

    if filename in trialnames:
        matching_dfs[filename] = df  # Store the matching DataFrame

for key in list(matching_dfs.keys()):  # Make a copy of the keys before looping
    final_df = matching_dfs[key]


    """ OMCS """
    OMCS_APDecelerationSacrum_VelocityIncrement_left = FinalData[key]['OMCS']['AP Deceleration Lumbar left - VelocityIncrement']
    OMCS_APDecelerationSacrum_Peak_left = FinalData[key]['OMCS']['AP Deceleration Lumbar left - Peak']
    OMCS_APAccelerationSacrum_VelocityIncrement_left = FinalData[key]['OMCS']['AP Acceleration Lumbar left - VelocityIncrement']
    OMCS_APAccelerationSacrum_Peak_left = FinalData[key]['OMCS']['AP Acceleration Lumbar left - Peak']
    
    
    """ Add OMCS_APDecelerationSacrum_VelocityIncrement_left columns to final_df (columns 40, 41 and 42) """
    start_index_column = []
    stop_index_column = []
    value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrum_VelocityIncrement_left rows where stop_index is between IC and TC
            for APDecelerationLumbar_VelocityIncrement in OMCS_APDecelerationSacrum_VelocityIncrement_left:
                start_index, stop_index, value = APDecelerationLumbar_VelocityIncrement
                
                # Check if the stop_index is between IC and TC
                if ic < stop_index < tc:
                    start_index_column.append(start_index)  # Add StartIndex to the list
                    stop_index_column.append(stop_index)  # Add StopIndex to the list
                    value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                start_index_column.append(np.nan)
                stop_index_column.append(np.nan)
                value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            start_index_column.append(np.nan)
            stop_index_column.append(np.nan)
            value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(40, 'OMCS_APDecelerationSacrum_VelocityIncrement_StartIndex_left', start_index_column)  
    final_df.insert(41, 'OMCS_APDecelerationSacrum_VelocityIncrement_StopIndex_left', stop_index_column)  
    final_df.insert(42, 'OMCS_APDecelerationSacrum_VelocityIncrement_Value_left', value_column)  
    
    
    """ Add OMCS_APDecelerationSacrum_Peak_left columns to final_df (columns 43 and 44) """
    APDecelerationLumbar_peak_index_column = []
    APDecelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrum_Peak_left rows where Index is between IC and TC
            for peak in OMCS_APDecelerationSacrum_Peak_left:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APDecelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationLumbar_peak_index_column.append(np.nan)
                APDecelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationLumbar_peak_index_column.append(np.nan)
            APDecelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(43, 'OMCS_APDecelerationSacrum_Peak_Index_left', APDecelerationLumbar_peak_index_column)  
    final_df.insert(44, 'OMCS_APDecelerationSacrum_Peak_Value_left', APDecelerationLumbar_peak_value_column)  

    
    """ Add OMCS_APAccelerationSacrum_VelocityIncrement_left columns to final_df (columns 45, 46 and 47) """
    APAccelerationLumbar_start_index_column = []
    APAccelerationLumbar_stop_index_column = []
    APAccelerationLumbar_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrum_VelocityIncrement_left rows where start_index is between IC and TC
            for APAccelerationLumbar_VelocityIncrement in OMCS_APAccelerationSacrum_VelocityIncrement_left:
                start_index, stop_index, value = APAccelerationLumbar_VelocityIncrement
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationLumbar_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationLumbar_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationLumbar_value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_start_index_column.append(np.nan)
                APAccelerationLumbar_stop_index_column.append(np.nan)
                APAccelerationLumbar_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_start_index_column.append(np.nan)
            APAccelerationLumbar_stop_index_column.append(np.nan)
            APAccelerationLumbar_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(45, 'OMCS_APAccelerationSacrum_VelocityIncrement_StartIndex_left', APAccelerationLumbar_start_index_column)  
    final_df.insert(46, 'OMCS_APAccelerationSacrum_VelocityIncrement_StopIndex_left', APAccelerationLumbar_stop_index_column)  
    final_df.insert(47, 'OMCS_APAccelerationSacrum_VelocityIncrement_Value_left', APAccelerationLumbar_value_column)  
    
    
    """ Add OMCS_APAccelerationSacrum_Peak_left columns to final_df (columns 48 and 49) """
    APAccelerationLumbar_peak_index_column = []
    APAccelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrum_Peak_left rows where Index is between IC and TC
            for peak in OMCS_APAccelerationSacrum_Peak_left:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APAccelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_peak_index_column.append(np.nan)
                APAccelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_peak_index_column.append(np.nan)
            APAccelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(48, 'OMCS_APAccelerationSacrum_Peak_Index_left', APAccelerationLumbar_peak_index_column)  
    final_df.insert(49, 'OMCS_APAccelerationSacrum_Peak_Value_left', APAccelerationLumbar_peak_value_column)  
                                

    """ IMU """  
    IMU_APDecelerationLumbar_VelocityIncrement_left = FinalData[key]['IMU']['AP Deceleration Lumbar left - VelocityIncrement']
    IMU_APDecelerationLumbar_Peak_left = FinalData[key]['IMU']['AP Deceleration Lumbar left - Peak']
    IMU_APAccelerationLumbar_VelocityIncrement_left = FinalData[key]['IMU']['AP Acceleration Lumbar left - VelocityIncrement']
    IMU_APAccelerationLumbar_Peak_left = FinalData[key]['IMU']['AP Acceleration Lumbar left - Peak']
    
    """ Add IMU_APDecelerationLumbar_VelocityIncrement_left columns to final_df (columns 50, 51 and 52) """
    start_index_column = []
    stop_index_column = []
    value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationLumbar_VelocityIncrement_left rows where stop_index is between IC and TC
            for APDecelerationLumbar_VelocityIncrement in IMU_APDecelerationLumbar_VelocityIncrement_left:
                start_index, stop_index, value = APDecelerationLumbar_VelocityIncrement
                
                # Check if the stop_index is between IC and TC
                if ic < stop_index < tc:
                    start_index_column.append(start_index)  # Add StartIndex to the list
                    stop_index_column.append(stop_index)  # Add StopIndex to the list
                    value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                start_index_column.append(np.nan)
                stop_index_column.append(np.nan)
                value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            start_index_column.append(np.nan)
            stop_index_column.append(np.nan)
            value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(50, 'IMU_APDecelerationLumbar_VelocityIncrement_StartIndex_left', start_index_column)  
    final_df.insert(51, 'IMU_APDecelerationLumbar_VelocityIncrement_StopIndex_left', stop_index_column)  
    final_df.insert(52, 'IMU_APDecelerationLumbar_VelocityIncrement_Value_left', value_column)  
    
    
    """ Add IMU_APDecelerationLumbar_Peak_left columns to final_df (columns 53 and 54) """
    APDecelerationLumbar_peak_index_column = []
    APDecelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationLumbar_Peak_left rows where Index is between IC and TC
            for peak in IMU_APDecelerationLumbar_Peak_left:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APDecelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationLumbar_peak_index_column.append(np.nan)
                APDecelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationLumbar_peak_index_column.append(np.nan)
            APDecelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(53, 'IMU_APDecelerationLumbar_Peak_Index_left', APDecelerationLumbar_peak_index_column)  
    final_df.insert(54, 'IMU_APDecelerationLumbar_Peak_Value_left', APDecelerationLumbar_peak_value_column)  

    
    """ Add IMU_APAccelerationLumbar_VelocityIncrement_left columns to final_df (columns 55, 56 and 57) """
    APAccelerationLumbar_start_index_column = []
    APAccelerationLumbar_stop_index_column = []
    APAccelerationLumbar_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationLumbar_VelocityIncrement_left rows where start_index is between IC and TC
            for APAccelerationLumbar_VelocityIncrement in IMU_APAccelerationLumbar_VelocityIncrement_left:
                start_index, stop_index, value = APAccelerationLumbar_VelocityIncrement
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationLumbar_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationLumbar_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationLumbar_value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_start_index_column.append(np.nan)
                APAccelerationLumbar_stop_index_column.append(np.nan)
                APAccelerationLumbar_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_start_index_column.append(np.nan)
            APAccelerationLumbar_stop_index_column.append(np.nan)
            APAccelerationLumbar_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(55, 'IMU_APAccelerationLumbar_VelocityIncrement_StartIndex_left', APAccelerationLumbar_start_index_column)  
    final_df.insert(56, 'IMU_APAccelerationLumbar_VelocityIncrement_StopIndex_left', APAccelerationLumbar_stop_index_column)  
    final_df.insert(57, 'IMU_APAccelerationLumbar_VelocityIncrement_Value_left', APAccelerationLumbar_value_column)  
    
    
    """ Add IMU_APAccelerationLumbar_Peak_left columns to final_df (columns 58 and 59) """
    APAccelerationLumbar_peak_index_column = []
    APAccelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationLumbar_Peak_left rows where Index is between IC and TC
            for peak in IMU_APAccelerationLumbar_Peak_left:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APAccelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_peak_index_column.append(np.nan)
                APAccelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_peak_index_column.append(np.nan)
            APAccelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(58, 'IMU_APAccelerationLumbar_Peak_Index_left', APAccelerationLumbar_peak_index_column)  
    final_df.insert(59, 'IMU_APAccelerationLumbar_Peak_Value_left', APAccelerationLumbar_peak_value_column)  


    matching_dfs[f] = final_df

# Store matching_dfs same format as df_left
for key in df_left_with_IMU_data.keys():
    filename = key[-1].split('/')[-1]  # Extract filename from the key
    
    if filename in matching_dfs:
        df_left_with_IMU_data[key] = matching_dfs[filename]  # Update with modified DataFrame


"""
Right leg
"""
matching_dfs = {}
for key, df in df_right_with_IMU_data.items():
    filename = key[-1].split('/')[-1]  # Extract filename from the key

    if filename in trialnames:
        matching_dfs[filename] = df  # Store the matching DataFrame

for key in list(matching_dfs.keys()):  # Make a copy of the keys before looping
    final_df = matching_dfs[key]
    
    """ OMCS """
    OMCS_APDecelerationSacrum_VelocityIncrement_right = FinalData[key]['OMCS']['AP Deceleration Lumbar right - VelocityIncrement']
    OMCS_APDecelerationSacrum_Peak_right = FinalData[key]['OMCS']['AP Deceleration Lumbar right - Peak']
    OMCS_APAccelerationSacrum_VelocityIncrement_right = FinalData[key]['OMCS']['AP Acceleration Lumbar right - VelocityIncrement']
    OMCS_APAccelerationSacrum_Peak_right = FinalData[key]['OMCS']['AP Acceleration Lumbar right - Peak']
    
    """ Add OMCS_APDecelerationSacrum_VelocityIncrement_right columns to final_df (columns 40, 41 and 42) """
    start_index_column = []
    stop_index_column = []
    value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrum_VelocityIncrement_right rows where stop_index is between IC and TC
            for APDecelerationLumbar_VelocityIncrement in OMCS_APDecelerationSacrum_VelocityIncrement_right:
                start_index, stop_index, value = APDecelerationLumbar_VelocityIncrement
                
                # Check if the stop_index is between IC and TC
                if ic < stop_index < tc:
                    start_index_column.append(start_index)  # Add StartIndex to the list
                    stop_index_column.append(stop_index)  # Add StopIndex to the list
                    value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                start_index_column.append(np.nan)
                stop_index_column.append(np.nan)
                value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            start_index_column.append(np.nan)
            stop_index_column.append(np.nan)
            value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(40, 'OMCS_APDecelerationSacrum_VelocityIncrement_StartIndex_right', start_index_column)  
    final_df.insert(41, 'OMCS_APDecelerationSacrum_VelocityIncrement_StopIndex_right', stop_index_column)  
    final_df.insert(42, 'OMCS_APDecelerationSacrum_VelocityIncrement_Value_right', value_column)  
    
    
    """ Add OMCS_APDecelerationSacrum_Peak_right columns to final_df (columns 43 and 44) """
    APDecelerationLumbar_peak_index_column = []
    APDecelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrum_Peak_right rows where Index is between IC and TC
            for peak in OMCS_APDecelerationSacrum_Peak_right:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APDecelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationLumbar_peak_index_column.append(np.nan)
                APDecelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationLumbar_peak_index_column.append(np.nan)
            APDecelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(43, 'OMCS_APDecelerationSacrum_Peak_Index_right', APDecelerationLumbar_peak_index_column)  
    final_df.insert(44, 'OMCS_APDecelerationSacrum_Peak_Value_right', APDecelerationLumbar_peak_value_column)  

    
    """ Add OMCS_APAccelerationSacrum_VelocityIncrement_right columns to final_df (columns 45, 46 and 47) """
    APAccelerationLumbar_start_index_column = []
    APAccelerationLumbar_stop_index_column = []
    APAccelerationLumbar_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrum_VelocityIncrement_right rows where start_index is between IC and TC
            for APAccelerationLumbar_VelocityIncrement in OMCS_APAccelerationSacrum_VelocityIncrement_right:
                start_index, stop_index, value = APAccelerationLumbar_VelocityIncrement
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationLumbar_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationLumbar_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationLumbar_value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_start_index_column.append(np.nan)
                APAccelerationLumbar_stop_index_column.append(np.nan)
                APAccelerationLumbar_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_start_index_column.append(np.nan)
            APAccelerationLumbar_stop_index_column.append(np.nan)
            APAccelerationLumbar_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(45, 'OMCS_APAccelerationSacrum_VelocityIncrement_StartIndex_right', APAccelerationLumbar_start_index_column)  
    final_df.insert(46, 'OMCS_APAccelerationSacrum_VelocityIncrement_StopIndex_right', APAccelerationLumbar_stop_index_column)  
    final_df.insert(47, 'OMCS_APAccelerationSacrum_VelocityIncrement_Value_right', APAccelerationLumbar_value_column)  
    
    
    """ Add OMCS_APAccelerationSacrum_Peak_right columns to final_df (columns 48 and 49) """
    APAccelerationLumbar_peak_index_column = []
    APAccelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrum_Peak_right rows where Index is between IC and TC
            for peak in OMCS_APAccelerationSacrum_Peak_right:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APAccelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_peak_index_column.append(np.nan)
                APAccelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_peak_index_column.append(np.nan)
            APAccelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(48, 'OMCS_APAccelerationSacrum_Peak_Index_right', APAccelerationLumbar_peak_index_column)  
    final_df.insert(49, 'OMCS_APAccelerationSacrum_Peak_Value_right', APAccelerationLumbar_peak_value_column)  


    """ IMU """
    IMU_APDecelerationLumbar_VelocityIncrement_right = FinalData[key]['IMU']['AP Deceleration Lumbar right - VelocityIncrement']
    IMU_APDecelerationLumbar_Peak_right = FinalData[key]['IMU']['AP Deceleration Lumbar right - Peak']
    IMU_APAccelerationLumbar_VelocityIncrement_right = FinalData[key]['IMU']['AP Acceleration Lumbar right - VelocityIncrement']
    IMU_APAccelerationLumbar_Peak_right = FinalData[key]['IMU']['AP Acceleration Lumbar right - Peak']
    
    """ Add IMU_APDecelerationLumbar_VelocityIncrement_right columns to final_df (columns 50, 51 and 52) """
    start_index_column = []
    stop_index_column = []
    value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationLumbar_VelocityIncrement_right rows where stop_index is between IC and TC
            for APDecelerationLumbar_VelocityIncrement in IMU_APDecelerationLumbar_VelocityIncrement_right:
                start_index, stop_index, value = APDecelerationLumbar_VelocityIncrement
                
                # Check if the stop_index is between IC and TC
                if ic < stop_index < tc:
                    start_index_column.append(start_index)  # Add StartIndex to the list
                    stop_index_column.append(stop_index)  # Add StopIndex to the list
                    value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                start_index_column.append(np.nan)
                stop_index_column.append(np.nan)
                value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            start_index_column.append(np.nan)
            stop_index_column.append(np.nan)
            value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(50, 'IMU_APDecelerationLumbar_VelocityIncrement_StartIndex_right', start_index_column)  
    final_df.insert(51, 'IMU_APDecelerationLumbar_VelocityIncrement_StopIndex_right', stop_index_column)  
    final_df.insert(52, 'IMU_APDecelerationLumbar_VelocityIncrement_Value_right', value_column)  
    
    
    """ Add IMU_APDecelerationLumbar_Peak_right columns to final_df (columns 53 and 54) """
    APDecelerationLumbar_peak_index_column = []
    APDecelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationLumbar_Peak_right rows where Index is between IC and TC
            for peak in IMU_APDecelerationLumbar_Peak_right:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APDecelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationLumbar_peak_index_column.append(np.nan)
                APDecelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationLumbar_peak_index_column.append(np.nan)
            APDecelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(53, 'IMU_APDecelerationLumbar_Peak_Index_right', APDecelerationLumbar_peak_index_column)  
    final_df.insert(54, 'IMU_APDecelerationLumbar_Peak_Value_right', APDecelerationLumbar_peak_value_column)  

    
    """ Add IMU_APAccelerationLumbar_VelocityIncrement_right columns to final_df (columns 55, 56 and 57) """
    APAccelerationLumbar_start_index_column = []
    APAccelerationLumbar_stop_index_column = []
    APAccelerationLumbar_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationLumbar_VelocityIncrement_right rows where start_index is between IC and TC
            for APAccelerationLumbar_VelocityIncrement in IMU_APAccelerationLumbar_VelocityIncrement_right:
                start_index, stop_index, value = APAccelerationLumbar_VelocityIncrement
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationLumbar_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationLumbar_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationLumbar_value_column.append(value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_start_index_column.append(np.nan)
                APAccelerationLumbar_stop_index_column.append(np.nan)
                APAccelerationLumbar_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_start_index_column.append(np.nan)
            APAccelerationLumbar_stop_index_column.append(np.nan)
            APAccelerationLumbar_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(55, 'IMU_APAccelerationLumbar_VelocityIncrement_StartIndex_right', APAccelerationLumbar_start_index_column)  
    final_df.insert(56, 'IMU_APAccelerationLumbar_VelocityIncrement_StopIndex_right', APAccelerationLumbar_stop_index_column)  
    final_df.insert(57, 'IMU_APAccelerationLumbar_VelocityIncrement_Value_right', APAccelerationLumbar_value_column)  
    
    
    """ Add IMU_APAccelerationLumbar_Peak_right columns to final_df (columns 58 and 59) """
    APAccelerationLumbar_peak_index_column = []
    APAccelerationLumbar_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationLumbar_Peak_right rows where Index is between IC and TC
            for peak in IMU_APAccelerationLumbar_Peak_right:
                peak_index, peak_value = peak
                
                # Check if the Index is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationLumbar_peak_index_column.append(peak_index)  # Add Index to the list
                    APAccelerationLumbar_peak_value_column.append(peak_value)  # Add Value to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationLumbar_peak_index_column.append(np.nan)
                APAccelerationLumbar_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationLumbar_peak_index_column.append(np.nan)
            APAccelerationLumbar_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(58, 'IMU_APAccelerationLumbar_Peak_Index_right', APAccelerationLumbar_peak_index_column)  
    final_df.insert(59, 'IMU_APAccelerationLumbar_Peak_Value_right', APAccelerationLumbar_peak_value_column)  

                                
    matching_dfs[f] = final_df

# Store matching_dfs same format as df_right
for key in df_right_with_IMU_data.keys():
    filename = key[-1].split('/')[-1]  # Extract filename from the key
    
    if filename in matching_dfs:
        df_right_with_IMU_data[key] = matching_dfs[filename]  # Update with modified DataFrame


# Save df_right_with_IMU_data and df_left_with_IMU_data to the working directory
with open('df_right_with_IMU_data.pkl', 'wb') as f:
    pickle.dump(df_right_with_IMU_data, f)
print("Successfully saved df_right_with_IMU_data to the working directory.")
with open('df_left_with_IMU_data.pkl', 'wb') as f:
    pickle.dump(df_left_with_IMU_data, f)
print("Successfully saved df_left_with_IMU_data to the working directory.")

            
# %% Create dataframe with all trials for Excel-sheet needed for analyses in RStudio
R_dataset_df_right = pd.concat(df_right_with_IMU_data.values(), ignore_index=True)
R_dataset_df_left = pd.concat(df_left_with_IMU_data.values(), ignore_index=True)
            

# %% Debug plots
if debugplot == True:
    
    from helpers_LumbarAccelerationIMU import plot_Acceleration_OMCS_and_IMU, plot_AP_GRF_and_AP_ACC, plot_Walking_Direction, plot_AP_GRF_and_AP_ACC_with_GaitEvents

    trial = '900_V_pp01_SP01.c3d'
    lower_X_lim = 2800
    upper_X_lim = 3000
    
    # Acceleration signal for OMCS, IMU_SF, IMU_EF, IMU_BF
    plot_Acceleration_OMCS_and_IMU(OMCS_ACC_Sacrum, IMU_ACC_SF_Lumbar, IMU_ACC_EF_Lumbar, IMU_ACC_BF_Lumbar, trial, lower_X_lim, upper_X_lim)
    
    # Walking direction from PCA; needed for calculation of IMU ACC in body frame (rotate earth frame ACC)
    plot_Walking_Direction(walking_directions)
    
    # AP-GRF, AP-ACC OMCS, AP-ACC IMU overlayed in a single plot
    plot_AP_GRF_and_AP_ACC(OMCS, OMCS_AP_GRF_left, OMCS_AP_GRF_right, OMCS_AP_ACC_Sacrum, IMU_AP_ACC_Lumbar_BF, trial, lower_X_lim, upper_X_lim)
    
    # AP-GRF, AP-ACC OMCS, AP-ACC IMU in seperate subplots with vertical lines for gait events
    plot_AP_GRF_and_AP_ACC_with_GaitEvents(OMCS, OMCS_gait_events, OMCS_gait_characteristics, IMU_gait_events, IMU_gait_characteristics, OMCS_AP_GRF_left, OMCS_AP_GRF_right, OMCS_AP_ACC_Sacrum, IMU_AP_ACC_Lumbar_BF, trial, lower_X_lim, upper_X_lim)

    for trial in trialnames:        
        for i in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]:
            plot_AP_GRF_and_AP_ACC_with_GaitEvents(OMCS, OMCS_gait_events, OMCS_gait_characteristics, IMU_gait_events, IMU_gait_characteristics, OMCS_AP_GRF_left, OMCS_AP_GRF_right, OMCS_AP_ACC_Sacrum, IMU_AP_ACC_Lumbar_BF, trial, i, i+1000)

