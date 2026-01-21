"""
Extends the OMCS step-by-step dataframe by adding anterior-posterior (AP) acceleration impulses and peaks calculated from OMCS and IMU data.

Version - Author:
    2025: Lars van Rengs - l.vanrengs@maartenskliniek.nl
"""

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import os
import copy

from helpers_APacceleration import dataimport, analyze_OMCS, OMCS_calculate_sacrum_acceleration, filter_data, APaccelerationLumbar


""" Import data """
# ---------- USER INPUTS REQUIRED ---------- #
# Set wether or not you want to create plot of the data (debugplot = True / False)
debugplot = False

# Set trialtype you wish to analyze to 'True'
analyze_trialtypes = dict()
analyze_trialtypes['Healthy GRAIL'] = True
analyze_trialtypes['Healthy Lab'] = False
analyze_trialtypes['CVA GRAIL'] = True
analyze_trialtypes['CVA_feedback GRAIL'] = True

# Set wether or not a saved .pkl file can be found in your directory (storedfile = True / False)
storedfile = False

if storedfile == True:
    filename = 'dataset_APacceleration.pkl'
elif storedfile == False:
    # Define filepaths for vicon and xsens data
    datafolder_validation_study = os.path.abspath('IMU_GaitAnalysis/data')
    datafolder_feedback_study = os.path.abspath('MovingReality/data')
    # Set name for file to be saved
    save_as = 'dataset_APacceleration.pkl' 

frame = 'body frame'               # 'sensor frame' or 'earth frame' or 'body frame'
gait_events_for_IMU = 'IMU'        # 'OMCS' or 'IMU'

        
# ---------- END USER INPUTS REQUIRED ---------- #

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

# Notes:
#   * 900_CVA_03 was unable to perform stepping stones trial --> This person performed 2 regular walking trials, remove one for further analysis ('900_CVA_03_FS02.c3d')
#   * 900_CVA_04_SP01.c3d >> walking mostly on one of the treadmill bands, not viable for gait event detection.
#   * 900_CVA_09 had to perform regular walking condition at a fixed treadmill speed ('900_CVA_09_FS01.c3d'); all other participants performed regular walking trial in self-paced mode
#   * '900_V_pp01_SP03.c3d': Fixed speed trial, accidentally wrongly named
#   * 900_V_pp07_SP01.c3d --> OMCS data is missing
#   * 1019_MR003_FBIC.c3d --> No xsens data available; recording error
#   * 1019_MR006_FBIC was removed from further analysis due to poor marker visibility
#   * 1019_MR006_2Reg02 --> Xsens data; recording error

exclude_trials = [
    # '900_CVA_03_FS01.c3d',
    '900_CVA_03_FS02.c3d',
    '900_CVA_04_SP01.c3d',
    # '900_CVA_09_FS01.c3d',
    '900_V_pp01_SP03.c3d',
    '900_V_pp07_SP01.c3d',
    '1019_MR003_FBIC.c3d',
    '1019_MR006_FBIC.c3d',
    '1019_MR006_2Reg02.c3d'
]
trialnames = [t for t in trialnames if t not in exclude_trials]


""" Select trial types of interest for plots """
def get_trial_type(trial):
    trial_id = trial.replace(".c3d", "")
    if "_FS0" in trial:
        return trial_id, "FS"
    elif "_SS0" in trial:
        return trial_id, "SS"
    elif "_SP0" in trial:
        return trial_id, "SP"
    elif "_1Reg" in trial:
        return trial_id, "1Reg"
    elif "_Reg" in trial:
        return trial_id, "1Reg"
    elif "_FBIC" in trial:
        return trial_id, "FBIC"
    elif "_FBPO" in trial:
        return trial_id, "FBPO"
    elif "_2FB" in trial:
        return trial_id, "2FB"
    elif "_2Reg" in trial:
        return trial_id, "2Reg"
    else:
        return trial_id, None

# Apply filtering
filtered_trials = []
for trial in trialnames:
    trial_id, trial_type = get_trial_type(trial)
    if trial_type is not None:
        filtered_trials.append((trial_id, trial_type))
 
# Extract only SP and 1Reg
interested_trial_types = {"SP", "1Reg"}

selected_trialnames = []
for trial in trialnames:
    trial_id, trial_type = get_trial_type(trial)
    if trial_type in interested_trial_types:
        selected_trialnames.append(trial)



""" Extract parameters from OMCS-data """
OMCS, OMCS_gait_events, OMCS_gait_characteristics = analyze_OMCS(OMCS, IMU, trialnames)



""" Extract OMCS AP-GRF """
OMCS_AP_GRF_left = dict()
OMCS_AP_GRF_right = dict()
for f in OMCS:
    try:
        OMCS_AP_GRF_left[f] = OMCS[f]['Analog data']['Force Y left filtered']
        OMCS_AP_GRF_right[f] = OMCS[f]['Analog data']['Force Y right filtered']       
    except:
        print('Cannot extract AP-GRF for trial ', f) 


        
""" Calculate OMCS acceleration """
OMCS_POS_Sacrum, OMCS_ACC_Sacrum = OMCS_calculate_sacrum_acceleration(OMCS)
        


""" Extract IMU acceleration """
IMU_ACC_SF_Lumbar = dict()
for f in IMU:
    try:
        IMU_ACC_SF_Lumbar[f] = IMU[f]['Lumbar']['raw']['Accelerometer Sensor Frame']
        
        IMU_ACCx_SF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_SF_Lumbar[f][:,0])   # filter signal: order, fcut, fs, signal
        IMU_ACCy_SF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_SF_Lumbar[f][:,1])   # filter signal: order, fcut, fs, signal
        IMU_ACCz_SF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_SF_Lumbar[f][:,2])   # filter signal: order, fcut, fs, signal
        IMU_ACC_SF_Lumbar[f] = np.column_stack((IMU_ACCx_SF, IMU_ACCy_SF, IMU_ACCz_SF))       
    except:
        print('Cannot extract IMU based acceleration (Sensor Frame) for trial ', f) 

IMU_ACC_EF_Lumbar = dict()
for f in IMU:
    try:
        IMU_ACC_EF_Lumbar[f] = IMU[f]['Lumbar']['raw']['Accelerometer Earth Frame']
        
        IMU_ACCx_EF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_EF_Lumbar[f][:,0])   # filter signal: order, fcut, fs, signal
        IMU_ACCy_EF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_EF_Lumbar[f][:,1])   # filter signal: order, fcut, fs, signal
        IMU_ACCz_EF = filter_data(2, 17, IMU[f]['Sample Frequency (Hz)'], IMU_ACC_EF_Lumbar[f][:,2])   # filter signal: order, fcut, fs, signal
        IMU_ACC_EF_Lumbar[f] = np.column_stack((IMU_ACCx_EF, IMU_ACCy_EF, IMU_ACCz_EF))
    except:
        print('Cannot extract IMU based acceleration (Earth Frame) for trial ', f) 



""" Calculate IMU acceleration in body frame """
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



""" Save OMCS and IMU acceleration to working directory """
with open('trialnames.pkl', 'wb') as f:
    pickle.dump(trialnames, f)
with open('OMCS_AP_GRF_left.pkl', 'wb') as f:
    pickle.dump(OMCS_AP_GRF_left, f)
with open('OMCS_AP_GRF_right.pkl', 'wb') as f:
    pickle.dump(OMCS_AP_GRF_right, f)
with open('OMCS_ACC_Sacrum.pkl', 'wb') as f:
    pickle.dump(OMCS_ACC_Sacrum, f)
# with open('IMU_ACC_SF_Lumbar.pkl', 'wb') as f:
#     pickle.dump(IMU_ACC_SF_Lumbar, f)
# with open('IMU_ACC_EF_Lumbar.pkl', 'wb') as f:
#     pickle.dump(IMU_ACC_EF_Lumbar, f)
with open('IMU_ACC_BF_Lumbar.pkl', 'wb') as f:
    pickle.dump(IMU_ACC_BF_Lumbar, f)    



""" Compare IMU acceleration data with Vicon acceleration data """
if debugplot == True:
    for trial in selected_trialnames:   # Plot AP acceleration data for the first couple of trials as an example
        try:
            OMCS_ACC_data = OMCS_ACC_Sacrum[trial]
            IMU_ACC_data_SF = IMU_ACC_SF_Lumbar[trial]
            IMU_ACC_data_EF = IMU_ACC_EF_Lumbar[trial]
            IMU_ACC_data_BF = IMU_ACC_BF_Lumbar[trial]

            fig, axs = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
            fig.suptitle(f"Acceleration Data for Trial: {trial}", fontsize=16, fontweight='bold')

            # OMCS X-axis
            axs[0, 0].plot(OMCS_ACC_data[:, 0], label='X-axis', color='r')
            axs[0, 0].axhline(np.nanmean(OMCS_ACC_data[:, 0]), color='k', linestyle='--', label='Mean')
            axs[0, 0].set_ylabel('Acceleration (mm/s²)')
            axs[0, 0].set_title('OMCS Acc Sacrum - X-axis')
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            # OMCS Y-axis
            axs[1, 0].plot(OMCS_ACC_data[:, 1], label='Y-axis', color='g')
            axs[1, 0].axhline(np.nanmean(OMCS_ACC_data[:, 1]), color='k', linestyle='--', label='Mean')
            axs[1, 0].set_ylabel('Acceleration (mm/s²)')
            axs[1, 0].set_title('OMCS Acc Sacrum - Y-axis')
            axs[1, 0].grid(True)
            axs[1, 0].legend()

            # OMCS Z-axis
            axs[2, 0].plot(OMCS_ACC_data[:, 2], label='Z-axis', color='b')
            axs[2, 0].axhline(np.nanmean(OMCS_ACC_data[:, 2]), color='k', linestyle='--', label='Mean')
            axs[2, 0].set_xlabel('Time (samples)')
            axs[2, 0].set_ylabel('Acceleration (mm/s²)')
            axs[2, 0].set_title('OMCS Acc Sacrum - Z-axis')
            axs[2, 0].grid(True)
            axs[2, 0].legend()

            # IMU SF X-axis
            axs[0, 1].plot(IMU_ACC_data_SF[:, 0], label='X-axis', color='r')
            axs[0, 1].axhline(np.nanmean(IMU_ACC_data_SF[:, 0]), color='k', linestyle='--', label='Mean')
            axs[0, 1].set_ylabel('Acceleration (m/s²)')
            axs[0, 1].set_title('IMU Acc Lumbar (Sensor Frame) - X-axis')
            axs[0, 1].grid(True)
            axs[0, 1].legend()

            # IMU SF Y-axis
            axs[1, 1].plot(IMU_ACC_data_SF[:, 1], label='Y-axis', color='g')
            axs[1, 1].axhline(np.nanmean(IMU_ACC_data_SF[:, 1]), color='k', linestyle='--', label='Mean')
            axs[1, 1].set_ylabel('Acceleration (m/s²)')
            axs[1, 1].set_title('IMU Acc Lumbar (Sensor Frame) - Y-axis')
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            # IMU SF Z-axis
            axs[2, 1].plot(IMU_ACC_data_SF[:, 2], label='Z-axis', color='b')
            axs[2, 1].axhline(np.nanmean(IMU_ACC_data_SF[:, 2]), color='k', linestyle='--', label='Mean')
            axs[2, 1].set_xlabel('Time (samples)')
            axs[2, 1].set_ylabel('Acceleration (m/s²)')
            axs[2, 1].set_title('IMU Acc Lumbar (Sensor Frame) - Z-axis')
            axs[2, 1].grid(True)
            axs[2, 1].legend()
            
            # IMU EF X-axis
            axs[0, 2].plot(IMU_ACC_data_EF[:, 0], label='X-axis', color='r')
            axs[0, 2].axhline(np.nanmean(IMU_ACC_data_EF[:, 0]), color='k', linestyle='--', label='Mean')
            axs[0, 2].set_ylabel('Acceleration (m/s²)')
            axs[0, 2].set_title('IMU Acc Lumbar (Earth Frame) - X-axis')
            axs[0, 2].grid(True)
            axs[0, 2].legend()

            # IMU EF Y-axis
            axs[1, 2].plot(IMU_ACC_data_EF[:, 1], label='Y-axis', color='g')
            axs[1, 2].axhline(np.nanmean(IMU_ACC_data_EF[:, 1]), color='k', linestyle='--', label='Mean')
            axs[1, 2].set_ylabel('Acceleration (m/s²)')
            axs[1, 2].set_title('IMU Acc Lumbar (Earth Frame) - Y-axis')
            axs[1, 2].grid(True)
            axs[1, 2].legend()

            # IMU EF Z-axis
            axs[2, 2].plot(IMU_ACC_data_EF[:, 2], label='Z-axis', color='b')
            axs[2, 2].axhline(np.nanmean(IMU_ACC_data_EF[:, 2]), color='k', linestyle='--', label='Mean')
            axs[2, 2].set_xlabel('Time (samples)')
            axs[2, 2].set_ylabel('Acceleration (m/s²)')
            axs[2, 2].set_title('IMU Acc Lumbar (Earth Frame) - Z-axis')
            axs[2, 2].grid(True)
            axs[2, 2].legend()

            # IMU BF X-axis
            axs[0, 3].plot(IMU_ACC_data_BF[:, 0], label='X-axis', color='r')
            axs[0, 3].axhline(np.nanmean(IMU_ACC_data_BF[:, 0]), color='k', linestyle='--', label='Mean')
            axs[0, 3].set_ylabel('Acceleration (m/s²)')
            axs[0, 3].set_title('IMU Acc Lumbar (Body Frame) - X-axis')
            axs[0, 3].grid(True)
            axs[0, 3].legend()

            # IMU BF Y-axis
            axs[1, 3].plot(IMU_ACC_data_BF[:, 1], label='Y-axis', color='g')
            axs[1, 3].axhline(np.nanmean(IMU_ACC_data_BF[:, 1]), color='k', linestyle='--', label='Mean')
            axs[1, 3].set_ylabel('Acceleration (m/s²)')
            axs[1, 3].set_title('IMU Acc Lumbar (Body Frame) - Y-axis')
            axs[1, 3].grid(True)
            axs[1, 3].legend()

            # IMU BF Z-axis
            axs[2, 3].plot(IMU_ACC_data_BF[:, 2], label='Z-axis', color='b')
            axs[2, 3].axhline(np.nanmean(IMU_ACC_data_BF[:, 2]), color='k', linestyle='--', label='Mean')
            axs[2, 3].set_xlabel('Time (samples)')
            axs[2, 3].set_ylabel('Acceleration (m/s²)')
            axs[2, 3].set_title('IMU Acc Lumbar (Body Frame) - Z-axis')
            axs[2, 3].grid(True)
            axs[2, 3].legend()

            # X-axis limits
            for row in axs:
                for ax in row:
                    ax.set_xlim(3000, 4000)

            plt.tight_layout()
            plt.show()

        except:
            print(f"Cannot plot acceleration data for trial: {trial}")



""" Extract AP acceleration for OMCS and IMU """
OMCS_AP_axis = 1
OMCS_AP_ACC_Sacrum = dict()
for f in OMCS:
    try:
        OMCS_AP_ACC_Sacrum[f] = OMCS_ACC_Sacrum[f][:, OMCS_AP_axis]
    except:
        print('Cannot extract AP acceleration of OMCS for trial ', f) 

IMU_AP_axis_SF = 2
IMU_AP_ACC_Lumbar_SF = dict()
for f in IMU:
    try:
        IMU_AP_ACC_Lumbar_SF[f] = IMU_ACC_SF_Lumbar[f][:, IMU_AP_axis_SF] - np.nanmean(IMU_ACC_SF_Lumbar[f][:, IMU_AP_axis_SF])         # Set IMU_AP_ACC_Lumbar to the same format as OMCS_AP_ACC_Sacrum
    except:
        print('Cannot extract AP acceleration (Sensor Frame) of IMU for trial ', f) 

IMU_AP_axis_EF = 0
IMU_AP_ACC_Lumbar_EF = dict()
for f in IMU:
    try:
        IMU_AP_ACC_Lumbar_EF[f] = -IMU_ACC_EF_Lumbar[f][:, IMU_AP_axis_EF]         # Set IMU_AP_ACC_Lumbar to the same format as OMCS_AP_ACC_Sacrum
    except:
        print('Cannot extract AP acceleration (Earth Frame) of IMU for trial ', f) 

IMU_AP_axis_BF = 0
IMU_AP_ACC_Lumbar_BF = dict()
for f in IMU:
    try:
        IMU_AP_ACC_Lumbar_BF[f] = -IMU_ACC_BF_Lumbar[f][:, IMU_AP_axis_BF]         # Set IMU_AP_ACC_Lumbar to the same format as OMCS_AP_ACC_Sacrum
    except:
        print('Cannot extract AP acceleration (Body Frame) of IMU for trial ', f) 
        
if debugplot == True:
    for trial in selected_trialnames:
        try:       
            OMCS_AP_GRF_data_left = OMCS_AP_GRF_left[trial]           
            OMCS_AP_GRF_data_right = OMCS_AP_GRF_right[trial]           
            OMCS_ACC_data = OMCS_AP_ACC_Sacrum[trial]           
            IMU_ACC_data_SF = IMU_AP_ACC_Lumbar_SF[trial]
            IMU_ACC_data_EF = IMU_AP_ACC_Lumbar_EF[trial]
            IMU_ACC_data_BF = IMU_AP_ACC_Lumbar_BF[trial]
              
            plt.figure(figsize=(10, 6))
            plt.plot(-OMCS_AP_GRF_data_left/69, label='AP-GRF left')
            plt.plot(-OMCS_AP_GRF_data_right/69, label='AP-GRF right')
            plt.plot(-OMCS_ACC_data/1000, label='OMCS')
            # plt.plot(IMU_ACC_data_SF, label='IMU_SF')
            # plt.plot(IMU_ACC_data_EF, label='IMU_EF')
            # plt.plot(IMU_ACC_data_BF, label='IMU_BF')
            plt.plot(IMU_ACC_BF_Lumbar[trial][:, 0], label='IMU_BF_X')
            # plt.plot(-IMU_ACC_BF_Lumbar[trial][:, 1], label='IMU_BF_Y')
            plt.xlabel('Time (samples)')
            plt.ylabel('Acceleration (m/s^2)')
            plt.title(f"Acceleration Data for Trial: {trial}")
            plt.xlim(2200, 2400)
            # plt.xlim(2240, 2320)
            plt.legend()
            plt.grid()
            plt.show()
        except:
            print(f"Cannot plot AP acceleration data for trial: {trial}") 
            


""" Walking direction plot """
# Now filter walking_direction
filtered_walking_directions = {
    k: v for k, v in walking_directions.items() if k in selected_trialnames
}   

if debugplot == True:
    # Prepare polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    for trial, angle_deg in filtered_walking_directions.items():
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)
    
        # Plot a line for each trial (arbitrary length = 1)
        ax.plot([angle_rad, angle_rad], [0, 1], label=trial, lw=1)
    
    # Add labels
    ax.set_theta_zero_location('E')  # 0° = east (X-axis)
    ax.set_theta_direction(-1)       # clockwise
    
    plt.title("Walking Directions for All Trials")
    plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.05))
    plt.show()
    
    
    
""" Extract acceleration peak/impulse for OMCS and IMU """
# Prepare dictornary
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

    FinalData[f]['OMCS']['AP Deceleration Sacrum left - Impulse'] = {}
    FinalData[f]['OMCS']['AP Deceleration Sacrum left - Peak'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum left - Impulse'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum left - Peak'] = {}
    FinalData[f]['OMCS']['AP Deceleration Sacrum right - Impulse'] = {}
    FinalData[f]['OMCS']['AP Deceleration Sacrum right - Peak'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum right - Impulse'] = {}
    FinalData[f]['OMCS']['AP Acceleration Sacrum right - Peak'] = {}

    FinalData[f]['IMU']['AP Deceleration Sacrum left - Impulse'] = {}
    FinalData[f]['IMU']['AP Deceleration Sacrum left - Peak'] = {}
    FinalData[f]['IMU']['AP Acceleration Sacrum left - Impulse'] = {}
    FinalData[f]['IMU']['AP Acceleration Sacrum left - Peak'] = {}
    FinalData[f]['IMU']['AP Deceleration Sacrum right - Impulse'] = {}
    FinalData[f]['IMU']['AP Deceleration Sacrum right - Peak'] = {}
    FinalData[f]['IMU']['AP Acceleration Sacrum right - Impulse'] = {}
    FinalData[f]['IMU']['AP Acceleration Sacrum right - Peak'] = {}


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
        FinalData[f]['OMCS']['Braking left - Impulse'] = OMCS_gait_characteristics[f]['Braking left']                   # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['Braking left - Peak'] = OMCS_gait_characteristics[f]['Peak braking left']                 # size(n,2) --> PeakIndex, PeakValue
        FinalData[f]['OMCS']['Propulsion left - Impulse'] = OMCS_gait_characteristics[f]['Propulsion left']             # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['Propulsion left - Peak'] = OMCS_gait_characteristics[f]['Peak propulsion left']           # size(n,2) --> PeakIndex, PeakValue
        
        FinalData[f]['OMCS']['Braking right - Impulse'] = OMCS_gait_characteristics[f]['Braking right']                 # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['Braking right - Peak'] = OMCS_gait_characteristics[f]['Peak braking right']               # size(n,2) --> PeakIndex, PeakValue
        FinalData[f]['OMCS']['Propulsion right - Impulse'] = OMCS_gait_characteristics[f]['Propulsion right']           # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['Propulsion right - Peak'] = OMCS_gait_characteristics[f]['Peak propulsion right']         # size(n,2) --> PeakIndex, PeakValue
    except:
        print('Cannot extract OMCS based propulsion force parameters for trial ', f) 


   
# OMCS - AP acceleration sacrum
for f in trialnames:
    try:
        # Prepare OMCS gait events for APaccelerationLumbar function
        OMCS_gait_events[f]['AP Acceleration Sacrum left start'] = {}
        OMCS_gait_events[f]['AP Acceleration Sacrum left stop'] = {}
        OMCS_gait_events[f]['AP Acceleration Sacrum right start'] = {}
        OMCS_gait_events[f]['AP Acceleration Sacrum right stop'] = {}
        OMCS_gait_events[f]['AP Deceleration Sacrum left start'] = {}
        OMCS_gait_events[f]['AP Deceleration Sacrum left stop'] = {}
        OMCS_gait_events[f]['AP Deceleration Sacrum right start'] = {}
        OMCS_gait_events[f]['AP Deceleration Sacrum right stop'] = {}
        OMCS_gait_events[f]['Peak AP Acceleration Sacrum left'] = {}
        OMCS_gait_events[f]['Peak AP Acceleration Sacrum right'] = {}
        OMCS_gait_events[f]['Peak AP Deceleration Sacrum left'] = {}
        OMCS_gait_events[f]['Peak AP Deceleration Sacrum right'] = {}
    
        # Prepare OMCS gait characteristics for APaccelerationLumbar function
        OMCS_gait_characteristics[f]['AP Acceleration Sacrum left'] = {}
        OMCS_gait_characteristics[f]['AP Acceleration Sacrum right'] = {}
        OMCS_gait_characteristics[f]['AP Deceleration Sacrum left'] = {}
        OMCS_gait_characteristics[f]['AP Deceleration Sacrum right'] = {}
        OMCS_gait_characteristics[f]['Peak AP Acceleration Sacrum left'] = {}
        OMCS_gait_characteristics[f]['Peak AP Acceleration Sacrum right'] = {}
        OMCS_gait_characteristics[f]['Peak AP Deceleration Sacrum left'] = {}
        OMCS_gait_characteristics[f]['Peak AP Deceleration Sacrum right'] = {}
    
        # Calculate peak/impulse values for OMCS data
        OMCS_gait_events[f], OMCS_gait_characteristics[f], OMCS_AP_ACC_Sacrum[f] = APaccelerationLumbar(OMCS_gait_events[f], OMCS_gait_characteristics[f], OMCS_AP_ACC_Sacrum[f], sample_frequency = OMCS[f]['Sample frequency marker data'], debugplot = debugplot, plot_title = f)

        # Store OMCS data in FinalData
        FinalData[f]['OMCS']['AP Deceleration Sacrum left - Impulse'] = OMCS_gait_characteristics[f]['AP Deceleration Sacrum left']           # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['AP Deceleration Sacrum left - Peak'] = OMCS_gait_characteristics[f]['Peak AP Deceleration Sacrum left']         # size(n,2) --> PeakIndex, PeakValue
        FinalData[f]['OMCS']['AP Acceleration Sacrum left - Impulse'] = OMCS_gait_characteristics[f]['AP Acceleration Sacrum left']           # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['AP Acceleration Sacrum left - Peak'] = OMCS_gait_characteristics[f]['Peak AP Acceleration Sacrum left']         # size(n,2) --> PeakIndex, PeakValue
        
        FinalData[f]['OMCS']['AP Deceleration Sacrum right - Impulse'] = OMCS_gait_characteristics[f]['AP Deceleration Sacrum right']         # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['AP Deceleration Sacrum right - Peak'] = OMCS_gait_characteristics[f]['Peak AP Deceleration Sacrum right']       # size(n,2) --> PeakIndex, PeakValue
        FinalData[f]['OMCS']['AP Acceleration Sacrum right - Impulse'] = OMCS_gait_characteristics[f]['AP Acceleration Sacrum right']         # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['OMCS']['AP Acceleration Sacrum right - Peak'] = OMCS_gait_characteristics[f]['Peak AP Acceleration Sacrum right']       # size(n,2) --> PeakIndex, PeakValue
    except:
        print('Cannot extract OMCS based AP acceleration parameters for trial ', f) 
        


# IMU - AP acceleration lumbar
for f in trialnames:
    try:
        # Prepare IMU gait events for APaccelerationLumbar function
        IMU_gait_events[f]['Index numbers initial contact left'] = {}
        IMU_gait_events[f]['Index numbers terminal contact left'] = {}
        IMU_gait_events[f]['Index numbers initial contact right'] = {}
        IMU_gait_events[f]['Index numbers terminal contact right'] = {}
        IMU_gait_events[f]['AP Acceleration Sacrum left start'] = {}
        IMU_gait_events[f]['AP Acceleration Sacrum left stop'] = {}
        IMU_gait_events[f]['AP Acceleration Sacrum right start'] = {}
        IMU_gait_events[f]['AP Acceleration Sacrum right stop'] = {}
        IMU_gait_events[f]['AP Deceleration Sacrum left start'] = {}
        IMU_gait_events[f]['AP Deceleration Sacrum left stop'] = {}
        IMU_gait_events[f]['AP Deceleration Sacrum right start'] = {}
        IMU_gait_events[f]['AP Deceleration Sacrum right stop'] = {}
        IMU_gait_events[f]['Peak AP Acceleration Sacrum left'] = {}
        IMU_gait_events[f]['Peak AP Acceleration Sacrum right'] = {}
        IMU_gait_events[f]['Peak AP Deceleration Sacrum left'] = {}
        IMU_gait_events[f]['Peak AP Deceleration Sacrum right'] = {}
           
        if gait_events_for_IMU == 'IMU':
            IMU_gait_events[f]['Index numbers initial contact left'] = IMU[f]['Left foot']['Gait Events']['Initial Contact']
            IMU_gait_events[f]['Index numbers terminal contact left'] = IMU[f]['Left foot']['Gait Events']['Terminal Contact']
            IMU_gait_events[f]['Index numbers initial contact right'] = IMU[f]['Right foot']['Gait Events']['Initial Contact']
            IMU_gait_events[f]['Index numbers terminal contact right'] = IMU[f]['Right foot']['Gait Events']['Terminal Contact']
        elif gait_events_for_IMU == 'OMCS':
            IMU_gait_events[f]['Index numbers initial contact left'] = OMCS_gait_events[f]['Index numbers initial contact left']
            IMU_gait_events[f]['Index numbers terminal contact left'] = OMCS_gait_events[f]['Index numbers terminal contact left']
            IMU_gait_events[f]['Index numbers initial contact right'] = OMCS_gait_events[f]['Index numbers initial contact right']
            IMU_gait_events[f]['Index numbers terminal contact right'] = OMCS_gait_events[f]['Index numbers terminal contact right']
                      
        # Prepare IMU gait characteristics for APaccelerationLumbar function
        IMU_gait_characteristics[f] = IMU[f]['Spatiotemporals']

        IMU_gait_characteristics[f]['AP Acceleration Sacrum left'] = {}
        IMU_gait_characteristics[f]['AP Acceleration Sacrum right'] = {}
        IMU_gait_characteristics[f]['AP Deceleration Sacrum left'] = {}
        IMU_gait_characteristics[f]['AP Deceleration Sacrum right'] = {}
        IMU_gait_characteristics[f]['Peak AP Acceleration Sacrum left'] = {}
        IMU_gait_characteristics[f]['Peak AP Acceleration Sacrum right'] = {}
        IMU_gait_characteristics[f]['Peak AP Deceleration Sacrum left'] = {}
        IMU_gait_characteristics[f]['Peak AP Deceleration Sacrum right'] = {}
        
        # Calculate peak/impulse values for IMU data
        if frame == 'sensor frame':
            IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_SF[f] = APaccelerationLumbar(IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_SF[f], sample_frequency = IMU[f]['Sample Frequency (Hz)'], debugplot = debugplot, plot_title = f)
        elif frame == 'earth frame':
            IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_EF[f] = APaccelerationLumbar(IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_EF[f], sample_frequency = IMU[f]['Sample Frequency (Hz)'], debugplot = debugplot, plot_title = f)
        elif frame == 'body frame':
            IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_BF[f] = APaccelerationLumbar(IMU_gait_events[f], IMU_gait_characteristics[f], IMU_AP_ACC_Lumbar_BF[f], sample_frequency = IMU[f]['Sample Frequency (Hz)'], debugplot = debugplot, plot_title = f)

        # Store IMU data in FinalData
        FinalData[f]['IMU']['AP Deceleration Sacrum left - Impulse'] = IMU_gait_characteristics[f]['AP Deceleration Sacrum left']           # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['IMU']['AP Deceleration Sacrum left - Peak'] = IMU_gait_characteristics[f]['Peak AP Deceleration Sacrum left']         # size(n,2) --> PeakIndex, PeakValue
        FinalData[f]['IMU']['AP Acceleration Sacrum left - Impulse'] = IMU_gait_characteristics[f]['AP Acceleration Sacrum left']           # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['IMU']['AP Acceleration Sacrum left - Peak'] = IMU_gait_characteristics[f]['Peak AP Acceleration Sacrum left']         # size(n,2) --> PeakIndex, PeakValue
        
        FinalData[f]['IMU']['AP Deceleration Sacrum right - Impulse'] = IMU_gait_characteristics[f]['AP Deceleration Sacrum right']         # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['IMU']['AP Deceleration Sacrum right - Peak'] = IMU_gait_characteristics[f]['Peak AP Deceleration Sacrum right']       # size(n,2) --> PeakIndex, PeakValue
        FinalData[f]['IMU']['AP Acceleration Sacrum right - Impulse'] = IMU_gait_characteristics[f]['AP Acceleration Sacrum right']         # size(n,3) --> StartIndex, StopIndex, ImpulseValue
        FinalData[f]['IMU']['AP Acceleration Sacrum right - Peak'] = IMU_gait_characteristics[f]['Peak AP Acceleration Sacrum right']       # size(n,2) --> PeakIndex, PeakValue
    except:
        print('Cannot extract IMU based AP acceleration parameters for trial ', f) 
        
             
""" Save OMCS and IMU gait events, gait characteristics, and AP acceleration to working directory """
with open('IMU_gait_events.pkl', 'wb') as f:
    pickle.dump(IMU_gait_events, f)
with open('IMU_gait_characteristics.pkl', 'wb') as f:
    pickle.dump(IMU_gait_characteristics, f)
with open('IMU_AP_ACC_Lumbar_BF.pkl', 'wb') as f:
    pickle.dump(IMU_AP_ACC_Lumbar_BF, f)
with open('OMCS_gait_events.pkl', 'wb') as f:
    pickle.dump(OMCS_gait_events, f)
with open('OMCS_gait_characteristics.pkl', 'wb') as f:
    pickle.dump(OMCS_gait_characteristics, f)
with open('OMCS_AP_ACC_Sacrum.pkl', 'wb') as f:
    pickle.dump(OMCS_AP_ACC_Sacrum, f)
with open('FinalData.pkl', 'wb') as f:
    pickle.dump(FinalData, f)
    
    
""" Add IMU and OMCS acceleration peak/impulse to dataframes df_left and df_right to determine correlations """
# Load df_right and df_left from the working directory
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
    OMCS_APDecelerationSacrumImpulse_left = FinalData[key]['OMCS']['AP Deceleration Sacrum left - Impulse']
    OMCS_APDecelerationSacrumPeak_left = FinalData[key]['OMCS']['AP Deceleration Sacrum left - Peak']
    OMCS_APAccelerationSacrumImpulse_left = FinalData[key]['OMCS']['AP Acceleration Sacrum left - Impulse']
    OMCS_APAccelerationSacrumPeak_left = FinalData[key]['OMCS']['AP Acceleration Sacrum left - Peak']
    
    """ Add OMCS_APDecelerationSacrumImpulse_left columns to final_df (columns 46, 47 and 48) """
    start_index_column = []
    stop_index_column = []
    impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrumImpulse_left rows where stop_index is between IC and TC
            for APDecelerationSacrum_impulse in OMCS_APDecelerationSacrumImpulse_left:
                start_index, stop_index, impulse_value = APDecelerationSacrum_impulse
                
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
    final_df.insert(46, 'OMCS_APDecelerationSacrumImpulse_StartIndex_left', start_index_column)  
    final_df.insert(47, 'OMCS_APDecelerationSacrumImpulse_StopIndex_left', stop_index_column)  
    final_df.insert(48, 'OMCS_APDecelerationSacrumImpulse_Value_left', impulse_value_column)  
    
    
    """ Add OMCS_APDecelerationSacrumPeak_left columns to final_df (columns 49 and 50) """
    APDecelerationSacrum_peak_index_column = []
    APDecelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrumPeak_left rows where PeakIndex is between IC and TC
            for peak in OMCS_APDecelerationSacrumPeak_left:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APDecelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationSacrum_peak_index_column.append(np.nan)
                APDecelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationSacrum_peak_index_column.append(np.nan)
            APDecelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(49, 'OMCS_APDecelerationSacrumPeak_PeakIndex_left', APDecelerationSacrum_peak_index_column)  
    final_df.insert(50, 'OMCS_APDecelerationSacrumPeak_Value_left', APDecelerationSacrum_peak_value_column)  

    
    """ Add OMCS_APAccelerationSacrumImpulse_left columns to final_df (columns 51, 52 and 53) """
    APAccelerationSacrum_start_index_column = []
    APAccelerationSacrum_stop_index_column = []
    APAccelerationSacrum_impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrumImpulse_left rows where start_index is between IC and TC
            for APAccelerationSacrum_impulse in OMCS_APAccelerationSacrumImpulse_left:
                start_index, stop_index, impulse_value = APAccelerationSacrum_impulse
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationSacrum_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationSacrum_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationSacrum_impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_start_index_column.append(np.nan)
                APAccelerationSacrum_stop_index_column.append(np.nan)
                APAccelerationSacrum_impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_start_index_column.append(np.nan)
            APAccelerationSacrum_stop_index_column.append(np.nan)
            APAccelerationSacrum_impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(51, 'OMCS_APAccelerationSacrumImpulse_StartIndex_left', APAccelerationSacrum_start_index_column)  
    final_df.insert(52, 'OMCS_APAccelerationSacrumImpulse_StopIndex_left', APAccelerationSacrum_stop_index_column)  
    final_df.insert(53, 'OMCS_APAccelerationSacrumImpulse_Value_left', APAccelerationSacrum_impulse_value_column)  
    
    
    """ Add OMCS_APAccelerationSacrumPeak_left columns to final_df (columns 54 and 55) """
    APAccelerationSacrum_peak_index_column = []
    APAccelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrumPeak_left rows where PeakIndex is between IC and TC
            for peak in OMCS_APAccelerationSacrumPeak_left:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APAccelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_peak_index_column.append(np.nan)
                APAccelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_peak_index_column.append(np.nan)
            APAccelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(54, 'OMCS_APAccelerationSacrumPeak_PeakIndex_left', APAccelerationSacrum_peak_index_column)  
    final_df.insert(55, 'OMCS_APAccelerationSacrumPeak_Value_left', APAccelerationSacrum_peak_value_column)  
                                

    """ IMU """  
    IMU_APDecelerationSacrumImpulse_left = FinalData[key]['IMU']['AP Deceleration Sacrum left - Impulse']
    IMU_APDecelerationSacrumPeak_left = FinalData[key]['IMU']['AP Deceleration Sacrum left - Peak']
    IMU_APAccelerationSacrumImpulse_left = FinalData[key]['IMU']['AP Acceleration Sacrum left - Impulse']
    IMU_APAccelerationSacrumPeak_left = FinalData[key]['IMU']['AP Acceleration Sacrum left - Peak']
    
    """ Add IMU_APDecelerationSacrumImpulse_left columns to final_df (columns 56, 57 and 58) """
    start_index_column = []
    stop_index_column = []
    impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationSacrumImpulse_left rows where stop_index is between IC and TC
            for APDecelerationSacrum_impulse in IMU_APDecelerationSacrumImpulse_left:
                start_index, stop_index, impulse_value = APDecelerationSacrum_impulse
                
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
    final_df.insert(56, 'IMU_APDecelerationSacrumImpulse_StartIndex_left', start_index_column)  
    final_df.insert(57, 'IMU_APDecelerationSacrumImpulse_StopIndex_left', stop_index_column)  
    final_df.insert(58, 'IMU_APDecelerationSacrumImpulse_Value_left', impulse_value_column)  
    
    
    """ Add IMU_APDecelerationSacrumPeak_left columns to final_df (columns 59 and 60) """
    APDecelerationSacrum_peak_index_column = []
    APDecelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationSacrumPeak_left rows where PeakIndex is between IC and TC
            for peak in IMU_APDecelerationSacrumPeak_left:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APDecelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationSacrum_peak_index_column.append(np.nan)
                APDecelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationSacrum_peak_index_column.append(np.nan)
            APDecelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(59, 'IMU_APDecelerationSacrumPeak_PeakIndex_left', APDecelerationSacrum_peak_index_column)  
    final_df.insert(60, 'IMU_APDecelerationSacrumPeak_Value_left', APDecelerationSacrum_peak_value_column)  

    
    """ Add IMU_APAccelerationSacrumImpulse_left columns to final_df (columns 61, 62 and 63) """
    APAccelerationSacrum_start_index_column = []
    APAccelerationSacrum_stop_index_column = []
    APAccelerationSacrum_impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationSacrumImpulse_left rows where start_index is between IC and TC
            for APAccelerationSacrum_impulse in IMU_APAccelerationSacrumImpulse_left:
                start_index, stop_index, impulse_value = APAccelerationSacrum_impulse
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationSacrum_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationSacrum_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationSacrum_impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_start_index_column.append(np.nan)
                APAccelerationSacrum_stop_index_column.append(np.nan)
                APAccelerationSacrum_impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_start_index_column.append(np.nan)
            APAccelerationSacrum_stop_index_column.append(np.nan)
            APAccelerationSacrum_impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(61, 'IMU_APAccelerationSacrumImpulse_StartIndex_left', APAccelerationSacrum_start_index_column)  
    final_df.insert(62, 'IMU_APAccelerationSacrumImpulse_StopIndex_left', APAccelerationSacrum_stop_index_column)  
    final_df.insert(63, 'IMU_APAccelerationSacrumImpulse_Value_left', APAccelerationSacrum_impulse_value_column)  
    
    
    """ Add IMU_APAccelerationSacrumPeak_left columns to final_df (columns 64 and 65) """
    APAccelerationSacrum_peak_index_column = []
    APAccelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationSacrumPeak_left rows where PeakIndex is between IC and TC
            for peak in IMU_APAccelerationSacrumPeak_left:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APAccelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_peak_index_column.append(np.nan)
                APAccelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_peak_index_column.append(np.nan)
            APAccelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(64, 'IMU_APAccelerationSacrumPeak_PeakIndex_left', APAccelerationSacrum_peak_index_column)  
    final_df.insert(65, 'IMU_APAccelerationSacrumPeak_Value_left', APAccelerationSacrum_peak_value_column)  


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
    OMCS_APDecelerationSacrumImpulse_right = FinalData[key]['OMCS']['AP Deceleration Sacrum right - Impulse']
    OMCS_APDecelerationSacrumPeak_right = FinalData[key]['OMCS']['AP Deceleration Sacrum right - Peak']
    OMCS_APAccelerationSacrumImpulse_right = FinalData[key]['OMCS']['AP Acceleration Sacrum right - Impulse']
    OMCS_APAccelerationSacrumPeak_right = FinalData[key]['OMCS']['AP Acceleration Sacrum right - Peak']
    
    """ Add OMCS_APDecelerationSacrumImpulse_right columns to final_df (columns 46, 47 and 48) """
    start_index_column = []
    stop_index_column = []
    impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrumImpulse_right rows where stop_index is between IC and TC
            for APDecelerationSacrum_impulse in OMCS_APDecelerationSacrumImpulse_right:
                start_index, stop_index, impulse_value = APDecelerationSacrum_impulse
                
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
    final_df.insert(46, 'OMCS_APDecelerationSacrumImpulse_StartIndex_right', start_index_column)  
    final_df.insert(47, 'OMCS_APDecelerationSacrumImpulse_StopIndex_right', stop_index_column)  
    final_df.insert(48, 'OMCS_APDecelerationSacrumImpulse_Value_right', impulse_value_column)  
    
    
    """ Add OMCS_APDecelerationSacrumPeak_right columns to final_df (columns 49 and 50) """
    APDecelerationSacrum_peak_index_column = []
    APDecelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APDecelerationSacrumPeak_right rows where PeakIndex is between IC and TC
            for peak in OMCS_APDecelerationSacrumPeak_right:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APDecelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationSacrum_peak_index_column.append(np.nan)
                APDecelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationSacrum_peak_index_column.append(np.nan)
            APDecelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(49, 'OMCS_APDecelerationSacrumPeak_PeakIndex_right', APDecelerationSacrum_peak_index_column)  
    final_df.insert(50, 'OMCS_APDecelerationSacrumPeak_Value_right', APDecelerationSacrum_peak_value_column)  

    
    """ Add OMCS_APAccelerationSacrumImpulse_right columns to final_df (columns 51, 52 and 53) """
    APAccelerationSacrum_start_index_column = []
    APAccelerationSacrum_stop_index_column = []
    APAccelerationSacrum_impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrumImpulse_right rows where start_index is between IC and TC
            for APAccelerationSacrum_impulse in OMCS_APAccelerationSacrumImpulse_right:
                start_index, stop_index, impulse_value = APAccelerationSacrum_impulse
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationSacrum_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationSacrum_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationSacrum_impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_start_index_column.append(np.nan)
                APAccelerationSacrum_stop_index_column.append(np.nan)
                APAccelerationSacrum_impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_start_index_column.append(np.nan)
            APAccelerationSacrum_stop_index_column.append(np.nan)
            APAccelerationSacrum_impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(51, 'OMCS_APAccelerationSacrumImpulse_StartIndex_right', APAccelerationSacrum_start_index_column)  
    final_df.insert(52, 'OMCS_APAccelerationSacrumImpulse_StopIndex_right', APAccelerationSacrum_stop_index_column)  
    final_df.insert(53, 'OMCS_APAccelerationSacrumImpulse_Value_right', APAccelerationSacrum_impulse_value_column)  
    
    
    """ Add OMCS_APAccelerationSacrumPeak_right columns to final_df (columns 54 and 55) """
    APAccelerationSacrum_peak_index_column = []
    APAccelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching OMCS_APAccelerationSacrumPeak_right rows where PeakIndex is between IC and TC
            for peak in OMCS_APAccelerationSacrumPeak_right:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APAccelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_peak_index_column.append(np.nan)
                APAccelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_peak_index_column.append(np.nan)
            APAccelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(54, 'OMCS_APAccelerationSacrumPeak_PeakIndex_right', APAccelerationSacrum_peak_index_column)  
    final_df.insert(55, 'OMCS_APAccelerationSacrumPeak_Value_right', APAccelerationSacrum_peak_value_column)  


    """ IMU """
    IMU_APDecelerationSacrumImpulse_right = FinalData[key]['IMU']['AP Deceleration Sacrum right - Impulse']
    IMU_APDecelerationSacrumPeak_right = FinalData[key]['IMU']['AP Deceleration Sacrum right - Peak']
    IMU_APAccelerationSacrumImpulse_right = FinalData[key]['IMU']['AP Acceleration Sacrum right - Impulse']
    IMU_APAccelerationSacrumPeak_right = FinalData[key]['IMU']['AP Acceleration Sacrum right - Peak']
    
    """ Add IMU_APDecelerationSacrumImpulse_right columns to final_df (columns 56, 57 and 58) """
    start_index_column = []
    stop_index_column = []
    impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationSacrumImpulse_right rows where stop_index is between IC and TC
            for APDecelerationSacrum_impulse in IMU_APDecelerationSacrumImpulse_right:
                start_index, stop_index, impulse_value = APDecelerationSacrum_impulse
                
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
    final_df.insert(56, 'IMU_APDecelerationSacrumImpulse_StartIndex_right', start_index_column)  
    final_df.insert(57, 'IMU_APDecelerationSacrumImpulse_StopIndex_right', stop_index_column)  
    final_df.insert(58, 'IMU_APDecelerationSacrumImpulse_Value_right', impulse_value_column)  
    
    
    """ Add IMU_APDecelerationSacrumPeak_right columns to final_df (columns 59 and 60) """
    APDecelerationSacrum_peak_index_column = []
    APDecelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APDecelerationSacrumPeak_right rows where PeakIndex is between IC and TC
            for peak in IMU_APDecelerationSacrumPeak_right:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APDecelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APDecelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APDecelerationSacrum_peak_index_column.append(np.nan)
                APDecelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APDecelerationSacrum_peak_index_column.append(np.nan)
            APDecelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame  
    final_df.insert(59, 'IMU_APDecelerationSacrumPeak_PeakIndex_right', APDecelerationSacrum_peak_index_column)  
    final_df.insert(60, 'IMU_APDecelerationSacrumPeak_Value_right', APDecelerationSacrum_peak_value_column)  

    
    """ Add IMU_APAccelerationSacrumImpulse_right columns to final_df (columns 61, 62 and 63) """
    APAccelerationSacrum_start_index_column = []
    APAccelerationSacrum_stop_index_column = []
    APAccelerationSacrum_impulse_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationSacrumImpulse_right rows where start_index is between IC and TC
            for APAccelerationSacrum_impulse in IMU_APAccelerationSacrumImpulse_right:
                start_index, stop_index, impulse_value = APAccelerationSacrum_impulse
                
                # Check if the start_index is between IC and TC
                if ic < start_index < tc:
                    APAccelerationSacrum_start_index_column.append(start_index)  # Add StartIndex to the list
                    APAccelerationSacrum_stop_index_column.append(stop_index)  # Add StopIndex to the list
                    APAccelerationSacrum_impulse_value_column.append(impulse_value)  # Add ImpulseValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_start_index_column.append(np.nan)
                APAccelerationSacrum_stop_index_column.append(np.nan)
                APAccelerationSacrum_impulse_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_start_index_column.append(np.nan)
            APAccelerationSacrum_stop_index_column.append(np.nan)
            APAccelerationSacrum_impulse_value_column.append(np.nan)
    
    # Add the columns to the DataFrame   
    final_df.insert(61, 'IMU_APAccelerationSacrumImpulse_StartIndex_right', APAccelerationSacrum_start_index_column)  
    final_df.insert(62, 'IMU_APAccelerationSacrumImpulse_StopIndex_right', APAccelerationSacrum_stop_index_column)  
    final_df.insert(63, 'IMU_APAccelerationSacrumImpulse_Value_right', APAccelerationSacrum_impulse_value_column)  
    
    
    """ Add IMU_APAccelerationSacrumPeak_right columns to final_df (columns 64 and 65) """
    APAccelerationSacrum_peak_index_column = []
    APAccelerationSacrum_peak_value_column = []
    
    # Loop through each row in final_df
    for row in final_df.iloc[:, 7:9].values:  # Check columns 7 (IC) and 8 (TC) of final_df
        ic, tc = row  # Extract IC and TC values from final_df
        
        if not np.isnan(ic) and not np.isnan(tc):
            # Find matching IMU_APAccelerationSacrumPeak_right rows where PeakIndex is between IC and TC
            for peak in IMU_APAccelerationSacrumPeak_right:
                peak_index, peak_value = peak
                
                # Check if the PeakIndex is between IC and TC
                if ic < peak_index < tc:
                    APAccelerationSacrum_peak_index_column.append(peak_index)  # Add PeakIndex to the list
                    APAccelerationSacrum_peak_value_column.append(peak_value)  # Add PeakValue to the list
                    break  # Stop after finding the first match
            else:
                # Append NaN if no match found
                APAccelerationSacrum_peak_index_column.append(np.nan)
                APAccelerationSacrum_peak_value_column.append(np.nan)
        else:
            # Append NaN if IC or TC is NaN
            APAccelerationSacrum_peak_index_column.append(np.nan)
            APAccelerationSacrum_peak_value_column.append(np.nan)
    
    # Add the columns to the DataFrame
    final_df.insert(64, 'IMU_APAccelerationSacrumPeak_PeakIndex_right', APAccelerationSacrum_peak_index_column)  
    final_df.insert(65, 'IMU_APAccelerationSacrumPeak_Value_right', APAccelerationSacrum_peak_value_column)  

                                
    matching_dfs[f] = final_df

# Store matching_dfs same format as df_right
for key in df_right_with_IMU_data.keys():
    filename = key[-1].split('/')[-1]  # Extract filename from the key
    
    if filename in matching_dfs:
        df_right_with_IMU_data[key] = matching_dfs[filename]  # Update with modified DataFrame


# Save df_right_with_IMU_data and df_left_with_IMU_data to the working directory
if frame == 'sensor frame' and gait_events_for_IMU == 'IMU':
    with open('df_right_with_IMU_data_SF.pkl', 'wb') as f:
        pickle.dump(df_right_with_IMU_data, f)
    print("Successfully saved df_right_with_IMU_data to the working directory.")
    with open('df_left_with_IMU_data_SF.pkl', 'wb') as f:
        pickle.dump(df_left_with_IMU_data, f)
    print("Successfully saved df_left_with_IMU_data to the working directory.")

elif frame == 'earth frame' and gait_events_for_IMU == 'IMU':
    with open('df_right_with_IMU_data_EF.pkl', 'wb') as f:
        pickle.dump(df_right_with_IMU_data, f)
    print("Successfully saved df_right_with_IMU_data to the working directory.")
    with open('df_left_with_IMU_data_EF.pkl', 'wb') as f:
        pickle.dump(df_left_with_IMU_data, f)
    print("Successfully saved df_left_with_IMU_data to the working directory.")

elif frame == 'body frame' and gait_events_for_IMU == 'IMU':
    with open('df_right_with_IMU_data_BF.pkl', 'wb') as f:
        pickle.dump(df_right_with_IMU_data, f)
    print("Successfully saved df_right_with_IMU_data to the working directory.")
    with open('df_left_with_IMU_data_BF.pkl', 'wb') as f:
        pickle.dump(df_left_with_IMU_data, f)
    print("Successfully saved df_left_with_IMU_data to the working directory.")
    
elif frame == 'sensor frame' and gait_events_for_IMU == 'OMCS':
    with open('df_right_with_IMU_data_SF_and_OMCSgaitevents.pkl', 'wb') as f:
        pickle.dump(df_right_with_IMU_data, f)
    print("Successfully saved df_right_with_IMU_data to the working directory.")
    with open('df_left_with_IMU_data_SF_and_OMCSgaitevents.pkl', 'wb') as f:
        pickle.dump(df_left_with_IMU_data, f)
    print("Successfully saved df_left_with_IMU_data to the working directory.")

elif frame == 'earth frame' and gait_events_for_IMU == 'OMCS':
    with open('df_right_with_IMU_data_EF_and_OMCSgaitevents.pkl', 'wb') as f:
        pickle.dump(df_right_with_IMU_data, f)
    print("Successfully saved df_right_with_IMU_data to the working directory.")
    with open('df_left_with_IMU_data_EF_and_OMCSgaitevents.pkl', 'wb') as f:
        pickle.dump(df_left_with_IMU_data, f)
    print("Successfully saved df_left_with_IMU_data to the working directory.")

elif frame == 'body frame' and gait_events_for_IMU == 'OMCS':
    with open('df_right_with_IMU_data_BF_and_OMCSgaitevents.pkl', 'wb') as f:
        pickle.dump(df_right_with_IMU_data, f)
    print("Successfully saved df_right_with_IMU_data to the working directory.")
    with open('df_left_with_IMU_data_BF_and_OMCSgaitevents.pkl', 'wb') as f:
        pickle.dump(df_left_with_IMU_data, f)
    print("Successfully saved df_left_with_IMU_data to the working directory.")


            
""" Create dataframe with all trials for Excel-sheet needed for analyses in RStudio """
# Load df_right_with_IMU_data and df_left_with_IMU_data from the working directory
if frame == 'sensor frame':
    with open('df_right_with_IMU_data_SF.pkl', 'rb') as f:
        df_right = pickle.load(f)
    with open('df_left_with_IMU_data_SF.pkl', 'rb') as f:
        df_left = pickle.load(f)
    print("Successfully loaded df_left and df_right from the working directory.")

elif frame == 'earth frame':
    with open('df_right_with_IMU_data_EF.pkl', 'rb') as f:
        df_right = pickle.load(f)
    with open('df_left_with_IMU_data_EF.pkl', 'rb') as f:
        df_left = pickle.load(f)
    print("Successfully loaded df_left and df_right from the working directory.")

elif frame == 'body frame':
    with open('df_right_with_IMU_data_BF.pkl', 'rb') as f:
        df_right = pickle.load(f)
    with open('df_left_with_IMU_data_BF.pkl', 'rb') as f:
        df_left = pickle.load(f)
    print("Successfully loaded df_left and df_right from the working directory.")
    
# Save all data from all subjects in one dataframe
R_dataset_df_right = pd.concat(df_right.values(), ignore_index=True)
R_dataset_df_left = pd.concat(df_left.values(), ignore_index=True)
            


""" Debug plots """
          
" Plot AP-GRF, AP acceleration OMCS, and AP acceleration IMU "
if debugplot == True:
    for trial in selected_trialnames:
        try:       
            OMCS_AP_GRF_data_left = OMCS_AP_GRF_left[trial]           
            OMCS_AP_GRF_data_right = OMCS_AP_GRF_right[trial]           
            OMCS_ACC_data = OMCS_AP_ACC_Sacrum[trial]           
            IMU_ACC_data_SF = IMU_AP_ACC_Lumbar_SF[trial]
            IMU_ACC_data_EF = IMU_AP_ACC_Lumbar_EF[trial]
            IMU_ACC_data_BF = IMU_AP_ACC_Lumbar_BF[trial]
              
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            # ---- Plot 1: GRF left + right ----
            axes[0].plot(-OMCS_AP_GRF_data_left/OMCS[trial]['body_mass'], label='AP-GRF left')
            axes[0].plot(-OMCS_AP_GRF_data_right/OMCS[trial]['body_mass'], label='AP-GRF right')
            axes[0].set_ylabel('AP-GRF (N/kg)')
            axes[0].legend()
            axes[0].grid()

            # ---- Plot 2: AP acceleration OMCS ----
            axes[1].plot(-OMCS_ACC_data/1000, label='OMCS')
            axes[1].set_ylabel('AP Acceleration (m/s²)')
            axes[1].legend()
            axes[1].grid()

            # ---- Plot 3: AP acceleration IMU ----
            axes[2].plot(-IMU_ACC_data_BF, label='IMU_BF')
            axes[2].set_xlabel('Time (samples)')
            axes[2].set_ylabel('AP Acceleration (m/s²)')
            axes[2].legend()
            axes[2].grid()          

            # ---- Add vertical lines for gait events ----
            # Use OMCS gait events for plots 0 and 1
            for ic in OMCS_gait_events[trial]['Index numbers initial contact left']:
                axes[0].axvline(x=ic, color='red', linestyle='--', linewidth=1, label='IC Left' if ic == OMCS_gait_events[trial]['Index numbers initial contact left'][0] else "")
                axes[1].axvline(x=ic, color='red', linestyle='--', linewidth=1)
            for ic in OMCS_gait_events[trial]['Index numbers initial contact right']:
                axes[0].axvline(x=ic, color='blue', linestyle='--', linewidth=1, label='IC Right' if ic == OMCS_gait_events[trial]['Index numbers initial contact right'][0] else "")
                axes[1].axvline(x=ic, color='blue', linestyle='--', linewidth=1)
            for tc in OMCS_gait_events[trial]['Index numbers terminal contact left']:
                axes[0].axvline(x=tc, color='red', linestyle='-.', linewidth=1, label='TC Left' if tc == OMCS_gait_events[trial]['Index numbers terminal contact left'][0] else "")
                axes[1].axvline(x=tc, color='red', linestyle='-.', linewidth=1)
            for tc in OMCS_gait_events[trial]['Index numbers terminal contact right']:
                axes[0].axvline(x=tc, color='blue', linestyle='-.', linewidth=1, label='TC Right' if tc == OMCS_gait_events[trial]['Index numbers terminal contact right'][0] else "")
                axes[1].axvline(x=tc, color='blue', linestyle='-.', linewidth=1)
            
            # Use IMU gait events for plot 2
            for ic in IMU_gait_events[trial]['Index numbers initial contact left']:
                axes[2].axvline(x=ic, color='red', linestyle='--', linewidth=1, label='IC Left' if ic == IMU_gait_events[trial]['Index numbers initial contact left'][0] else "")
            for ic in IMU_gait_events[trial]['Index numbers initial contact right']:
                axes[2].axvline(x=ic, color='blue', linestyle='--', linewidth=1, label='IC Right' if ic == IMU_gait_events[trial]['Index numbers initial contact right'][0] else "")
            for tc in IMU_gait_events[trial]['Index numbers terminal contact left']:
                axes[2].axvline(x=tc, color='red', linestyle='-.', linewidth=1, label='TC Left' if tc == IMU_gait_events[trial]['Index numbers terminal contact left'][0] else "")
            for tc in IMU_gait_events[trial]['Index numbers terminal contact right']:
                axes[2].axvline(x=tc, color='blue', linestyle='-.', linewidth=1, label='TC Right' if tc == IMU_gait_events[trial]['Index numbers terminal contact right'][0] else "")

            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())                 

            plt.xlim(5000, 5500)
            plt.suptitle(f"AP-GRF and AP Acceleration Data for Trial: {trial}")
            plt.show()
            
        except:
            print(f"Cannot plot AP-GRF and AP Acceleration data for trial: {trial}") 



" Plot AP acceleration IMU for all trials of one participant "
if debugplot == True:    
 
    healthy_participants = ["pp01","pp03","pp04","pp05","pp06","pp07","pp08","pp09","pp10","pp11","pp12","pp13","pp14","pp15","pp16","pp18","pp19","pp20","pp21","pp22"]
    
    for pid in healthy_participants:
        
        trialnames_healthy_participants = [t for t in trialnames if t.startswith("900_V_pp")]
    
        participants = {}
        for trial in trialnames_healthy_participants:
            parts = trial.split("_")
            trial_pid = parts[2]
        
            if trial_pid not in participants:
                participants[trial_pid] = {}
        
            if "SP" in trial:                               # Self-paced treadmill
                participants[trial_pid]["Self-paced treadmill"] = trial
            elif "FS_SS" in trial or "SS" in trial:         # Fixed-speed stepping stones
                participants[trial_pid]["Fixed-speed stepping stones treadmill"] = trial
            elif "2MWT" in trial or "SW" in trial:          # Overground walking
                participants[trial_pid]["Overground walking"] = trial
                
        trials_to_plot = participants[pid]
        
        x_lim = 3500
        y_lim = 4500
    
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
        
        for row, (title, trial) in enumerate(trials_to_plot.items()):
            # Left = GRF, Right = IMU
            ax_grf = axes[row, 0]
            ax_imu = axes[row, 1]
        
            try:
                # === IMU acceleration (always plotted) ===
                IMU_ACC_data_BF = IMU_AP_ACC_Lumbar_BF[trial]
                ax_imu.plot(-IMU_ACC_data_BF, label='IMU_BF')
                ax_imu.set_xlim(x_lim, y_lim)
                ax_imu.set_ylabel('AP Acceleration (m/s²)')
                ax_imu.set_title(f"{title} – IMU ({trial})")
                ax_imu.grid()
        
                # ---- Add IMU gait events ----
                for ic in IMU_gait_events[trial]['Index numbers initial contact left']:
                    ax_imu.axvline(x=ic, color='red', linestyle='--', linewidth=1,
                                   label='IC Left' if ic == IMU_gait_events[trial]['Index numbers initial contact left'][0] else "")
                for ic in IMU_gait_events[trial]['Index numbers initial contact right']:
                    ax_imu.axvline(x=ic, color='blue', linestyle='--', linewidth=1,
                                   label='IC Right' if ic == IMU_gait_events[trial]['Index numbers initial contact right'][0] else "")
                for tc in IMU_gait_events[trial]['Index numbers terminal contact left']:
                    ax_imu.axvline(x=tc, color='red', linestyle='-.', linewidth=1,
                                   label='TC Left' if tc == IMU_gait_events[trial]['Index numbers terminal contact left'][0] else "")
                for tc in IMU_gait_events[trial]['Index numbers terminal contact right']:
                    ax_imu.axvline(x=tc, color='blue', linestyle='-.', linewidth=1,
                                   label='TC Right' if tc == IMU_gait_events[trial]['Index numbers terminal contact right'][0] else "")
        
                # Deduplicate legend
                handles, labels = ax_imu.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_imu.legend(by_label.values(), by_label.keys(), fontsize=8)
        
                # === OMCS propulsion forces (only treadmill trials) ===
                if "SP01" in trial or "FS_SS01" in trial:
                    OMCS_AP_GRF_data_left = OMCS_AP_GRF_left[trial]
                    OMCS_AP_GRF_data_right = OMCS_AP_GRF_right[trial]
                    body_mass = OMCS[trial]['body_mass']
        
                    ax_grf.plot(-OMCS_AP_GRF_data_left/body_mass, label='AP-GRF Left')
                    ax_grf.plot(-OMCS_AP_GRF_data_right/body_mass, label='AP-GRF Right')
                    ax_grf.set_xlim(x_lim, y_lim)
                    ax_grf.set_ylabel('AP-GRF (N/kg)')
                    ax_grf.set_title(f"{title} – Propulsion Force ({trial})")
                    ax_grf.grid()
        
                    # ---- Add OMCS gait events ----
                    for ic in OMCS_gait_events[trial]['Index numbers initial contact left']:
                        ax_grf.axvline(x=ic, color='red', linestyle='--', linewidth=1,
                                       label='IC Left' if ic == OMCS_gait_events[trial]['Index numbers initial contact left'][0] else "")
                    for ic in OMCS_gait_events[trial]['Index numbers initial contact right']:
                        ax_grf.axvline(x=ic, color='blue', linestyle='--', linewidth=1,
                                       label='IC Right' if ic == OMCS_gait_events[trial]['Index numbers initial contact right'][0] else "")
                    for tc in OMCS_gait_events[trial]['Index numbers terminal contact left']:
                        ax_grf.axvline(x=tc, color='red', linestyle='-.', linewidth=1,
                                       label='TC Left' if tc == OMCS_gait_events[trial]['Index numbers terminal contact left'][0] else "")
                    for tc in OMCS_gait_events[trial]['Index numbers terminal contact right']:
                        ax_grf.axvline(x=tc, color='blue', linestyle='-.', linewidth=1,
                                       label='TC Right' if tc == OMCS_gait_events[trial]['Index numbers terminal contact right'][0] else "")
        
                    # Deduplicate legend
                    handles, labels = ax_grf.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax_grf.legend(by_label.values(), by_label.keys(), fontsize=8)
        
                else:
                    # Overground trial: leave GRF panel empty
                    ax_grf.axis("off")
        
            except:
                print(f"Cannot plot data for trial: {trial}")
        
        # X-labels
        axes[-1, 0].set_xlabel("Time (samples)")
        axes[-1, 1].set_xlabel("Time (samples)")
        
        plt.suptitle("OMCS Propulsion Forces and IMU AP Acceleration with Gait Events", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



" Plot AP-GRF and AP acceleration IMU for every 1000 samples of the trial "
if debugplot == True:    

    for trial in selected_trialnames:
        
        x_limits = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]
        
        for x_start in x_limits:
            x_end = x_start + 1000
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            # -------------------------
            # Plot 1: AP-GRF + OMCS gait events
            # -------------------------
            AP_GRF_left = OMCS_AP_GRF_left[trial] / OMCS[trial]['body_mass']
            axes[0].plot(-AP_GRF_left, label='AP-GRF Left', color='black')
            axes[0].set_ylabel("AP-GRF Left (N/kg)")
            axes[0].set_xlim(x_start, x_end)
            axes[0].set_title(f"Trial {trial} - AP-GRF Left with OMCS gait events")
            axes[0].grid()
            
            # OMCS gait events
            for ic in OMCS_gait_events[trial]['Index numbers initial contact left']:
                axes[0].axvline(x=ic, color='red', linestyle='--', linewidth=1)
            for tc in OMCS_gait_events[trial]['Index numbers terminal contact left']:
                axes[0].axvline(x=tc, color='red', linestyle='-.', linewidth=1)
            
            # Shaded areas
            for phase, color in zip(['Braking left', 'Propulsion left'], ['red', 'green']):
                periods = OMCS_gait_characteristics[trial][phase]
                for period in periods:
                    start, stop = period[0], period[1]
                    axes[0].axvspan(start, stop, color=color, alpha=0.3)
            
            axes[0].legend()
            
            # -------------------------
            # Plot 2: AP-GRF Right + OMCS gait events
            # -------------------------
            AP_GRF_right = OMCS_AP_GRF_right[trial] / OMCS[trial]['body_mass']
            axes[1].plot(-AP_GRF_right, label='AP-GRF Right', color='black')
            axes[1].set_ylabel("AP-GRF Right (N/kg)")
            axes[1].set_title(f"Trial {trial} - AP-GRF Right with OMCS gait events")
            axes[1].grid()
            
            # OMCS gait events
            for ic in OMCS_gait_events[trial]['Index numbers initial contact right']:
                axes[1].axvline(x=ic, color='blue', linestyle='--', linewidth=1)
            for tc in OMCS_gait_events[trial]['Index numbers terminal contact right']:
                axes[1].axvline(x=tc, color='blue', linestyle='-.', linewidth=1)
            
            # Shaded areas
            for phase, color in zip(['Braking right', 'Propulsion right'], ['red', 'green']):
                periods = OMCS_gait_characteristics[trial][phase]
                for period in periods:
                    start, stop = period[0], period[1]
                    axes[1].axvspan(start, stop, color=color, alpha=0.3)
            
            axes[1].legend()
            
            # -------------------------
            # Plot 3: IMU AP Acceleration + IMU gait events
            # -------------------------
            IMU_ACC_BF = IMU_AP_ACC_Lumbar_BF[trial]
            axes[2].plot(-IMU_ACC_BF, label='IMU_BF', color='black')
            axes[2].set_ylabel("AP Acceleration (m/s²)")
            axes[2].set_title(f"Trial {trial} - IMU AP Acceleration with IMU gait events")
            axes[2].grid()
            
            # IMU gait events
            for ic in IMU_gait_events[trial]['Index numbers initial contact left']:
                axes[2].axvline(x=ic, color='red', linestyle='--', linewidth=1)
            for ic in IMU_gait_events[trial]['Index numbers initial contact right']:
                axes[2].axvline(x=ic, color='blue', linestyle='--', linewidth=1)
            for tc in IMU_gait_events[trial]['Index numbers terminal contact left']:
                axes[2].axvline(x=tc, color='red', linestyle='-.', linewidth=1)
            for tc in IMU_gait_events[trial]['Index numbers terminal contact right']:
                axes[2].axvline(x=tc, color='blue', linestyle='-.', linewidth=1)
            
            # Shaded areas (IMU braking/propulsion)
            for phase, color in zip(['AP Deceleration Sacrum left', 'AP Deceleration Sacrum right',
                                     'AP Acceleration Sacrum left', 'AP Acceleration Sacrum right'],
                                    ['red', 'red', 'green', 'green']):
                periods = IMU_gait_characteristics[trial][phase]
                for period in periods:
                    start, stop = period[0], period[1]
                    axes[2].axvspan(start, stop, color=color, alpha=0.3)
            
            axes[2].legend()
            axes[2].set_xlabel("Time (samples)")
            
            plt.tight_layout()
            plt.show()

