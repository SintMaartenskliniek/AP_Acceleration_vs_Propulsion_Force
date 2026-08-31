"""
Contains helper functions that are used in main_LumbarAcceleration.py.

Version - Author:
    2026: Lars van Rengs - l.vanrengs@maartenskliniek.nl
"""

# Import dependencies
import numpy as np
import os
import samplerate
from scipy import signal
import matplotlib.pyplot as plt

# Import dependencies to analyze gait data
import IMU_GaitAnalysis.gaittool.feet_processor.processor as feet
from IMU_GaitAnalysis.gaittool.helpers.preprocessor import data_filelist, data_preprocessor

from OMCS_GaitAnalysis.readmarkerdata import readmarkerdata
from OMCS_GaitAnalysis.gaiteventdetection import gaiteventdetection
from OMCS_GaitAnalysis.gaitcharacteristics import spatiotemporals, propulsion


def dataimport(datafolder, datafolder2, trialtype):
    """
    Find corresponding files for Vicon and Xsens data
    Import data of Vicon and Xsens files
    """
    showfigure = 'hide'
        
    # Prepare datastructure
    vicon = dict()
    xsens = dict()
    errors = dict()
       
    # Set subfolder for xsens data
    subfolderxsens = 'Xsens/exported'
    
    # Set subfolder for vicon data
    subfolderviconGRAIL = 'Vicon/GRAIL'     # data IMU_GaitAnalysis
    subfoldervicon = 'Vicon'                # data MovingReality
    
    # Define xsens trialnumber with corresponding vicon measurement
    corresponding_files = dict()
    # All files
    files = dict()
    
    # Trials Healthy_controls (IMU_GaitAnalysis)
    if trialtype['Healthy_controls'] == True:
        subfolder = '/Healthy_controls'
        mainpath = datafolder + subfolder
        dirnames = os.listdir(mainpath)
        dirnames = [item for item in dirnames if item.startswith('900_V')]
        ppfolders = []
        ppfoldersvicon = []
        ppfoldersxsens = []
        for i in range(0, len(dirnames)):
            ppfolders.append(mainpath + '/' + dirnames[i])
        for i in range(0, len(ppfolders)):
            # date = os.listdir(ppfolders[i])
            ppfoldersvicon.append(ppfolders[i] + '/' + subfolderviconGRAIL) # + '/' + date[0]
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) # + '/' + date[0]
            
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            with os.scandir(ppfoldersvicon[i]) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_file():
                        files[entry.name] = (ppfoldersvicon[i] + '/' + entry.name)
                        
                        # Define xsens exports
                        if entry.name == '900_V_pp01_SP01.c3d':
                            xsensnum[entry.name] = '005'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp03_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp04_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp05_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp06_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp07_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp08_SP02.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp09_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp10_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp11_SP01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp12_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                                        
                        elif entry.name == '900_V_pp13_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp14_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp15_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp16_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp18_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp19_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp20_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp21_SP01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp22_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
        
        corresponding_files['Healthy_controls'] = dict()
        corresponding_files['Healthy_controls']['xsensnum'] = xsensnum
        corresponding_files['Healthy_controls']['xsensfilepaths'] = xsensfilepaths
        
    # Trials CVA (IMU_GaitAnalysis)
    if trialtype['CVA'] == True:
        subfolder = '/CVA'
        mainpath = datafolder + subfolder
        dirnames = os.listdir(mainpath)
        dirnames = [item for item in dirnames if item.startswith('900_CVA')]
        ppfolders = []
        ppfoldersvicon = []
        ppfoldersxsens = []
        for i in range(0, len(dirnames)):
            ppfolders.append(mainpath + '/' + dirnames[i])
        for i in range(0, len(ppfolders)):
            # date = os.listdir(ppfolders[i])
            ppfoldersvicon.append(ppfolders[i] + '/' + subfolderviconGRAIL) # + '/' + date[0]
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) # + '/' + date[0]
            
        # files=dict()
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            with os.scandir(ppfoldersvicon[i]) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_file():
                        files[entry.name] = (ppfoldersvicon[i]+'/'+entry.name)
                        
                        # Define xsens exports
                        if entry.name == '900_CVA_01_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_pp02_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_02' in item][0]+xsensnum[entry.name]
                                           
                        elif entry.name == '900_CVA_04_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_CVA_05_SP01.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_06_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_06' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_07_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_08_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_08' in item][0]+xsensnum[entry.name]
                                                
                        elif entry.name == '900_CVA_10_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_10' in item][0]+xsensnum[entry.name]

        corresponding_files['CVA'] = dict()
        corresponding_files['CVA']['xsensnum'] = xsensnum
        corresponding_files['CVA']['xsensfilepaths'] = xsensfilepaths
    
    # Trials CVA_feedback
    if trialtype['CVA_feedback'] == True:
        mainpath = datafolder2
        dirnames = os.listdir(mainpath)
        dirnames = [item for item in dirnames if item.startswith('1019_pp')]
        ppfolders = []
        ppfoldersvicon = []
        ppfoldersxsens = []
        for i in range(0, len(dirnames)):
            ppfolders.append(mainpath + '/' + dirnames[i])
        for i in range(0, len(ppfolders)):
            # date = os.listdir(ppfolders[i])
            ppfoldersvicon.append(ppfolders[i] + '/' + subfoldervicon) # + '/' + date[0]
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) # + '/' + date[0]
            
        # files=dict()
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            with os.scandir(ppfoldersvicon[i]) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_file():
                        files[entry.name] = (ppfoldersvicon[i]+'/'+entry.name)
                        
                        # Define xsens exports
                        if entry.name == '1019_MR001_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR002_Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp02' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR003_1Reg02.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp03' in item][0]+xsensnum[entry.name]

                        elif entry.name == '1019_MR004_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '1019_MR005_1Reg01.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR006_1Reg.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp06' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR007_1Reg02.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR008_1Reg02.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR009_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR010_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp10' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '1019_MR011_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp11' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '1019_MR012_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp12' in item][0]+xsensnum[entry.name]

        corresponding_files['CVA_feedback'] = dict()
        corresponding_files['CVA_feedback']['xsensnum'] = xsensnum
        corresponding_files['CVA_feedback']['xsensfilepaths'] = xsensfilepaths
   
    # 3 sets
    if trialtype['Healthy_controls'] == True and trialtype['CVA'] == True and trialtype['CVA_feedback'] == True:
        xsensfilepaths = {**corresponding_files['Healthy_controls']['xsensfilepaths'], **corresponding_files['CVA']['xsensfilepaths'], **corresponding_files['CVA_feedback']['xsensfilepaths']}
    # 2 sets
    elif trialtype['Healthy_controls'] == True and trialtype['CVA'] == True and trialtype['CVA_feedback'] == False:
        xsensfilepaths = {**corresponding_files['Healthy_controls']['xsensfilepaths'], **corresponding_files['CVA']['xsensfilepaths']}
    elif trialtype['Healthy_controls'] == True and trialtype['CVA'] == False and trialtype['CVA_feedback'] == True:
        xsensfilepaths = {**corresponding_files['Healthy_controls']['xsensfilepaths'], **corresponding_files['CVA_feedback']['xsensfilepaths']}
    elif trialtype['Healthy_controls'] == False and trialtype['CVA'] == True and trialtype['CVA_feedback'] == True:
        xsensfilepaths = {**corresponding_files['CVA']['xsensfilepaths'], **corresponding_files['CVA_feedback']['xsensfilepaths']}
    # 1 set
    elif trialtype['Healthy_controls'] == True and trialtype['CVA'] == False and trialtype['CVA_feedback'] == False:
        xsensfilepaths = corresponding_files['Healthy_controls']['xsensfilepaths']
    elif trialtype['Healthy_controls'] == False and trialtype['CVA'] == True and trialtype['CVA_feedback'] == False:
        xsensfilepaths = corresponding_files['CVA']['xsensfilepaths']
    elif trialtype['Healthy_controls'] == False and trialtype['CVA'] == False and trialtype['CVA_feedback'] == True:
        xsensfilepaths = corresponding_files['CVA_feedback']['xsensfilepaths']
    
    # Sort files on task
    filesGRAIL = dict()
    filesIMU = dict()   # files from IMU_GaitAnalysis
    filesMR = dict()    # files from MovingReality
    
    removekeys=[]
    for key in files:
        if ('V_pp02' or 'V_pp17') in key: # exclusion of these test persons
            removekeys.append(key)
    for key in removekeys:
        files.pop(key)

    for key in xsensfilepaths:
        # GRAIL trials
        if '_SP0' in key:
            if key == '900_V_pp01_SP03.c3d': # Fixed speed trial, accidentally wrongly named
                pass
            else:
                filesGRAIL[key] = files[key]
                filesIMU[key] = files[key]
        if '1019_MR' in key:
            filesGRAIL[key] = files[key]
            filesMR[key] = files[key]
    
    # Set trialnames to be analyzed
    trialnames = list()
    if trialtype['Healthy_controls'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '_V_' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['CVA'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '_CVA_' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['CVA_feedback'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '1019_MR' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    
    trialnames = list(set(trialnames))
    
    # Read markerdata vicon        
    for trial in range(0,len(trialnames)):
        try:
            print('Start vicon import of trial: ', trialnames[trial], ' (',trial,'/',len(trialnames),')')
            datavicon, fs_markerdata, analogdata, fs_analogdata = readmarkerdata(files[trialnames[trial]], analogdata=True ) #ParameterGroup, 
        
            # Check the markernames
            dataviconfilt = {}
            for key in datavicon:
                if 'LASI' in key:
                    dataviconfilt['LASI'] = datavicon[key]
                elif 'RASI' in key:
                    dataviconfilt['RASI'] = datavicon[key]
                elif 'LPSI' in key:
                    dataviconfilt['LPSI'] = datavicon[key]
                elif 'RPSI' in key:
                    dataviconfilt['RPSI'] = datavicon[key]
                elif 'LTHI' in key:
                    dataviconfilt['LTHI'] = datavicon[key]
                elif 'LKNE' in key:
                    dataviconfilt['LKNE'] = datavicon[key]
                elif 'LTIB' in key:
                    dataviconfilt['LTIB'] = datavicon[key]
                elif 'LANK' in key:
                    dataviconfilt['LANK'] = datavicon[key]
                elif 'LHEE' in key:
                    dataviconfilt['LHEE'] = datavicon[key]
                elif 'LTOE' in key:
                    dataviconfilt['LTOE'] = datavicon[key]
                elif 'RTHI' in key:
                    dataviconfilt['RTHI'] = datavicon[key]
                elif 'RKNE' in key:
                    dataviconfilt['RKNE'] = datavicon[key]
                elif 'RTIB' in key:
                    dataviconfilt['RTIB'] = datavicon[key]
                elif 'RANK' in key:
                    dataviconfilt['RANK'] = datavicon[key]
                elif 'RHEE' in key:
                    dataviconfilt['RHEE'] = datavicon[key]
                elif 'RTOE' in key:
                    dataviconfilt['RTOE'] = datavicon[key]
                                
            # Interpolate missing values
            if trialnames[trial] == '900_V_pp08_SP02.c3d': # Gap fill (3 x 1 sample)
                for key in dataviconfilt:
                    missingvalues = np.unique(np.where(dataviconfilt[key] == 0)[0])
                    nonmissingvalues = (np.where(dataviconfilt[key] != 0)[0])
                    dataviconfilt[key][missingvalues,0] = np.interp(missingvalues, nonmissingvalues, dataviconfilt[key][nonmissingvalues,0])
                    dataviconfilt[key][missingvalues,1] = np.interp(missingvalues, nonmissingvalues, dataviconfilt[key][nonmissingvalues,1])
                    dataviconfilt[key][missingvalues,2] = np.interp(missingvalues, nonmissingvalues, dataviconfilt[key][nonmissingvalues,2])

            # Only analyze last 120 seconds of MovingReality trials
            if '1019_MR' in trialnames[trial]:
                for key in dataviconfilt:
                    dataviconfilt[key] = dataviconfilt[key][int(-120*fs_markerdata):,:]
                for key in analogdata:
                    try:
                        analogdata[key] = analogdata[key][int(-120*fs_analogdata):,:]
                    except IndexError:
                        analogdata[key] = analogdata[key][int(-120*fs_analogdata):]
                    
            dataviconfilt['Analog data'] = analogdata
            dataviconfilt['Sample frequency marker data'] = fs_markerdata
            dataviconfilt['Sample frequency analog data'] = fs_analogdata
            
            # Add subject information to dictionary
            # Healthy controls
            if trialnames[trial] == '900_V_pp01_SP01.c3d':
                gender = 'F'
                body_mass = 72.0
                height = 1680
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp03_SP01.c3d':
                gender = 'F'
                body_mass = 74.8
                height = 1640
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp04_SP01.c3d':
                gender = 'M'
                body_mass = 76.8
                height = 1660
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp05_SP01.c3d':
                gender = 'F'
                body_mass = 67.8
                height = 1650
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp06_SP01.c3d':
                gender = 'M'
                body_mass = 77.2
                height = 1830
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp07_SP01.c3d':
                gender = 'F'
                body_mass = 62.4
                height = 1730
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp08_SP02.c3d':
                gender = 'F'
                body_mass = 63.6
                height = 1680
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp09_SP01.c3d':
                gender = 'M'
                body_mass = 69.0
                height = 1790
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp10_SP01.c3d':
                gender = 'M'
                body_mass = 93.0
                height = 1860
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp11_SP01.c3d':
                gender = 'M'
                body_mass = 77.6
                height = 1810
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp12_SP01.c3d':
                gender = 'F'
                body_mass = 78.2
                height = 1800
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp13_SP01.c3d':
                gender = 'M'
                body_mass = 88.6
                height = 1800
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp14_SP01.c3d':
                gender = 'F'
                body_mass = 68.4
                height = 1700
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp15_SP01.c3d':
                gender = 'F'
                body_mass = 66.2
                height = 1620
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp16_SP01.c3d':
                gender = 'F'
                body_mass = 70.4
                height = 1660
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp18_SP01.c3d':
                gender = 'M'
                body_mass = 77.0
                height = 1820
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp19_SP01.c3d':
                gender = 'F'
                body_mass = 70.0
                height = 1740
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp20_SP01.c3d':
                gender = 'M'
                body_mass = 76.8
                height = 1800
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp21_SP01.c3d':
                gender = 'M'
                body_mass = 89.2
                height = 1790
                affected_leg = 'none'
            elif trialnames[trial] == '900_V_pp22_SP01.c3d':
                gender = 'M'
                body_mass = 73.4
                height = 1760
                affected_leg = 'none'        
            # CVA
            elif trialnames[trial] == '900_CVA_01_SP01.c3d':
                gender = 'F'
                body_mass = 70.0
                height = 1620
                affected_leg = 'right'
            elif trialnames[trial] == '900_CVA_pp02_SP01.c3d':
                gender = 'M'
                body_mass = 82.0
                height = 1830
                affected_leg = 'left'
            elif trialnames[trial] == '900_CVA_04_SP01.c3d':
                gender = 'M'
                body_mass = 93.0
                height = 1810
                affected_leg = 'left'
            elif trialnames[trial] == '900_CVA_05_SP01.c3d':
                gender = 'M'
                body_mass = 91.0
                height = 1710
                affected_leg = 'left'
            elif trialnames[trial] == '900_CVA_06_SP01.c3d':
                gender = 'F'
                body_mass = 71.0
                height = 1760
                affected_leg = 'right'
            elif trialnames[trial] == '900_CVA_07_SP01.c3d':
                gender = 'M'
                body_mass = 95.4
                height = 1840
                affected_leg = 'right'
            elif trialnames[trial] == '900_CVA_08_SP01.c3d':
                gender = 'M'
                body_mass = 85.0
                height = 1840
                affected_leg = 'right'
            elif trialnames[trial] == '900_CVA_10_SP01.c3d':
                gender = 'F'
                body_mass = 77.0
                height = 1650
                affected_leg = 'right'
            # CVA_feedback
            elif trialnames[trial] == '1019_MR001_1Reg.c3d':
                gender = 'M'
                body_mass = 122.0
                height = 1775
                affected_leg = 'left'
            elif trialnames[trial] == '1019_MR002_Reg.c3d':
                gender = 'F'
                body_mass = 68.0
                height = 1710
                affected_leg = 'right'
            elif trialnames[trial] == '1019_MR003_1Reg02.c3d':
                gender = 'M'
                body_mass = 75.0
                height = 1720
                affected_leg = 'right'
            elif trialnames[trial] == '1019_MR004_1Reg.c3d':
                gender = 'F'
                body_mass = 70.0
                height = 1630
                affected_leg = 'left'
            elif trialnames[trial] == '1019_MR005_1Reg01.c3d':
                gender = 'M'
                body_mass = 80.0
                height = 1830
                affected_leg = 'left'
            elif trialnames[trial] == '1019_MR006_1Reg.c3d':
                gender = 'M'
                body_mass = 85.0
                height = 1940
                affected_leg = 'left'
            elif trialnames[trial] == '1019_MR007_1Reg02.c3d':
                gender = 'M'
                body_mass = 90.0
                height = 1830
                affected_leg = 'right'
            elif trialnames[trial] == '1019_MR008_1Reg02.c3d':
                gender = 'F'
                body_mass = 91.0
                height = 1720
                affected_leg = 'right'
            elif trialnames[trial] == '1019_MR009_1Reg.c3d':
                gender = 'M'
                body_mass = 74.0
                height = 1710
                affected_leg = 'right'
            elif trialnames[trial] == '1019_MR010_1Reg.c3d':
                gender = 'M'
                body_mass = 82.0
                height = 1880
                affected_leg = 'right'
            elif trialnames[trial] == '1019_MR011_1Reg.c3d':
                gender = 'F'
                body_mass = 104.0
                height = 1700
                affected_leg = 'left'
            elif trialnames[trial] == '1019_MR012_1Reg.c3d':
                gender = 'F'
                body_mass = 79.0
                height = 1720
                affected_leg = 'left'
            
            dataviconfilt['gender'] = gender
            dataviconfilt['body_mass'] = body_mass
            dataviconfilt['height'] = height
            dataviconfilt['affected_leg'] = affected_leg
            
            vicon[trialnames[trial]] = dataviconfilt
            
        except:
            print('Cannot import OMCS data for trial ', trialnames[trial]) 

    
    # Analyze xsens data
    for trial in range(0,len(trialnames)):
        try:
            print('Start xsens import of trial: ', trialnames[trial], ' (',trial,'/',len(trialnames),')')
            filepaths, sensortype, fs = data_filelist(xsensfilepaths[trialnames[trial]])
            if len(filepaths) > 0:
                # Define data dictionary with all sensordata
                data_dict = data_preprocessor(filepaths, sensortype)
                data_dict['trialType'] = 'GRAIL'
                                        
                if '900_V_pp15' in trialnames[trial]:
                    data_dict['L'] = data_dict['Right foot']
                    data_dict['Right foot'] = data_dict['Left foot']
                    data_dict['Left foot'] = data_dict['L']
                
                # 900_V_pp01 data collected at 40 Hz sample frequency, correct for that
                if '900_V_01' in xsensfilepaths[trialnames[trial]] and data_dict['trialType'] == 'GRAIL':
                    wrongfs = 40
                    for key in data_dict:
                        if key == 'Timestamp':
                            data_dict[key] = samplerate.resample(data_dict[key], 100/wrongfs, 'sinc_best')
                        elif key == 'Sample Frequency (Hz)':
                            data_dict[key] = data_dict[key]
                        elif key == 'Left foot' or key == 'Right foot' or key == 'Lumbar' or key == 'Sternum':
                            for subkey in data_dict[key]['raw']:
                                if np.shape(data_dict[key]['raw'][subkey])[1] == 3:
                                    a = samplerate.resample(data_dict[key]['raw'][subkey][:,0], 100/wrongfs, 'sinc_best')
                                    b = samplerate.resample(data_dict[key]['raw'][subkey][:,1], 100/wrongfs, 'sinc_best')
                                    c = samplerate.resample(data_dict[key]['raw'][subkey][:,2], 100/wrongfs, 'sinc_best')
                                    data_dict[key]['raw'][subkey] = np.vstack((a,b,c))
                                    data_dict[key]['raw'][subkey] = np.swapaxes(data_dict[key]['raw'][subkey], 0, 1)
                                elif np.shape(data_dict[key]['raw'][subkey])[1] == 4:
                                    a = samplerate.resample(data_dict[key]['raw'][subkey][:,0], 100/wrongfs, 'sinc_best')
                                    b = samplerate.resample(data_dict[key]['raw'][subkey][:,1], 100/wrongfs, 'sinc_best')
                                    c = samplerate.resample(data_dict[key]['raw'][subkey][:,2], 100/wrongfs, 'sinc_best')
                                    d = samplerate.resample(data_dict[key]['raw'][subkey][:,3], 100/wrongfs, 'sinc_best')
                                    data_dict[key]['raw'][subkey] = np.vstack((a,b,c,d))
                                    data_dict[key]['raw'][subkey] = np.swapaxes(data_dict[key]['raw'][subkey], 0, 1)
                
                # Only analyze last 120 seconds of MovingReality trials
                if '1019_pp' in xsensfilepaths[trialnames[trial]]:
                    for var in data_dict['Left foot']['raw']:
                        data_dict['Left foot']['raw'][var] = data_dict['Left foot']['raw'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Left shank']['raw']:
                        data_dict['Left shank']['raw'][var] = data_dict['Left shank']['raw'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Right foot']['raw']:
                        data_dict['Right foot']['raw'][var] = data_dict['Right foot']['raw'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Right shank']['raw']:
                        data_dict['Right shank']['raw'][var] = data_dict['Right shank']['raw'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Lumbar']['raw']:
                        data_dict['Lumbar']['raw'][var] = data_dict['Lumbar']['raw'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Left foot']['derived']:
                        data_dict['Left foot']['derived'][var] = data_dict['Left foot']['derived'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Left shank']['derived']:
                        data_dict['Left shank']['derived'][var] = data_dict['Left shank']['derived'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Right foot']['derived']:
                        data_dict['Right foot']['derived'][var] = data_dict['Right foot']['derived'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Right shank']['derived']:
                        data_dict['Right shank']['derived'][var] = data_dict['Right shank']['derived'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    for var in data_dict['Lumbar']['derived']:
                        data_dict['Lumbar']['derived'][var] = data_dict['Lumbar']['derived'][var][-120*data_dict['Sample Frequency (Hz)']:,:]
                    data_dict['Timestamp'] =  data_dict['Timestamp'][-120*data_dict['Sample Frequency (Hz)']:]
                
                xsens[trialnames[trial]], errors[trialnames[trial]] = feet.process(data_dict, showfigure)

        except:
            print('Cannot import IMU data for trial ', trialnames[trial]) 
        
    return corresponding_files, trialnames, vicon, xsens, errors


def analyze_OMCS(OMCS, IMU, trialnames):
    # Prepare datastructure
    OMCS_gait_events = dict()
    OMCS_gait_characteristics = dict()
    
    # Detect gait events in vicon data
    for f in trialnames:
        if IMU[f]['trialType'] == 'GRAIL':
            # Detect gait events GRAIL vicon data
            OMCS_gait_events[f] = gaiteventdetection(OMCS[f], OMCS[f]['Sample frequency marker data'], algorithmtype ='velocity', trialtype='treadmill')
            
            # Determine spatiotemporal parameters from vicon data
            try:
                OMCS_gait_characteristics[f] = spatiotemporals(OMCS[f], OMCS_gait_events[f], sample_frequency = OMCS[f]['Sample frequency marker data'], trialtype='treadmill')
            except:
                print('Cannot calculate OMCS based gait characteristics for trial ', f) 
                
            # Determine propulsion parameters from vicon data
            try:
                OMCS_gait_events[f], OMCS_gait_characteristics[f], OMCS[f]['Analog data'] = propulsion(OMCS_gait_events[f], OMCS_gait_characteristics[f], OMCS[f]['Analog data'], bodyweight = OMCS[f]['body_mass'])
            except:
                print('Cannot calculate OMCS based propulsion for trial ', f) 
                                
    return OMCS, OMCS_gait_events, OMCS_gait_characteristics


def OMCS_calculate_sacrum_acceleration(OMCS):
    # Prepare datastructure
    sacrum = dict()
    OMCS_POS_Sacrum = dict()
    OMCS_VEL_Sacrum = dict()
    OMCS_ACC_Sacrum = dict()

    for f in OMCS:
        # Create a dictionary for this trial
        sacrum[f] = dict()
        OMCS_POS_Sacrum[f] = dict()
        OMCS_VEL_Sacrum[f] = dict()
        OMCS_ACC_Sacrum[f] = dict()

        try:
            markerdata = OMCS[f]
            fs_markerdata = OMCS[f]['Sample frequency marker data']
                
            # Define markers to process
            keys_to_modify = {"LPSI", "RPSI"}

            # Filtering specified markers
            for key in keys_to_modify.intersection(markerdata.keys()):
                markerdataX = filter_data(2, 15, fs_markerdata, markerdata[key][:,0])    # filter signal: order, fcut, fs, signal
                markerdataY = filter_data(2, 15, fs_markerdata, markerdata[key][:,1])    # filter signal: order, fcut, fs, signal
                markerdataZ = filter_data(2, 15, fs_markerdata, markerdata[key][:,2])    # filter signal: order, fcut, fs, signal
                markerdata[key + ' filtered'] = np.column_stack((markerdataX, markerdataY, markerdataZ))
                for i in range(len(markerdata[key])):
                    if np.all(markerdata[key][i,:] == [0,0,0]) or np.all(np.isnan(markerdata[key])[i,:]):  # Check if X, Y, Z are [0, 0, 0] or [NaN, NaN, NaN]
                        markerdata[key + ' filtered'][i,:] = [np.nan, np.nan, np.nan]  # Set the filtered data to [NaN, NaN, NaN]
            
            # Define Sacrum
            if 'LPSI' in markerdata and 'RPSI' in markerdata:
                sacrum_pos = (markerdata['LPSI filtered'] + markerdata['RPSI filtered']) / 2 # Middle between Left and Right Posterior Superior Iliac Spine
            # Correct for missing data in either LPSI or RPSI marker data
            for i in range(len(markerdata['LPSI'])):
                if np.all(markerdata['LPSI'][i,:] == [0,0,0]) or np.all(markerdata['RPSI'][i,:] == [0,0,0]) or np.all(np.isnan(markerdata['LPSI'])[i,:]) or np.all(np.isnan(markerdata['RPSI'])[i,:]):
                    sacrum_pos[i, :] = [np.nan, np.nan, np.nan]
                    
            # Save sacrum position
            sacrum[f]['Position Sacrum X'] = sacrum_pos[:,0]
            sacrum[f]['Position Sacrum Y'] = sacrum_pos[:,1]
            sacrum[f]['Position Sacrum Z'] = sacrum_pos[:,2]        
            
            OMCS_POS_Sacrum[f] = np.column_stack((sacrum[f]['Position Sacrum X'], sacrum[f]['Position Sacrum Y'], sacrum[f]['Position Sacrum Z'])) 

            # Calculate sacrum velocity
            sacrum_vel = np.gradient(sacrum_pos, 1 / fs_markerdata, axis=0)

            sacrum[f]['Velocity Sacrum X'] = sacrum_vel[:,0]
            sacrum[f]['Velocity Sacrum Y'] = sacrum_vel[:,1]
            sacrum[f]['Velocity Sacrum Z'] = sacrum_vel[:,2]

            OMCS_VEL_Sacrum[f] = np.column_stack((sacrum[f]['Velocity Sacrum X'], sacrum[f]['Velocity Sacrum Y'], sacrum[f]['Velocity Sacrum Z'])) 

            # Calculate sacrum acceleration
            sacrum_acc = np.gradient(sacrum_vel, 1 / fs_markerdata, axis=0)

            sacrum[f]['Acceleration Sacrum X'] = sacrum_acc[:,0]
            sacrum[f]['Acceleration Sacrum Y'] = sacrum_acc[:,1]
            sacrum[f]['Acceleration Sacrum Z'] = sacrum_acc[:,2]

            OMCS_ACC_Sacrum[f] = np.column_stack((sacrum[f]['Acceleration Sacrum X'], sacrum[f]['Acceleration Sacrum Y'], sacrum[f]['Acceleration Sacrum Z'])) 
            
        except:
            print('Cannot calculate OMCS based acceleration for trial ', f) 

    return OMCS_POS_Sacrum, OMCS_VEL_Sacrum, OMCS_ACC_Sacrum


def filter_data(order, fcut, fs, datasignal, **kwargs):
    """
    Zero-phase Butterworth filter using given arguments. Default is lowpass filter.
    :param order: filter order
    :param fcut: cut-off frequency
    :param fs: sample frequency
    :param datasignal: signal to filter
    :param kwargs: optional arguments for filter type, for example 'highpass'.
    :return:
    """
    try:
        b, a = signal.butter(order, fcut / (fs/2), btype = kwargs['type'])
    except:
        b, a = signal.butter(order, fcut / (fs/2))

    signal_filt = signal.filtfilt(b, a, datasignal)

    return signal_filt


def APaccelerationLumbar(gaitevents, gaitcharacteristics, APacceleration, **kwargs):
    """
    Calculate measures from anterior-posterior acceleration lumbar
    """   
    # Set defaults
    sample_frequency = 100 # Sample frequency of the marker data
    bodyweight = 1
    th_crossings = 0 # Set threshold_crossings at 0 m/s^2 to identify crossings in acceleration in AP direction
    # th_crosssteps = -10 * 0.90 * bodyweight # Set threshold_crosssteps at 10 times 90% of the bodyweight to identify cross steps and deem artefact
    debugplot = False
    title = ' '
    
    # Check optional input arguments
    for key, value in kwargs.items():
        if key == 'sample_frequency':
            sample_frequency = value
        if key == 'debugplot':
            debugplot = value
        if key == 'plot_title':
            title = value
        if key == 'bodyweight':
            bodyweight = value


    # Filter acceleration data
    # Cut-off frequency: 20Hz
    fc = 20  # Cut-off frequency of the filter
    omega = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter (scipy.signal.filtfilt is a forwrd-backward linear filter meaning the Nth-order*2 is applied)
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
       
    APacceleration_filtered = signal.filtfilt(b, a, APacceleration) # Apply filter
    
    # Very low pass filter for first segmentation of stance in braking and propulsion areas
    fc = 5  # Cut-off frequency of the filter
    omega = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter (scipy.signal.filtfilt is a forwrd-backward linear filter meaning the Nth-order*2 is applied)
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
    
    lumbar_acc_y = signal.filtfilt(b, a, APacceleration) # Apply filter
               
    # First determine stance phase from IC till TC according to vicon data,
    # Deem cross steps as faulty stance phases to calculate propulsive force,
    # Then find the local minimum,
    # Last find zero crossing around local minima as start and stop of propulsion.
    
    # Left side
    gaitcharacteristics['Stance left index numbers'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Lumbar left start'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Lumbar left stop'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Lumbar left start'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Lumbar left stop'] = np.array([], dtype=int)
        
    for i in range(0, len(gaitevents['Index numbers initial contact left'])):
        try:
            # start = gaitevents['Index numbers initial contact left'][i] # start of stance phase
            # stop = gaitevents['Index numbers terminal contact left'][ gaitevents['Index numbers terminal contact left'] > gaitevents['Index numbers initial contact left'][i] ][0] # end of stance phase
            
            start = gaitevents['Index numbers initial contact left'][i]
            if i + 1 < len(gaitevents['Index numbers initial contact left']):
                next_start = gaitevents['Index numbers initial contact left'][i + 1]
            else:
                next_start = np.inf  # no next start, so set bound to infinity
            valid_stops = gaitevents['Index numbers terminal contact left'][ (gaitevents['Index numbers terminal contact left'] > start) & (gaitevents['Index numbers terminal contact left'] < next_start) ]
            if len(valid_stops) == 0: # If no valid stop, skip this stance phase
                continue
            stop = valid_stops[0]

# Identify crossstep: force in Z direction should cross 90% of the bodyweight, force in Z direction of the contralateral side should reach almost 0 at some point during the stance, force in Z direction should at some point before heel-strike and after toe-off reach almost zero
# if np.min(analogdata['Force Z left filtered'][start:stop]) < th_crosssteps and np.any(analogdata['Force Z right filtered'][start:stop] > -1) and analogdata['Force Z left filtered'][start-10] > -10 and analogdata['Force Z left filtered'][stop+10] > -10: # If not cross step: continue
            # Stance phase with correction for cross steps
            gaitcharacteristics['Stance left index numbers'] = np.append(gaitcharacteristics['Stance left index numbers'], np.arange(start, stop, step=1)) # save the index numbers of the stance phase
            
            # Find local maximum peak in AP Acceleration Lumbar (= braking force)
            maxpeaks = signal.find_peaks(lumbar_acc_y[start+5:stop-10])[0] + start+5
            if len(maxpeaks)>0:
                localmax = np.argmax(lumbar_acc_y[maxpeaks])
                localmax = int(maxpeaks[localmax])
                if lumbar_acc_y[localmax] < th_crossings: # all data is negative and thus propulsion, (no braking was generated)
                    localmax = False
            else: # no braking peaks
                localmax = False
            
            # Find local minimum peak in AP Acceleration Lumbar (= forward force) after the maximum braking
            minpeaks = signal.find_peaks(-lumbar_acc_y[start+10:stop-5])[0] + start+10
            if type(localmax) == int:
                minpeaks = minpeaks[minpeaks>localmax]
            elif localmax == False:
                minpeaks = minpeaks[minpeaks>start]
            if len(minpeaks)>0:
                localmin = np.argmin(lumbar_acc_y[minpeaks])
                localmin = int(minpeaks[localmin])
                if lumbar_acc_y[localmin] > th_crossings: # all data is positive and thus braking, (no propulsion was generated)
                        localmin = False
            else: # no propulsion peaks
                localmin = False

            
            # Find approximate braking to propulsion point at first positive to negative zero crossing in highly filtered signal
            if type(localmin) == int and type(localmax) == int: # both braking and propulsion
                braking_to_propulsion = np.argwhere(lumbar_acc_y[localmax:localmin] < th_crossings) +localmax
                if len(braking_to_propulsion) > 0:
                    braking_to_propulsion = int(braking_to_propulsion[0])
                else:
                    braking_to_propulsion = False # local minimum and local maxium were found, but data not smaller than 0 > only braking
            elif localmin == False or localmax == False: # no braking-to-propulsion transition
                braking_to_propulsion = False
               
                          
            # Find actual braking-to-propulsion point based on 20Hz filtered signal
            if type(braking_to_propulsion) == int:
                signs = np.sign(APacceleration_filtered[int(braking_to_propulsion-10) : int(braking_to_propulsion+10)])
                crossings = np.argwhere(np.diff(signs)<-1) + int(braking_to_propulsion-10) # positive to negative direction
                true_braking_to_propulsion = int(crossings[np.argmin(np.abs(crossings-braking_to_propulsion))])
                if true_braking_to_propulsion < start:
                    if np.nanmean(APacceleration_filtered[start:stop]) < 0: # only propulsion
                        true_braking_to_propulsion = start # assume no braking, only propulsion during this stance phase
                    if np.nanmean(APacceleration_filtered[start:stop]) > 0: # only braking
                        true_braking_to_propulsion = stop # assume no braking, only propulsion during this stance phase
                # gaitevents['AP Deceleration Lumbar left stop'] = np.append(gaitevents['AP Deceleration Lumbar left stop'], true_braking_to_propulsion)
                # gaitevents['AP Acceleration Lumbar left start'] = np.append(gaitevents['AP Acceleration Lumbar left start'], true_braking_to_propulsion)
            
            elif type(braking_to_propulsion) == bool: # no braking-to-propulsion transition
                if type(localmin) == int and localmax == False: # No braking
                    true_braking_to_propulsion = int(start)
                    # gaitevents['AP Deceleration Lumbar left stop'] = np.append(gaitevents['AP Deceleration Lumbar left stop'], start)
                    # gaitevents['AP Acceleration Lumbar left start'] = np.append(gaitevents['AP Acceleration Lumbar left start'], start)
                elif localmin == False and type(localmax) == int: # No propulsion
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Lumbar left stop'] = np.append(gaitevents['AP Deceleration Lumbar left stop'], stop)
                    # gaitevents['AP Acceleration Lumbar left start'] = np.append(gaitevents['AP Acceleration Lumbar left start'], stop)
                elif type(localmin) == int and type(localmax) == int:
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Lumbar left stop'] = np.append(gaitevents['AP Deceleration Lumbar left stop'], stop)
                    # gaitevents['AP Acceleration Lumbar left start'] = np.append(gaitevents['AP Acceleration Lumbar left start'], stop)
                    
            # Find approximate start of braking at "almost zero-crossing" in highly filtered signal    
            if type(localmax) == int:
                signs = np.sign(((lumbar_acc_y/bodyweight)-0.01)[start-10 : localmax])
                crossings = np.argwhere(np.diff(signs)>1) + int(start-10)
                if len(crossings) > 0:
                    start_brake = crossings[-1]
                else:
                    start_brake = np.argmin(((lumbar_acc_y/bodyweight)-0.01)[start : localmax]) + start
            elif type(localmax) == bool:
                start_brake = int(start)
                
            # Find actual start of braking at closest zero-crossing in 20 Hz filterd signal around approximate start of the break in negative to positive direction
            if type(localmax) == int:
                signs = np.sign(APacceleration_filtered[start-10 : localmax])
                crossings = np.argwhere(np.diff(signs)>1) + int(start-10) # negative to positive direction
                if len(crossings) > 0:
                    true_start_brake = int(crossings[np.argmin(np.abs(crossings-start_brake))])
                else:
                    true_start_brake = int(start)
            elif type(localmax) == bool:
                true_start_brake = int(start)
                
            # gaitevents['AP Deceleration Lumbar left start'] = np.append(gaitevents['AP Deceleration Lumbar left start'], true_start_brake)
            
            # Find approximate stop of propulsion at "almost zero-crossing" in highly filtered signal
            if type(localmin) == int:
                signs = np.sign(((lumbar_acc_y/bodyweight)+0.01)[localmin : stop +10])
                crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positve direction
                if len(crossings) > 0:
                    stop_prop = crossings[-1]
                else:
                    stop_prop = np.argmax(((lumbar_acc_y/bodyweight)+0.01)[localmin:stop]) + localmin
            elif type(localmin) == bool:
                stop_prop = int(stop)
            
            # Find actual stop of propulsion at closest zero-crossing in 20 Hz filterd signal around approximate stop of propulsion in negative to positive direction
            if type(localmin) == int:
                signs = np.sign(APacceleration_filtered[localmin : stop +10])
                crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positive direction
                if len(crossings) > 0:
                    true_stop_prop = int(crossings[np.argmin(np.abs(crossings-stop_prop))])
                else:
                    true_stop_prop = int(stop)                    
            elif type(localmin) == bool:
                true_stop_prop = int(stop)

            # gaitevents['AP Acceleration Lumbar left stop'] = np.append(gaitevents['AP Acceleration Lumbar left stop'], true_stop_prop)
            
            if (true_braking_to_propulsion is not None and
                true_start_brake is not None and
                true_stop_prop is not None and
                isinstance(true_braking_to_propulsion, int) and
                isinstance(true_start_brake, int) and
                isinstance(true_stop_prop, int) and
                true_start_brake < true_braking_to_propulsion < true_stop_prop):
            
                gaitevents['AP Deceleration Lumbar left start'] = np.append(
                    gaitevents['AP Deceleration Lumbar left start'], true_start_brake)
                gaitevents['AP Deceleration Lumbar left stop'] = np.append(
                    gaitevents['AP Deceleration Lumbar left stop'], true_braking_to_propulsion)
            
                gaitevents['AP Acceleration Lumbar left start'] = np.append(
                    gaitevents['AP Acceleration Lumbar left start'], true_braking_to_propulsion)
                gaitevents['AP Acceleration Lumbar left stop'] = np.append(
                    gaitevents['AP Acceleration Lumbar left stop'], true_stop_prop)

                if debugplot == True:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(start+5, stop-10), lumbar_acc_y[start+5:stop-10], label='Acceleration', color='black')
                    plt.plot(localmax, lumbar_acc_y[localmax], 'r*', label='Local Max')  # Red stars
                    plt.plot(localmin, lumbar_acc_y[localmin], 'g*', label='Local Min')  # Green stars
                    # plt.plot(braking_to_propulsion, lumbar_acc_y[braking_to_propulsion], 'bs', label='Braking → Propulsion')  # Blue squares
                    plt.plot(true_braking_to_propulsion, lumbar_acc_y[true_braking_to_propulsion], 'bs', label='True Braking → Propulsion')  # Blue squares
                    # plt.plot(int(start_brake), lumbar_acc_y[start_brake], 'mo', label='Start Brake')  # Magenta circles
                    plt.plot(true_start_brake, lumbar_acc_y[true_start_brake], 'mo', label='True Start Brake')  # Yellow circles
                    # plt.plot(int(stop_prop), lumbar_acc_y[stop_prop], 'co', label='Stop Propulsion')  # Cyan circles
                    plt.plot(true_stop_prop, lumbar_acc_y[true_stop_prop], 'co', label='True Stop Propulsion')  # Black circles
                    plt.grid(True)
                    plt.legend(loc='upper right', fontsize='small', frameon=True)
                    plt.title('Stance Phase with Gait Events - left')
                    plt.xlabel('Time (samples)')
                    plt.ylabel('Acceleration (lumbar y-axis)')
                    plt.tight_layout()
                    plt.show()

            else:
                # print(f"Skipping stance {i}: invalid brake/propulsion pair.")
                continue
            
        except:
            pass
                
    # Right side
    gaitcharacteristics['Stance right index numbers'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Lumbar right start'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Lumbar right stop'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Lumbar right start'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Lumbar right stop'] = np.array([], dtype=int)
    
    for i in range(0, len(gaitevents['Index numbers initial contact right'])):
        try:
            # start = gaitevents['Index numbers initial contact right'][i] # start of stance phase
            # stop = gaitevents['Index numbers terminal contact right'][ gaitevents['Index numbers terminal contact right'] > gaitevents['Index numbers initial contact right'][i] ][0] # end of stance phase
            
            start = gaitevents['Index numbers initial contact right'][i]
            if i + 1 < len(gaitevents['Index numbers initial contact right']):
                next_start = gaitevents['Index numbers initial contact right'][i + 1]
            else:
                next_start = np.inf  # no next start, so set bound to infinity
            valid_stops = gaitevents['Index numbers terminal contact right'][ (gaitevents['Index numbers terminal contact right'] > start) & (gaitevents['Index numbers terminal contact right'] < next_start) ]
            if len(valid_stops) == 0: # If no valid stop, skip this stance phase
                continue
            stop = valid_stops[0]

# Identify crossstep: force in Z direction should cross 90% of the bodyweight, force in Z direction of the contralateral side should reach almost 0 at some point during the stance, force in Z direction should at some point before heel-strike and after toe-off reach almost zero
# if np.min(analogdata['Force Z right filtered'][start:stop]) < th_crosssteps and np.any(analogdata['Force Z left filtered'][start:stop] > -1) and analogdata['Force Z right filtered'][start-10] > -10 and analogdata['Force Z right filtered'][stop+10] > -10: # If not cross step: continue
            # Stance phase with correction for cross steps
            gaitcharacteristics['Stance right index numbers'] = np.append(gaitcharacteristics['Stance right index numbers'], np.arange(start, stop, step=1)) # save the index numbers of the stance phase
            
            # Find local maximum peak in strongly filtered Y force (= braking force)
            maxpeaks = signal.find_peaks(lumbar_acc_y[start+5:stop-10])[0] + start+5
            if len(maxpeaks)>0:
                localmax = np.argmax(lumbar_acc_y[maxpeaks])
                localmax = int(maxpeaks[localmax])
                if lumbar_acc_y[localmax] < th_crossings: # all data is negative and thus propulsion, (no braking force was generated)
                    localmax = False
            else: # no braking peaks
                localmax = False
            
            # Find local minimum peak in strongly filtered Y force (= forward force) after the maximum braking
            minpeaks = signal.find_peaks(-lumbar_acc_y[start+10:stop-5])[0] + start+10
            if type(localmax) == int:
                minpeaks = minpeaks[minpeaks>localmax]
            elif localmax == False:
                minpeaks = minpeaks[minpeaks>start]
            if len(minpeaks)>0:
                localmin = np.argmin(lumbar_acc_y[minpeaks])
                localmin = int(minpeaks[localmin])
                if lumbar_acc_y[localmin] > th_crossings: # all data is positive and thus braking, (no propulsive forcef was generated)
                        localmin = False
            else: # no propulsion peaks
                localmin = False

            
            # Find approximate braking to propulsion point at first positive to negative zero crossing in highly filtered signal
            if type(localmin) == int and type(localmax) == int: # both braking and propulsion
                braking_to_propulsion = np.argwhere(lumbar_acc_y[localmax:localmin] < th_crossings) +localmax
                if len(braking_to_propulsion) > 0:
                    braking_to_propulsion = int(braking_to_propulsion[0])
                else:
                    braking_to_propulsion = False # local minimum and local maxium were found, but data not smaller than 0 > only braking
            elif localmin == False or localmax == False: # no braking-to-propulsion transition
                braking_to_propulsion = False
               
                          
            # Find actual braking-to-propulsion point based on 20Hz filtered signal
            if type(braking_to_propulsion) == int:
                signs = np.sign(APacceleration_filtered[int(braking_to_propulsion-10) : int(braking_to_propulsion+10)])
                crossings = np.argwhere(np.diff(signs)<-1) + int(braking_to_propulsion-10) # positive to negative direction
                true_braking_to_propulsion = int(crossings[np.argmin(np.abs(crossings-braking_to_propulsion))])
                if true_braking_to_propulsion < start:
                    if np.nanmean(APacceleration_filtered[start:stop]) < 0: # only propulsion
                        true_braking_to_propulsion = start # assume no braking, only propulsion during this stance phase
                    if np.nanmean(APacceleration_filtered[start:stop]) > 0: # only braking
                        true_braking_to_propulsion = stop # assume no braking, only propulsion during this stance phase
                # gaitevents['AP Deceleration Lumbar right stop'] = np.append(gaitevents['AP Deceleration Lumbar right stop'], true_braking_to_propulsion)
                # gaitevents['AP Acceleration Lumbar right start'] = np.append(gaitevents['AP Acceleration Lumbar right start'], true_braking_to_propulsion)
            
            elif type(braking_to_propulsion) == bool: # no braking-to-propulsion transition
                if type(localmin) == int and localmax == False: # No braking
                    true_braking_to_propulsion = int(start)
                    # gaitevents['AP Deceleration Lumbar right stop'] = np.append(gaitevents['AP Deceleration Lumbar right stop'], start)
                    # gaitevents['AP Acceleration Lumbar right start'] = np.append(gaitevents['AP Acceleration Lumbar right start'], start)
                elif localmin == False and type(localmax) == int: # No propulsion
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Lumbar right stop'] = np.append(gaitevents['AP Deceleration Lumbar right stop'], stop)
                    # gaitevents['AP Acceleration Lumbar right start'] = np.append(gaitevents['AP Acceleration Lumbar right start'], stop)
                elif type(localmin) == int and type(localmax) == int:
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Lumbar right stop'] = np.append(gaitevents['AP Deceleration Lumbar right stop'], stop)
                    # gaitevents['AP Acceleration Lumbar right start'] = np.append(gaitevents['AP Acceleration Lumbar right start'], stop)
                    
            # Find approximate start of braking at "almost zero-crossing" in highly filtered signal    
            if type(localmax) == int:
                signs = np.sign(((lumbar_acc_y/bodyweight)-0.01)[start-10 : localmax])
                crossings = np.argwhere(np.diff(signs)>1) + int(start-10)
                if len(crossings) > 0:
                    start_brake = crossings[-1]
                else:
                    start_brake = np.argmin(((lumbar_acc_y/bodyweight)-0.01)[start : localmax]) + start
            elif type(localmax) == bool:
                start_brake = int(start)
                
            # Find actual start of braking at closest zero-crossing in 20 Hz filterd signal around approximate start of the break in negative to positive direction
            if type(localmax) == int:
                signs = np.sign(APacceleration_filtered[start-10 : localmax])
                crossings = np.argwhere(np.diff(signs)>1) + int(start-10) # negative to positive direction
                if len(crossings) > 0:
                    true_start_brake = int(crossings[np.argmin(np.abs(crossings-start_brake))])
                else:
                    true_start_brake = int(start)
            elif type(localmax) == bool:
                true_start_brake = int(start)
                
            # gaitevents['AP Deceleration Lumbar right start'] = np.append(gaitevents['AP Deceleration Lumbar right start'], true_start_brake)
            
            # Find approximate stop of propulsion at "almost zero-crossing" in highly filtered signal
            if type(localmin) == int:
                signs = np.sign(((lumbar_acc_y/bodyweight)+0.01)[localmin : stop +10])
                crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positve direction
                if len(crossings) > 0:
                    stop_prop = crossings[-1]
                else:
                    stop_prop = np.argmax(((lumbar_acc_y/bodyweight)+0.01)[localmin:stop]) + localmin
            elif type(localmin) == bool:
                stop_prop = int(stop)
            
            # Find actual stop of propulsion at closest zero-crossing in 20 Hz filterd signal around approximate stop of propulsion in negative to positive direction
            if type(localmin) == int:
                signs = np.sign(APacceleration_filtered[localmin : stop +10])
                crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positive direction
                if len(crossings) > 0:
                    true_stop_prop = int(crossings[np.argmin(np.abs(crossings-stop_prop))])
                else:
                    true_stop_prop = int(stop)                    
            elif type(localmin) == bool:
                true_stop_prop = int(stop)

            # gaitevents['AP Acceleration Lumbar right stop'] = np.append(gaitevents['AP Acceleration Lumbar right stop'], true_stop_prop)
            
            if (true_braking_to_propulsion is not None and
                true_start_brake is not None and
                true_stop_prop is not None and
                isinstance(true_braking_to_propulsion, int) and
                isinstance(true_start_brake, int) and
                isinstance(true_stop_prop, int) and
                true_start_brake < true_braking_to_propulsion < true_stop_prop):
            
                gaitevents['AP Deceleration Lumbar right start'] = np.append(
                    gaitevents['AP Deceleration Lumbar right start'], true_start_brake)
                gaitevents['AP Deceleration Lumbar right stop'] = np.append(
                    gaitevents['AP Deceleration Lumbar right stop'], true_braking_to_propulsion)
            
                gaitevents['AP Acceleration Lumbar right start'] = np.append(
                    gaitevents['AP Acceleration Lumbar right start'], true_braking_to_propulsion)
                gaitevents['AP Acceleration Lumbar right stop'] = np.append(
                    gaitevents['AP Acceleration Lumbar right stop'], true_stop_prop)

                if debugplot == True:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(start+5, stop-10), lumbar_acc_y[start+5:stop-10], label='Acceleration', color='black')
                    plt.plot(localmax, lumbar_acc_y[localmax], 'r*', label='Local Max')  # Red stars
                    plt.plot(localmin, lumbar_acc_y[localmin], 'g*', label='Local Min')  # Green stars
                    # plt.plot(braking_to_propulsion, lumbar_acc_y[braking_to_propulsion], 'bs', label='Braking → Propulsion')  # Blue squares
                    plt.plot(true_braking_to_propulsion, lumbar_acc_y[true_braking_to_propulsion], 'bs', label='True Braking → Propulsion')  # Blue squares
                    # plt.plot(int(start_brake), lumbar_acc_y[start_brake], 'mo', label='Start Brake')  # Magenta circles
                    plt.plot(true_start_brake, lumbar_acc_y[true_start_brake], 'mo', label='True Start Brake')  # Yellow circles
                    # plt.plot(int(stop_prop), lumbar_acc_y[stop_prop], 'co', label='Stop Propulsion')  # Cyan circles
                    plt.plot(true_stop_prop, lumbar_acc_y[true_stop_prop], 'co', label='True Stop Propulsion')  # Black circles
                    plt.grid(True)
                    plt.legend(loc='upper right', fontsize='small', frameon=True)
                    plt.title('Stance Phase with Gait Events - right')
                    plt.xlabel('Time (samples)')
                    plt.ylabel('Acceleration (lumbar y-axis)')
                    plt.tight_layout()
                    plt.show()

            else:
                # print(f"Skipping stance {i}: invalid brake/propulsion pair.")
                continue
            
        except:
            pass
    
    
    # Remove acceleration start/stop events in first 10 seconds of trial
    gaitevents['AP Acceleration Lumbar left start'] = gaitevents['AP Acceleration Lumbar left start'][gaitevents['AP Acceleration Lumbar left start'] > 10*sample_frequency]
    try:
        gaitevents['AP Acceleration Lumbar left stop'] = gaitevents['AP Acceleration Lumbar left stop'][gaitevents['AP Acceleration Lumbar left stop'] >= gaitevents['AP Acceleration Lumbar left start'][0]]
    except IndexError:
        gaitevents['AP Acceleration Lumbar left stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Acceleration Lumbar left start'] = gaitevents['AP Acceleration Lumbar left start'][gaitevents['AP Acceleration Lumbar left start'] <= gaitevents['AP Acceleration Lumbar left stop'][-1]]
    except IndexError:
        gaitevents['AP Acceleration Lumbar left start'] = np.array([], dtype=int)
    
    gaitevents['AP Deceleration Lumbar left start'] = gaitevents['AP Deceleration Lumbar left start'][gaitevents['AP Deceleration Lumbar left start'] > 10*sample_frequency]
    try:
        gaitevents['AP Deceleration Lumbar left stop'] = gaitevents['AP Deceleration Lumbar left stop'][gaitevents['AP Deceleration Lumbar left stop'] >= gaitevents['AP Deceleration Lumbar left start'][0]]
    except IndexError:
        gaitevents['AP Deceleration Lumbar left stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Deceleration Lumbar left start'] = gaitevents['AP Deceleration Lumbar left start'][gaitevents['AP Deceleration Lumbar left start'] <= gaitevents['AP Deceleration Lumbar left stop'][-1]]
    except IndexError:
        gaitevents['AP Deceleration Lumbar left start'] = np.array([], dtype=int)
    
    gaitevents['AP Acceleration Lumbar right start'] = gaitevents['AP Acceleration Lumbar right start'][gaitevents['AP Acceleration Lumbar right start'] > 10*sample_frequency]
    try:
        gaitevents['AP Acceleration Lumbar right stop'] = gaitevents['AP Acceleration Lumbar right stop'][gaitevents['AP Acceleration Lumbar right stop'] >= gaitevents['AP Acceleration Lumbar right start'][0]]
    except IndexError:
        gaitevents['AP Acceleration Lumbar right stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Acceleration Lumbar right start'] = gaitevents['AP Acceleration Lumbar right start'][gaitevents['AP Acceleration Lumbar right start'] <= gaitevents['AP Acceleration Lumbar right stop'][-1]]
    except IndexError:
        gaitevents['AP Acceleration Lumbar right start'] = np.array([], dtype=int)
    
    gaitevents['AP Deceleration Lumbar right start'] = gaitevents['AP Deceleration Lumbar right start'][gaitevents['AP Deceleration Lumbar right start'] > 10*sample_frequency]
    try:
        gaitevents['AP Deceleration Lumbar right stop'] = gaitevents['AP Deceleration Lumbar right stop'][gaitevents['AP Deceleration Lumbar right stop'] >= gaitevents['AP Deceleration Lumbar right start'][0]]
    except IndexError:
        gaitevents['AP Deceleration Lumbar right stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Deceleration Lumbar right start'] = gaitevents['AP Deceleration Lumbar right start'][gaitevents['AP Deceleration Lumbar right start'] <= gaitevents['AP Deceleration Lumbar right stop'][-1]]
    except IndexError:
        gaitevents['AP Deceleration Lumbar right start'] = np.array([], dtype=int)
    
    
    # Peak deceleration and acceleration
    gaitevents['Peak AP Acceleration Lumbar left'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Acceleration Lumbar left start'])):
        try:
            idxmin = np.argmin(APacceleration_filtered [gaitevents['AP Acceleration Lumbar left start'][i] : gaitevents['AP Acceleration Lumbar left stop'][i]])
            gaitevents['Peak AP Acceleration Lumbar left'] = np.append(gaitevents['Peak AP Acceleration Lumbar left'], gaitevents['AP Acceleration Lumbar left start'][i]+idxmin)
        except ValueError:
            pass
    gaitevents['Peak AP Deceleration Lumbar left'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Deceleration Lumbar left start'])):
        try:
            idxmax = np.argmax(APacceleration_filtered [gaitevents['AP Deceleration Lumbar left start'][i] : gaitevents['AP Deceleration Lumbar left stop'][i]])
            gaitevents['Peak AP Deceleration Lumbar left'] = np.append(gaitevents['Peak AP Deceleration Lumbar left'], gaitevents['AP Deceleration Lumbar left start'][i]+idxmax)
        except ValueError:
            pass
    gaitevents['Peak AP Acceleration Lumbar right'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Acceleration Lumbar right start'])):
        try:
            idxmin = np.argmin(APacceleration_filtered [gaitevents['AP Acceleration Lumbar right start'][i] : gaitevents['AP Acceleration Lumbar right stop'][i]])
            gaitevents['Peak AP Acceleration Lumbar right'] = np.append(gaitevents['Peak AP Acceleration Lumbar right'], gaitevents['AP Acceleration Lumbar right start'][i]+idxmin)
        except ValueError:
            pass
    gaitevents['Peak AP Deceleration Lumbar right'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Deceleration Lumbar right start'])):
        try:
            idxmax = np.argmax(APacceleration_filtered [gaitevents['AP Deceleration Lumbar right start'][i] : gaitevents['AP Deceleration Lumbar right stop'][i]])
            gaitevents['Peak AP Deceleration Lumbar right'] = np.append(gaitevents['Peak AP Deceleration Lumbar right'], gaitevents['AP Deceleration Lumbar right start'][i]+idxmax)
        except ValueError:
            pass
    
    # Debug plot
    if debugplot == True:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        axs[0].set_title(title, fontsize=20)
        # Left
        axs[0].plot(APacceleration_filtered/bodyweight, 'k', label='Acceleration Lumbar Y left')
        axs[0].plot(lumbar_acc_y/bodyweight, 'grey', label='Filtered Acceleration Lumbar Y left')
        # axs[0].plot(markerdata['Acceleration Lumbar Z filtered']/bodyweight, 'orange', label='Acceleration Lumbar Z left')
        axs[0].plot(gaitevents['Index numbers initial contact left'], APacceleration_filtered[gaitevents['Index numbers initial contact left']]/bodyweight, 'r.')
        axs[0].plot(gaitevents['Index numbers terminal contact left'], APacceleration_filtered[gaitevents['Index numbers terminal contact left']]/bodyweight, 'g.')
        # axs[0].plot(gaitevents['AP Deceleration Lumbar left start'], APacceleration_filtered[gaitevents['AP Deceleration Lumbar left start']]/bodyweight, 'kx', label='Braking start')
        axs[0].vlines(x=gaitevents['AP Deceleration Lumbar left start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='red')
        axs[0].vlines(x=gaitevents['AP Acceleration Lumbar left start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='grey')
        axs[0].vlines(x=gaitevents['AP Acceleration Lumbar left stop'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='green')
        # axs[0].plot(gaitevents['AP Acceleration Lumbar left stop'], APacceleration_filtered[gaitevents['AP Acceleration Lumbar left stop']]/bodyweight, 'kx', label='Propulsion stop')
        axs[0].plot(gaitevents['Peak AP Acceleration Lumbar left'], APacceleration_filtered[gaitevents['Peak AP Acceleration Lumbar left']]/bodyweight, 'gx', label='AP Acceleration Lumbar peak')
        axs[0].plot(gaitevents['Peak AP Deceleration Lumbar left'], APacceleration_filtered[gaitevents['Peak AP Deceleration Lumbar left']]/bodyweight, 'rx', label='AP Deceleration Lumbar peak')
        axs[0].hlines(xmin=0, xmax=len(APacceleration_filtered), y=0, color='grey')
        
        for i in range(0, len(gaitevents['AP Acceleration Lumbar left start'])):
            axs[0].fill_between(x=np.arange(gaitevents['AP Acceleration Lumbar left start'][i], gaitevents['AP Acceleration Lumbar left stop'][i]), y1=APacceleration_filtered[gaitevents['AP Acceleration Lumbar left start'][i] : gaitevents['AP Acceleration Lumbar left stop'][i]]/bodyweight, y2=0, color='lightgreen')
        for i in range(0, len(gaitevents['AP Deceleration Lumbar left start'])):
            axs[0].fill_between(x=np.arange(gaitevents['AP Deceleration Lumbar left start'][i], gaitevents['AP Deceleration Lumbar left stop'][i]), y1=APacceleration_filtered[gaitevents['AP Deceleration Lumbar left start'][i] : gaitevents['AP Deceleration Lumbar left stop'][i]]/bodyweight, y2=0, color='pink')
                
        #Right
        axs[1].plot(APacceleration_filtered/bodyweight, 'k', label='Acceleration Lumbar Y right')
        axs[1].plot(lumbar_acc_y/bodyweight, 'grey', label='Filtered Acceleration Lumbar Y right')
        # axs[1].plot(markerdata['Acceleration Lumbar Z filtered']/bodyweight, 'orange', label='Acceleration Lumbar Z')
        axs[1].plot(gaitevents['Index numbers initial contact right'], APacceleration_filtered[gaitevents['Index numbers initial contact right']]/bodyweight, 'r.', label = 'IC')
        axs[1].plot(gaitevents['Index numbers terminal contact right'], APacceleration_filtered[gaitevents['Index numbers terminal contact right']]/bodyweight, 'g.', label = 'TC')
        # axs[1].plot(gaitevents['AP Acceleration Lumbar right start'], APacceleration_filtered[gaitevents['AP Acceleration Lumbar right start']]/bodyweight, 'gv', label='Propulsion start')
        # axs[1].plot(gaitevents['AP Acceleration Lumbar right stop'], APacceleration_filtered[gaitevents['AP Acceleration Lumbar right stop']]/bodyweight, 'rv', label='Propulsion stop')
        axs[1].vlines(x=gaitevents['AP Deceleration Lumbar right start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='red')
        axs[1].vlines(x=gaitevents['AP Acceleration Lumbar right start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='grey')
        axs[1].vlines(x=gaitevents['AP Acceleration Lumbar right stop'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='green')
        axs[1].plot(gaitevents['Peak AP Acceleration Lumbar right'], APacceleration_filtered[gaitevents['Peak AP Acceleration Lumbar right']]/bodyweight, 'gx', label='AP Acceleration Lumbar peak')
        axs[1].plot(gaitevents['Peak AP Deceleration Lumbar right'], APacceleration_filtered[gaitevents['Peak AP Deceleration Lumbar right']]/bodyweight, 'rx', label='AP Deceleration Lumbar peak')
        axs[1].hlines(xmin=0, xmax=len(APacceleration_filtered), y=0, color='grey')
        
        for i in range(0, len(gaitevents['AP Acceleration Lumbar right start'])):
            axs[1].fill_between(x=np.arange(gaitevents['AP Acceleration Lumbar right start'][i], gaitevents['AP Acceleration Lumbar right stop'][i]), y1=APacceleration_filtered[gaitevents['AP Acceleration Lumbar right start'][i] : gaitevents['AP Acceleration Lumbar right stop'][i]]/bodyweight, y2=0, color='lightgreen')
        
        for i in range(0, len(gaitevents['AP Deceleration Lumbar right start'])):
            axs[1].fill_between(x=np.arange(gaitevents['AP Deceleration Lumbar right start'][i], gaitevents['AP Deceleration Lumbar right stop'][i]), y1=APacceleration_filtered[gaitevents['AP Deceleration Lumbar right start'][i] : gaitevents['AP Deceleration Lumbar right stop'][i]]/bodyweight, y2=0, color='pink')
        axs[1].legend()

        
    # Left side
    # AP Acceleration Lumbar = area under the negative curve
    gaitcharacteristics['AP Acceleration Lumbar left'] = np.zeros(shape=(len(gaitevents['AP Acceleration Lumbar left start']),3)) *np.nan
    for i in range(len(gaitevents['AP Acceleration Lumbar left start'])):
        gaitcharacteristics['AP Acceleration Lumbar left'][i,0] = gaitevents['AP Acceleration Lumbar left start'][i]
        gaitcharacteristics['AP Acceleration Lumbar left'][i,1] = gaitevents['AP Acceleration Lumbar left stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_propulsion = APacceleration_filtered[gaitevents['AP Acceleration Lumbar left start'][i]:gaitevents['AP Acceleration Lumbar left stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Acceleration Lumbar left'][i,2] = forward_acceleration - backward_acceleration
        if gaitcharacteristics['AP Acceleration Lumbar left'][i,2] < 0:
            gaitcharacteristics['AP Acceleration Lumbar left'][i,2]= np.nan
    # Peak AP Acceleration Lumbar
    gaitcharacteristics['Peak AP Acceleration Lumbar left'] = np.zeros(shape=(len(gaitevents['Peak AP Acceleration Lumbar left']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Acceleration Lumbar left'])):
        gaitcharacteristics['Peak AP Acceleration Lumbar left'][i,0] = gaitevents['Peak AP Acceleration Lumbar left'][i]
        gaitcharacteristics['Peak AP Acceleration Lumbar left'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Acceleration Lumbar left'][i]])/bodyweight
    # AP Deceleration Lumbar = area under the curve
    gaitcharacteristics['AP Deceleration Lumbar left'] = np.zeros(shape=(len(gaitevents['AP Deceleration Lumbar left start']),3)) *np.nan
    for i in range(len(gaitevents['AP Deceleration Lumbar left start'])):
        gaitcharacteristics['AP Deceleration Lumbar left'][i,0] = gaitevents['AP Deceleration Lumbar left start'][i]
        gaitcharacteristics['AP Deceleration Lumbar left'][i,1] = gaitevents['AP Deceleration Lumbar left stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_brake = APacceleration_filtered[gaitevents['AP Deceleration Lumbar left start'][i]:gaitevents['AP Deceleration Lumbar left stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_brake[this_brake<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_brake[this_brake>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Deceleration Lumbar left'][i,2] = backward_acceleration - forward_acceleration
        if gaitcharacteristics['AP Deceleration Lumbar left'][i,2] < 0:
            gaitcharacteristics['AP Deceleration Lumbar left'][i,2]= np.nan
    # Peak AP Deceleration Lumbar
    gaitcharacteristics['Peak AP Deceleration Lumbar left'] = np.zeros(shape=(len(gaitevents['Peak AP Deceleration Lumbar left']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Deceleration Lumbar left'])):
        gaitcharacteristics['Peak AP Deceleration Lumbar left'][i,0] = gaitevents['Peak AP Deceleration Lumbar left'][i]
        gaitcharacteristics['Peak AP Deceleration Lumbar left'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Deceleration Lumbar left'][i]])/bodyweight
    
    # Right side
    # AP Acceleration Lumbar = area under the negative curve
    gaitcharacteristics['AP Acceleration Lumbar right'] = np.zeros(shape=(len(gaitevents['AP Acceleration Lumbar right start']),3)) *np.nan
    for i in range(len(gaitevents['AP Acceleration Lumbar right start'])):
        gaitcharacteristics['AP Acceleration Lumbar right'][i,0] = gaitevents['AP Acceleration Lumbar right start'][i]
        gaitcharacteristics['AP Acceleration Lumbar right'][i,1] = gaitevents['AP Acceleration Lumbar right stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_propulsion = APacceleration_filtered[gaitevents['AP Acceleration Lumbar right start'][i]:gaitevents['AP Acceleration Lumbar right stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Acceleration Lumbar right'][i,2] = forward_acceleration - backward_acceleration
        if gaitcharacteristics['AP Acceleration Lumbar right'][i,2] < 0:
            gaitcharacteristics['AP Acceleration Lumbar right'][i,2]= np.nan
    # Peak AP Acceleration Lumbar
    gaitcharacteristics['Peak AP Acceleration Lumbar right'] = np.zeros(shape=(len(gaitevents['Peak AP Acceleration Lumbar right']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Acceleration Lumbar right'])):
        gaitcharacteristics['Peak AP Acceleration Lumbar right'][i,0] = gaitevents['Peak AP Acceleration Lumbar right'][i]
        gaitcharacteristics['Peak AP Acceleration Lumbar right'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Acceleration Lumbar right'][i]])/bodyweight
    # AP Deceleration Lumbar = area under the curve
    gaitcharacteristics['AP Deceleration Lumbar right'] = np.zeros(shape=(len(gaitevents['AP Deceleration Lumbar right start']),3)) *np.nan
    for i in range(len(gaitevents['AP Deceleration Lumbar right start'])):
        gaitcharacteristics['AP Deceleration Lumbar right'][i,0] = gaitevents['AP Deceleration Lumbar right start'][i]
        gaitcharacteristics['AP Deceleration Lumbar right'][i,1] = gaitevents['AP Deceleration Lumbar right stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_brake = APacceleration_filtered[gaitevents['AP Deceleration Lumbar right start'][i]:gaitevents['AP Deceleration Lumbar right stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_brake[this_brake<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_brake[this_brake>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Deceleration Lumbar right'][i,2] = backward_acceleration - forward_acceleration
        if gaitcharacteristics['AP Deceleration Lumbar right'][i,2] < 0:
            gaitcharacteristics['AP Deceleration Lumbar right'][i,2]= np.nan
    # Peak AP Deceleration Lumbar
    gaitcharacteristics['Peak AP Deceleration Lumbar right'] = np.zeros(shape=(len(gaitevents['Peak AP Deceleration Lumbar right']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Deceleration Lumbar right'])):
        gaitcharacteristics['Peak AP Deceleration Lumbar right'][i,0] = gaitevents['Peak AP Deceleration Lumbar right'][i]
        gaitcharacteristics['Peak AP Deceleration Lumbar right'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Deceleration Lumbar right'][i]])/bodyweight
    
    
    # Replace zeros with NaN values
    parameters = [
        'AP Deceleration Lumbar left', 'AP Deceleration Lumbar right',
        'AP Acceleration Lumbar left', 'AP Acceleration Lumbar right'
    ]
    for param in parameters:
        for i in range(len(gaitcharacteristics[param])):  
            if gaitcharacteristics[param][i, 2] == 0:       # Check if third column is 0
                gaitcharacteristics[param][i, 2] = np.nan   # Replace with NaN
    
       
    return gaitevents, gaitcharacteristics, APacceleration


# %% Helper functions for debug plots
def plot_Acceleration_OMCS_and_IMU(OMCS_ACC_Sacrum, IMU_ACC_SF_Lumbar, IMU_ACC_EF_Lumbar, IMU_ACC_BF_Lumbar, trial, lower_X_lim, upper_X_lim):
    
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
        axs[0, 0].legend(loc='upper right')

        # OMCS Y-axis
        axs[1, 0].plot(OMCS_ACC_data[:, 1], label='Y-axis', color='g')
        axs[1, 0].axhline(np.nanmean(OMCS_ACC_data[:, 1]), color='k', linestyle='--', label='Mean')
        axs[1, 0].set_ylabel('Acceleration (mm/s²)')
        axs[1, 0].set_title('OMCS Acc Sacrum - Y-axis')
        axs[1, 0].grid(True)
        axs[1, 0].legend(loc='upper right')

        # OMCS Z-axis
        axs[2, 0].plot(OMCS_ACC_data[:, 2], label='Z-axis', color='b')
        axs[2, 0].axhline(np.nanmean(OMCS_ACC_data[:, 2]), color='k', linestyle='--', label='Mean')
        axs[2, 0].set_xlabel('Time (samples)')
        axs[2, 0].set_ylabel('Acceleration (mm/s²)')
        axs[2, 0].set_title('OMCS Acc Sacrum - Z-axis')
        axs[2, 0].grid(True)
        axs[2, 0].legend(loc='upper right')

        # IMU SF X-axis
        axs[0, 1].plot(IMU_ACC_data_SF[:, 0], label='X-axis', color='r')
        axs[0, 1].axhline(np.nanmean(IMU_ACC_data_SF[:, 0]), color='k', linestyle='--', label='Mean')
        axs[0, 1].set_ylabel('Acceleration (m/s²)')
        axs[0, 1].set_title('IMU Acc Lumbar (Sensor Frame) - X-axis')
        axs[0, 1].grid(True)
        axs[0, 1].legend(loc='upper right')

        # IMU SF Y-axis
        axs[1, 1].plot(IMU_ACC_data_SF[:, 1], label='Y-axis', color='g')
        axs[1, 1].axhline(np.nanmean(IMU_ACC_data_SF[:, 1]), color='k', linestyle='--', label='Mean')
        axs[1, 1].set_ylabel('Acceleration (m/s²)')
        axs[1, 1].set_title('IMU Acc Lumbar (Sensor Frame) - Y-axis')
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='upper right')

        # IMU SF Z-axis
        axs[2, 1].plot(IMU_ACC_data_SF[:, 2], label='Z-axis', color='b')
        axs[2, 1].axhline(np.nanmean(IMU_ACC_data_SF[:, 2]), color='k', linestyle='--', label='Mean')
        axs[2, 1].set_xlabel('Time (samples)')
        axs[2, 1].set_ylabel('Acceleration (m/s²)')
        axs[2, 1].set_title('IMU Acc Lumbar (Sensor Frame) - Z-axis')
        axs[2, 1].grid(True)
        axs[2, 1].legend(loc='upper right')
        
        # IMU EF X-axis
        axs[0, 2].plot(IMU_ACC_data_EF[:, 0], label='X-axis', color='r')
        axs[0, 2].axhline(np.nanmean(IMU_ACC_data_EF[:, 0]), color='k', linestyle='--', label='Mean')
        axs[0, 2].set_ylabel('Acceleration (m/s²)')
        axs[0, 2].set_title('IMU Acc Lumbar (Earth Frame) - X-axis')
        axs[0, 2].grid(True)
        axs[0, 2].legend(loc='upper right')

        # IMU EF Y-axis
        axs[1, 2].plot(IMU_ACC_data_EF[:, 1], label='Y-axis', color='g')
        axs[1, 2].axhline(np.nanmean(IMU_ACC_data_EF[:, 1]), color='k', linestyle='--', label='Mean')
        axs[1, 2].set_ylabel('Acceleration (m/s²)')
        axs[1, 2].set_title('IMU Acc Lumbar (Earth Frame) - Y-axis')
        axs[1, 2].grid(True)
        axs[1, 2].legend(loc='upper right')

        # IMU EF Z-axis
        axs[2, 2].plot(IMU_ACC_data_EF[:, 2], label='Z-axis', color='b')
        axs[2, 2].axhline(np.nanmean(IMU_ACC_data_EF[:, 2]), color='k', linestyle='--', label='Mean')
        axs[2, 2].set_xlabel('Time (samples)')
        axs[2, 2].set_ylabel('Acceleration (m/s²)')
        axs[2, 2].set_title('IMU Acc Lumbar (Earth Frame) - Z-axis')
        axs[2, 2].grid(True)
        axs[2, 2].legend(loc='upper right')

        # IMU BF X-axis
        axs[0, 3].plot(IMU_ACC_data_BF[:, 0], label='X-axis', color='r')
        axs[0, 3].axhline(np.nanmean(IMU_ACC_data_BF[:, 0]), color='k', linestyle='--', label='Mean')
        axs[0, 3].set_ylabel('Acceleration (m/s²)')
        axs[0, 3].set_title('IMU Acc Lumbar (Body Frame) - X-axis')
        axs[0, 3].grid(True)
        axs[0, 3].legend(loc='upper right')

        # IMU BF Y-axis
        axs[1, 3].plot(IMU_ACC_data_BF[:, 1], label='Y-axis', color='g')
        axs[1, 3].axhline(np.nanmean(IMU_ACC_data_BF[:, 1]), color='k', linestyle='--', label='Mean')
        axs[1, 3].set_ylabel('Acceleration (m/s²)')
        axs[1, 3].set_title('IMU Acc Lumbar (Body Frame) - Y-axis')
        axs[1, 3].grid(True)
        axs[1, 3].legend(loc='upper right')

        # IMU BF Z-axis
        axs[2, 3].plot(IMU_ACC_data_BF[:, 2], label='Z-axis', color='b')
        axs[2, 3].axhline(np.nanmean(IMU_ACC_data_BF[:, 2]), color='k', linestyle='--', label='Mean')
        axs[2, 3].set_xlabel('Time (samples)')
        axs[2, 3].set_ylabel('Acceleration (m/s²)')
        axs[2, 3].set_title('IMU Acc Lumbar (Body Frame) - Z-axis')
        axs[2, 3].grid(True)
        axs[2, 3].legend(loc='upper right')

        # X-axis limits
        for row in axs:
            for ax in row:
                ax.set_xlim(lower_X_lim, upper_X_lim)

        plt.tight_layout()
        plt.show()

    except:
        print(f"Cannot plot acceleration data for trial: {trial}")


def plot_Walking_Direction(walking_directions):
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    for trial, angle_deg in walking_directions.items():
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)
    
        # Plot a line for each trial (arbitrary length = 1)
        ax.plot([angle_rad, angle_rad], [0, 1], label=trial, lw=1)
    
    ax.set_theta_zero_location('E')  # 0° = east (X-axis)
    ax.set_theta_direction(-1)       # clockwise
    
    plt.title("Walking Directions for All Trials")
    plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.05))
    plt.show()
    

def plot_AP_GRF_and_AP_ACC(OMCS, OMCS_AP_GRF_left, OMCS_AP_GRF_right, OMCS_AP_ACC_Sacrum, IMU_AP_ACC_Lumbar_BF, trial, lower_X_lim, upper_X_lim):

    try:             
        plt.figure(figsize=(10, 6))
        plt.plot(-OMCS_AP_GRF_left[trial] / OMCS[trial]['body_mass'], label='AP-GRF left')
        plt.plot(-OMCS_AP_GRF_right[trial] / OMCS[trial]['body_mass'], label='AP-GRF right')
        plt.plot(-OMCS_AP_ACC_Sacrum[trial]/1000, label='AP-ACC OMCS')
        plt.plot(-IMU_AP_ACC_Lumbar_BF[trial], label='AP-ACC IMU')
        plt.xlabel('Time (samples)')
        plt.ylabel('Force (N/kg) // Acceleration (m/s^2)')
        plt.title(f"AP-GRF and AP-ACC for Trial: {trial}")
        plt.xlim(lower_X_lim, upper_X_lim)
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()
    except:
        print(f"Cannot plot AP acceleration data for trial: {trial}")


def plot_AP_GRF_and_AP_ACC_with_GaitEvents(OMCS, OMCS_gait_events, OMCS_gait_characteristics, IMU_gait_events, IMU_gait_characteristics, OMCS_AP_GRF_left, OMCS_AP_GRF_right, OMCS_AP_ACC_Sacrum, IMU_AP_ACC_Lumbar_BF, trial, lower_X_lim, upper_X_lim):
    try:
        fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)

        # Helper function to plot gait vertical lines (IC and TC)
        def add_event_lines(ax, events, side_prefix, side_label):
            ic_key = f'Index numbers initial contact {side_prefix}'
            tc_key = f'Index numbers terminal contact {side_prefix}'
            
            if ic_key in events[trial]:
                for ic in events[trial][ic_key]:
                    ax.axvline(x=ic, color='black', linestyle='--', linewidth=1, label=f'IC {side_label}')
            if tc_key in events[trial]:
                for tc in events[trial][tc_key]:
                    ax.axvline(x=tc, color='black', linestyle=':', linewidth=1, label=f'TC {side_label}')

        # Helper function to add shaded intervals
        def shade_intervals(ax, data, key_red, key_green, alpha=0.2):
            if key_red in data[trial] and len(data[trial][key_red]) > 0:
                intervals = np.atleast_2d(data[trial][key_red])
                for start, end in intervals[:, :2]:
                    ax.axvspan(start, end, color="red", alpha=alpha, label="Braking / Deceleration", zorder=0)
                    
            if key_green in data[trial] and len(data[trial][key_green]) > 0:
                intervals = np.atleast_2d(data[trial][key_green])
                for start, end in intervals[:, :2]:
                    ax.axvspan(start, end, color="green", alpha=alpha, label="Propulsion / Acceleration", zorder=0)

        # Helper to clean up legend duplicates
        def deduplicate_legend(ax):
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # -------------------------------------------------------------------
        # ROW 1: AP-GRF (Left vs. Right)
        # -------------------------------------------------------------------
        # Left Leg
        axes[0, 0].plot(-OMCS_AP_GRF_left[trial] / OMCS[trial]['body_mass'], label='AP-GRF Left', color='tab:blue')
        axes[0, 0].set_ylabel('AP-GRF (N/kg)')
        axes[0, 0].set_title('Left Leg - AP Ground Reaction Force')
        add_event_lines(axes[0, 0], OMCS_gait_events, 'left', 'Left')
        shade_intervals(axes[0, 0], OMCS_gait_characteristics, 'Braking left', 'Propulsion left')

        # Right Leg
        axes[0, 1].plot(-OMCS_AP_GRF_right[trial] / OMCS[trial]['body_mass'], label='AP-GRF Right', color='tab:orange')
        axes[0, 1].set_ylabel('AP-GRF (N/kg)')
        axes[0, 1].set_title('Right Leg - AP Ground Reaction Force')
        add_event_lines(axes[0, 1], OMCS_gait_events, 'right', 'Right')
        shade_intervals(axes[0, 1], OMCS_gait_characteristics, 'Braking right', 'Propulsion right')

        # -------------------------------------------------------------------
        # ROW 2: OMCS AP Acceleration (Left vs. Right)
        # -------------------------------------------------------------------
        # Left Leg
        axes[1, 0].plot(-OMCS_AP_ACC_Sacrum[trial] / 1000, label='AP-ACC OMCS', color='tab:green')
        axes[1, 0].set_ylabel('AP Acceleration (m/s²)')
        axes[1, 0].set_title('Left Leg - AP Acceleration (OMCS)')
        add_event_lines(axes[1, 0], OMCS_gait_events, 'left', 'Left')
        shade_intervals(axes[1, 0], OMCS_gait_characteristics, 'AP Deceleration Lumbar left', 'AP Acceleration Lumbar left')

        # Right Leg
        axes[1, 1].plot(-OMCS_AP_ACC_Sacrum[trial] / 1000, label='AP-ACC OMCS', color='tab:green')
        axes[1, 1].set_ylabel('AP Acceleration (m/s²)')
        axes[1, 1].set_title('Right Leg - AP Acceleration (OMCS)')
        add_event_lines(axes[1, 1], OMCS_gait_events, 'right', 'Right')
        shade_intervals(axes[1, 1], OMCS_gait_characteristics, 'AP Deceleration Lumbar right', 'AP Acceleration Lumbar right')

        # -------------------------------------------------------------------
        # ROW 3: IMU AP Acceleration (Left vs. Right)
        # -------------------------------------------------------------------
        # Left Leg
        axes[2, 0].plot(-IMU_AP_ACC_Lumbar_BF[trial], label='AP-ACC IMU', color='tab:red')
        axes[2, 0].set_xlabel('Time (samples)')
        axes[2, 0].set_ylabel('AP Acceleration (m/s²)')
        axes[2, 0].set_title('Left Leg - AP Acceleration (IMU)')
        add_event_lines(axes[2, 0], IMU_gait_events, 'left', 'Left')
        shade_intervals(axes[2, 0], IMU_gait_characteristics, 'AP Deceleration Lumbar left', 'AP Acceleration Lumbar left')

        # Right Leg
        axes[2, 1].plot(-IMU_AP_ACC_Lumbar_BF[trial], label='AP-ACC IMU', color='tab:red')
        axes[2, 1].set_xlabel('Time (samples)')
        axes[2, 1].set_ylabel('AP Acceleration (m/s²)')
        axes[2, 1].set_title('Right Leg - AP Acceleration (IMU)')
        add_event_lines(axes[2, 1], IMU_gait_events, 'right', 'Right')
        shade_intervals(axes[2, 1], IMU_gait_characteristics, 'AP Deceleration Lumbar right', 'AP Acceleration Lumbar right')

        # -------------------------------------------------------------------
        # Formatting Grid, Reference Lines, Legends, and Axes limits
        # -------------------------------------------------------------------
        for row in axes:
            for ax in row:
                ax.axhline(0, color='black', linestyle='--', linewidth=1.2, zorder=4)
                ax.grid(True)
                deduplicate_legend(ax)

        plt.xlim(lower_X_lim, upper_X_lim)
        plt.suptitle(f"AP-GRF and AP-ACC for Trial: {trial}", fontsize=14)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Cannot plot AP-GRF and AP Acceleration data for trial: {trial}. Error: {e}")

