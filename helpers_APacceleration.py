"""
Contains helper functions that are used by main_APacceleration.py.

Version - Author:
    2025: Lars van Rengs - l.vanrengs@maartenskliniek.nl
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
    
    # Define if vicon data is from GRAIL (../GRAIL/..) or overground lab (../GBA/..) trials
    subfolderviconGRAIL = 'Vicon/GRAIL'
    subfolderviconGBA = 'Vicon/GBA'
    subfoldervicon = 'Vicon'
    
    # Define xsens trialnumber with corresponding vicon measurement
    corresponding_files = dict()
    # All files
    files = dict()
    
    # HEALTHY GRAIL TRIALS
    if trialtype['Healthy GRAIL'] == True:
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
                        elif entry.name == '900_V_pp01_FS_SS01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp03_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp03_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp04_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp04_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp05_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp05_SS01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp06_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp06_FS_SS01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp07_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp07_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp08_SP02.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp08_FS_SS02.c3d':
                            xsensnum[entry.name] = '013'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp09_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp09_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp10_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp10_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp11_SP01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp11_FS_SS01.c3d':
                            xsensnum[entry.name] = '013'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp12_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp12_FS_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                                        
                        elif entry.name == '900_V_pp13_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp13_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp14_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp14_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp15_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp15_FS_SS01.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name] 
                        
                        elif entry.name == '900_V_pp16_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp16_FS_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp18_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp18_FS_SS02.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp19_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp19_FS_SS01.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp20_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp20_FS_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp21_SP01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp21_FS_SS01.c3d':
                            xsensnum[entry.name] = '013'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp22_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp22_FS_SS01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
        
        corresponding_files['Healthy GRAIL'] = dict()
        corresponding_files['Healthy GRAIL']['xsensnum'] = xsensnum
        corresponding_files['Healthy GRAIL']['xsensfilepaths'] = xsensfilepaths
        
    # CVA GRAIL TRIALS
    if trialtype['CVA GRAIL'] == True:
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
                        elif entry.name == '900_CVA_01_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_pp02_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_02' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_pp02_FS_SS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_02' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_03_FS01.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_03_FS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_03' in item][0]+xsensnum[entry.name]
                    
                        elif entry.name == '900_CVA_04_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_04_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_CVA_05_SP01.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_05_FS_SS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_06_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_06_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_06' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_07_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_07_FS_SS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_08_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_08_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_09_FS01.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_09_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_10_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_10_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_10' in item][0]+xsensnum[entry.name]
        corresponding_files['CVA GRAIL'] = dict()
        corresponding_files['CVA GRAIL']['xsensnum'] = xsensnum
        corresponding_files['CVA GRAIL']['xsensfilepaths'] = xsensfilepaths
    
    # CVA_FEEDBACK TRIALS
    if trialtype['CVA_feedback GRAIL'] == True:
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
                        elif entry.name == '1019_MR001_FBIC.c3d':
                            xsensnum[entry.name] = '006'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp01' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR001_FBPO.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp01' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR001_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp01' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR001_2Reg.c3d':
                            xsensnum[entry.name] = '007'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR002_Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp02' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR002_FBIC.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp02' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR002_FBPO.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp02' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR002_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp02' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR002_2Reg.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp02' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR003_1Reg02.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR003_FBIC.c3d':
                            xsensnum[entry.name] = ''  # Xsens recording error
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR003_FBPO.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR003_2FB.c3d':
                            xsensnum[entry.name] = '005'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR003_2Reg.c3d':
                            xsensnum[entry.name] = '006'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp03' in item][0]+xsensnum[entry.name]
                    
                        elif entry.name == '1019_MR004_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR004_FBIC.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR004_FBPO.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR004_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR004_2Reg02.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '1019_MR005_1Reg01.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR005_FBIC.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR005_FBPO.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR005_2FB.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR005_2Reg.c3d':
                            xsensnum[entry.name] = '005'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR006_1Reg.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR006_FBIC.c3d':
                            xsensnum[entry.name] = '002'  # Vicon data not of good quality; no sufficient gold-standard
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR006_FBPO.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR006_2FB.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR006_2Reg02.c3d':
                            xsensnum[entry.name] = '' # Xsens data; recording error
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp06' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR007_1Reg02.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR007_FBIC.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR007_FBPO.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR007_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR007_2Reg.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR008_1Reg02.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR008_FBIC.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR008_FBPO.c3d':
                            xsensnum[entry.name] = '005'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR008_2FB.c3d':
                            xsensnum[entry.name] = '006'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR008_2Reg.c3d':
                            xsensnum[entry.name] = '007'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR009_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR009_FBIC.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR009_FBPO.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR009_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR009_2Reg.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '1019_MR010_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR010_FBIC.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR010_FBPO.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR010_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR010_2Reg.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp10' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '1019_MR011_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp11' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR011_FBIC.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp11' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR011_FBPO.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp11' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR011_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp11' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR011_2Reg.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp11' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '1019_MR012_1Reg.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp12' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR012_FBIC.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp12' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR012_FBPO.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp12' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR012_2FB.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp12' in item][0]+xsensnum[entry.name]
                        elif entry.name == '1019_MR012_2Reg.c3d':
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '1019_pp12' in item][0]+xsensnum[entry.name]
        corresponding_files['CVA_feedback GRAIL'] = dict()
        corresponding_files['CVA_feedback GRAIL']['xsensnum'] = xsensnum
        corresponding_files['CVA_feedback GRAIL']['xsensfilepaths'] = xsensfilepaths

    # HEALTHY LAB TRIALS
    if trialtype['Healthy Lab'] == True:
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
            ppfoldersvicon.append(ppfolders[i] + '/' + subfolderviconGBA) #+ '/' + date[0] 
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) #+ '/' + date[0]
            
        # files=dict()
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            try:
                with os.scandir(ppfoldersvicon[i]) as it:        
                    for entry in it:
                        if not entry.name.startswith('.') and entry.is_file():
                            files[entry.name] = (ppfoldersvicon[i]+'/'+entry.name)
                            
                            # Define xsens exports
                            if entry.name == '900_V_pp01_2MWT01.c3d':
                                xsensnum[entry.name] = '004'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_01' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp03_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp04_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp05_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp06_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp07_SW03.c3d':
                                xsensnum[entry.name] = '003'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp08_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp09_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp10_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp11_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp12_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp13_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp14_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp15_SW01.c3d':
                                xsensnum[entry.name] = '006'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp16_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp18_SW01.c3d':
                                xsensnum[entry.name] = '005'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp19_SW01.c3d':
                                xsensnum[entry.name] = '004'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp20_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp21_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp22_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
                           
            except FileNotFoundError:
                xsensnum[ppfolders[i]] = 'Unavailable'
                xsensfilepaths[ppfolders[i]] = 'Unavailable'
        
        corresponding_files['Healthy Lab'] = dict()
        corresponding_files['Healthy Lab']['xsensnum'] = xsensnum
        corresponding_files['Healthy Lab']['xsensfilepaths'] = xsensfilepaths
    
    # 4 sets
    if trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    # 3 sets
    if trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths']}
    if trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    if trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    if trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    # 2 sets
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['Healthy Lab']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = {**corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['CVA GRAIL']['xsensfilepaths'], **corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']}
    # 1 set
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = corresponding_files['Healthy GRAIL']['xsensfilepaths']
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = corresponding_files['Healthy Lab']['xsensfilepaths']
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == True and trialtype['CVA_feedback GRAIL'] == False:
        xsensfilepaths = corresponding_files['CVA GRAIL']['xsensfilepaths']
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == False and trialtype['CVA_feedback GRAIL'] == True:
        xsensfilepaths = corresponding_files['CVA_feedback GRAIL']['xsensfilepaths']
    
    # Sort files on task
    filesGRAIL = dict()
    filesREG = dict()
    filesIRREG = dict()
    filesGBA = dict()
    filesSW = dict() # Straight ahead walking in measurement volume
    filesFB = dict() # Feedback study
    
    removekeys=[]
    for key in files:
        if ('V_pp02' or 'V_pp17' or '900_V_pp11_LT03') in key: # exclusion of these test persons
            removekeys.append(key)
    for key in removekeys:
        files.pop(key)

    for key in xsensfilepaths:
        # GRAIL trials
        if '_FS0' in key:
            filesREG[key] = files[key]
            filesGRAIL[key] = files[key]
        if '_SP0' in key:
            if key == '900_V_pp01_SP03.c3d': # Fixed speed trial, accidentally wrongly named
                pass
            else:
                filesREG[key] = files[key]
                filesGRAIL[key] = files[key]
        if '_SS' in key:
            filesIRREG[key] = files[key]
            filesGRAIL[key] = files[key]
        if '1019_MR' in key:
            filesFB[key] = files[key]
            filesGRAIL[key] = files[key]
        # Overground trials
        if '_SW' in key:
            filesSW[key] = files[key]
            filesGBA[key] = files[key]
        if '_2MWT' in key:
            filesSW[key] = files[key]
            filesGBA[key] = files[key]
    
    # Set trialnames to be analyzed
    trialnames = list()
    if trialtype['Healthy GRAIL'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '_V_' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['CVA GRAIL'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '_CVA_' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['CVA_feedback GRAIL'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '1019_MR' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['Healthy Lab'] == True:
        # trialnames.extend( [string for string in list(filesGBA.keys()) if '_V_' in string] )
        trialnames.extend(list(filesSW.keys()))
    
    trialnames = list(set(trialnames))
    trialnames.remove('900_CVA_03_FS02.c3d') # This person performed 2 regular walking trials, remove one for further analysis
    
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
            
            # Two trials with some part 'flickering' markers; set these time periods to missing markerdata
            # if trialnames[trial] == '900_V_pp12_FS01.c3d': # no data labeling (bad dataquality)
            #     for key in dataviconfilt:
            #         dataviconfilt[key][5522:5651,:] = 0
            if trialnames[trial] == '900_V_pp21_FS_SS01.c3d': # no data labeling (bad dataquality)
                for key in dataviconfilt:
                    dataviconfilt[key][10800:10855,:] = 0
                    
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
            # CVA_feedback
            if trialnames[trial] == '1019_MR001_1Reg.c3d' or trialnames[trial] == '1019_MR001_FBIC.c3d' or trialnames[trial] == '1019_MR001_FBPO.c3d' or trialnames[trial] == '1019_MR001_2FB.c3d' or trialnames[trial] == '1019_MR001_2Reg.c3d':
                gender = 'M'
                body_mass = 122.0
                height = 1775
                affected_leg = 'left'
            if trialnames[trial] == '1019_MR002_Reg.c3d' or trialnames[trial] == '1019_MR002_FBIC.c3d' or trialnames[trial] == '1019_MR002_FBPO.c3d' or trialnames[trial] == '1019_MR002_2FB.c3d' or trialnames[trial] == '1019_MR002_2Reg.c3d':
                gender = 'F'
                body_mass = 68.0
                height = 1710
                affected_leg = 'right'
            if trialnames[trial] == '1019_MR003_1Reg02.c3d' or trialnames[trial] == '1019_MR003_FBIC.c3d' or trialnames[trial] == '1019_MR003_FBPO.c3d' or trialnames[trial] == '1019_MR003_2FB.c3d' or trialnames[trial] == '1019_MR003_2Reg.c3d':
                gender = 'M'
                body_mass = 75.0
                height = 1720
                affected_leg = 'right'
            if trialnames[trial] == '1019_MR004_1Reg.c3d' or trialnames[trial] == '1019_MR004_FBIC.c3d' or trialnames[trial] == '1019_MR004_FBPO.c3d' or trialnames[trial] == '1019_MR004_2FB.c3d' or trialnames[trial] == '1019_MR004_2Reg02.c3d':
                gender = 'F'
                body_mass = 70.0
                height = 1630
                affected_leg = 'left'
            if trialnames[trial] == '1019_MR005_1Reg01.c3d' or trialnames[trial] == '1019_MR005_FBIC.c3d' or trialnames[trial] == '1019_MR005_FBPO.c3d' or trialnames[trial] == '1019_MR005_2FB.c3d' or trialnames[trial] == '1019_MR005_2Reg.c3d':
                gender = 'M'
                body_mass = 80.0
                height = 1830
                affected_leg = 'left'
            if trialnames[trial] == '1019_MR006_1Reg.c3d' or trialnames[trial] == '1019_MR006_FBPO.c3d' or trialnames[trial] == '1019_MR006_2FB.c3d' or trialnames[trial] == '1019_MR006_2Reg02.c3d':
                gender = 'M'
                body_mass = 85.0
                height = 1940
                affected_leg = 'left'
            if trialnames[trial] == '1019_MR007_1Reg02.c3d' or trialnames[trial] == '1019_MR007_FBIC.c3d' or trialnames[trial] == '1019_MR007_FBPO.c3d' or trialnames[trial] == '1019_MR007_2FB.c3d' or trialnames[trial] == '1019_MR007_2Reg.c3d':
                gender = 'M'
                body_mass = 90.0
                height = 1830
                affected_leg = 'right'
            if trialnames[trial] == '1019_MR008_1Reg02.c3d' or trialnames[trial] == '1019_MR008_FBIC.c3d' or trialnames[trial] == '1019_MR008_FBPO.c3d' or trialnames[trial] == '1019_MR008_2FB.c3d' or trialnames[trial] == '1019_MR008_2Reg.c3d':
                gender = 'F'
                body_mass = 91.0
                height = 1720
                affected_leg = 'right'
            if trialnames[trial] == '1019_MR009_1Reg.c3d' or trialnames[trial] == '1019_MR009_FBIC.c3d' or trialnames[trial] == '1019_MR009_FBPO.c3d' or trialnames[trial] == '1019_MR009_2FB.c3d' or trialnames[trial] == '1019_MR009_2Reg.c3d':
                gender = 'M'
                body_mass = 74.0
                height = 1710
                affected_leg = 'right'
            if trialnames[trial] == '1019_MR010_1Reg.c3d' or trialnames[trial] == '1019_MR010_FBIC.c3d' or trialnames[trial] == '1019_MR010_FBPO.c3d' or trialnames[trial] == '1019_MR010_2FB.c3d' or trialnames[trial] == '1019_MR010_2Reg.c3d':
                gender = 'M'
                body_mass = 82.0
                height = 1880
                affected_leg = 'right'
            if trialnames[trial] == '1019_MR011_1Reg.c3d' or trialnames[trial] == '1019_MR011_FBIC.c3d' or trialnames[trial] == '1019_MR011_FBPO.c3d' or trialnames[trial] == '1019_MR011_2FB.c3d' or trialnames[trial] == '1019_MR011_2Reg.c3d':
                gender = 'F'
                body_mass = 104.0
                height = 1700
                affected_leg = 'left'
            if trialnames[trial] == '1019_MR012_1Reg.c3d' or trialnames[trial] == '1019_MR012_FBIC.c3d' or trialnames[trial] == '1019_MR012_FBPO.c3d' or trialnames[trial] == '1019_MR012_2FB.c3d' or trialnames[trial] == '1019_MR012_2Reg.c3d':
                gender = 'F'
                body_mass = 79.0
                height = 1720
                affected_leg = 'left'
            # CVA
            if trialnames[trial] == '900_CVA_01_FS_SS01.c3d' or trialnames[trial] == '900_CVA_01_SP01.c3d':
                gender = 'F'
                body_mass = 70.0
                height = 1620
                affected_leg = 'right'
            if trialnames[trial] == '900_CVA_pp02_FS_SS02.c3d' or trialnames[trial] == '900_CVA_pp02_SP01.c3d':
                gender = 'M'
                body_mass = 82.0
                height = 1830
                affected_leg = 'left'
            if trialnames[trial] == '900_CVA_03_FS01.c3d' or trialnames[trial] == '900_CVA_03_FS02.c3d':
                gender = 'M'
                body_mass = 71.0
                height = 1780
                affected_leg = 'left'
            if trialnames[trial] == '900_CVA_04_FS_SS01.c3d' or trialnames[trial] == '900_CVA_04_SP01.c3d':
                gender = 'M'
                body_mass = 93.0
                height = 1810
                affected_leg = 'left'
            if trialnames[trial] == '900_CVA_05_FS_SS02.c3d' or trialnames[trial] == '900_CVA_05_SP01.c3d':
                gender = 'M'
                body_mass = 91.0
                height = 1710
                affected_leg = 'left'
            if trialnames[trial] == '900_CVA_06_FS_SS01.c3d' or trialnames[trial] == '900_CVA_06_SP01.c3d':
                gender = 'F'
                body_mass = 71.0
                height = 1760
                affected_leg = 'right'
            if trialnames[trial] == '900_CVA_07_FS_SS02.c3d' or trialnames[trial] == '900_CVA_07_SP01.c3d':
                gender = 'M'
                body_mass = 95.4
                height = 1840
                affected_leg = 'right'
            if trialnames[trial] == '900_CVA_08_FS_SS01.c3d' or trialnames[trial] == '900_CVA_08_SP01.c3d':
                gender = 'M'
                body_mass = 85.0
                height = 1840
                affected_leg = 'right'
            if trialnames[trial] == '900_CVA_09_FS_SS01.c3d' or trialnames[trial] == '900_CVA_09_FS01.c3d':
                gender = 'M'
                body_mass = 76.0
                height = 1800
                affected_leg = 'right'
            if trialnames[trial] == '900_CVA_10_FS_SS01.c3d' or trialnames[trial] == '900_CVA_10_SP01.c3d':
                gender = 'F'
                body_mass = 77.0
                height = 1650
                affected_leg = 'right'
            # Healthy controls
            if trialnames[trial] == '900_V_pp01_FS_SS01.c3d' or trialnames[trial] == '900_V_pp01_SP01.c3d' or trialnames[trial] == '900_V_pp01_SP03.c3d':
                gender = 'F'
                body_mass = 72.0
                height = 1680
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp03_FS_SS01.c3d' or trialnames[trial] == '900_V_pp03_FS01.c3d' or trialnames[trial] == '900_V_pp03_SP01.c3d':
                gender = 'F'
                body_mass = 74.8
                height = 1640
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp04_FS01.c3d' or trialnames[trial] == '900_V_pp04_SP01.c3d' or trialnames[trial] == '900_V_pp04_SS01.c3d':
                gender = 'M'
                body_mass = 76.8
                height = 1660
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp05_FS01.c3d' or trialnames[trial] == '900_V_pp05_SP01.c3d' or trialnames[trial] == '900_V_pp05_SS01.c3d':
                gender = 'F'
                body_mass = 67.8
                height = 1650
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp06_FS_SS01.c3d' or trialnames[trial] == '900_V_pp06_FS01.c3d' or trialnames[trial] == '900_V_pp06_SP01.c3d':
                gender = 'M'
                body_mass = 77.2
                height = 1830
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp07_FS_SS01.c3d' or trialnames[trial] == '900_V_pp07_FS01.c3d' or trialnames[trial] == '900_V_pp07_SP01.c3d':
                gender = 'F'
                body_mass = 62.4
                height = 1730
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp08_FS_SS02.c3d' or trialnames[trial] == '900_V_pp08_FS01.c3d' or trialnames[trial] == '900_V_pp08_SP02.c3d':
                gender = 'F'
                body_mass = 63.6
                height = 1680
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp09_FS_SS01.c3d' or trialnames[trial] == '900_V_pp09_FS01.c3d' or trialnames[trial] == '900_V_pp09_SP01.c3d':
                gender = 'M'
                body_mass = 69.0
                height = 1790
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp10_FS01.c3d' or trialnames[trial] == '900_V_pp10_SP01.c3d' or trialnames[trial] == '900_V_pp10_SS01.c3d':
                gender = 'M'
                body_mass = 93.0
                height = 1860
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp11_FS_SS01.c3d' or trialnames[trial] == '900_V_pp11_FS01.c3d' or trialnames[trial] == '900_V_pp11_SP01.c3d':
                gender = 'M'
                body_mass = 77.6
                height = 1810
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp12_FS_SS01.c3d' or trialnames[trial] == '900_V_pp12_FS01.c3d' or trialnames[trial] == '900_V_pp12_SP01.c3d':
                gender = 'F'
                body_mass = 78.2
                height = 1800
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp13_FS_SS01.c3d' or trialnames[trial] == '900_V_pp13_FS02.c3d' or trialnames[trial] == '900_V_pp13_SP01.c3d':
                gender = 'M'
                body_mass = 88.6
                height = 1800
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp14_FS_SS01.c3d' or trialnames[trial] == '900_V_pp14_FS01.c3d' or trialnames[trial] == '900_V_pp14_SP01.c3d':
                gender = 'F'
                body_mass = 68.4
                height = 1700
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp15_FS_SS01.c3d' or trialnames[trial] == '900_V_pp15_FS01.c3d' or trialnames[trial] == '900_V_pp15_SP01.c3d':
                gender = 'F'
                body_mass = 66.2
                height = 1620
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp16_FS_SS01.c3d' or trialnames[trial] == '900_V_pp16_FS01.c3d' or trialnames[trial] == '900_V_pp16_SP01.c3d':
                gender = 'F'
                body_mass = 70.4
                height = 1660
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp18_FS_SS02.c3d' or trialnames[trial] == '900_V_pp18_FS01.c3d' or trialnames[trial] == '900_V_pp18_SP01.c3d':
                gender = 'M'
                body_mass = 77.0
                height = 1820
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp19_FS_SS01.c3d' or trialnames[trial] == '900_V_pp19_FS01.c3d' or trialnames[trial] == '900_V_pp19_SP01.c3d':
                gender = 'F'
                body_mass = 70.0
                height = 1740
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp20_FS_SS01.c3d' or trialnames[trial] == '900_V_pp20_FS01.c3d' or trialnames[trial] == '900_V_pp20_SP01.c3d':
                gender = 'M'
                body_mass = 76.8
                height = 1800
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp21_FS_SS01.c3d' or trialnames[trial] == '900_V_pp21_FS01.c3d' or trialnames[trial] == '900_V_pp21_SP01.c3d':
                gender = 'M'
                body_mass = 89.2
                height = 1790
                affected_leg = 'none'
            if trialnames[trial] == '900_V_pp22_FS_SS01.c3d' or trialnames[trial] == '900_V_pp22_FS01.c3d' or trialnames[trial] == '900_V_pp22_SP01.c3d':
                gender = 'M'
                body_mass = 73.4
                height = 1760
                affected_leg = 'none'        
            
                   
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
                
                # Determine trialType based on foldername or kwargs item
                if 'L-test' in xsensfilepaths[trialnames[trial]]:
                    data_dict['trialType'] = 'L-test'
                elif '2-minuten looptest' in xsensfilepaths[trialnames[trial]]:
                    data_dict['trialType'] = '2MWT'
                elif trialnames[trial] in filesSW.keys():
                    data_dict['trialType'] = '2MWT'
                else:
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
                    data_dict['trialType'] = 'Feedback treadmill'
                
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
        if IMU[f]['trialType'] == 'GRAIL' or IMU[f]['trialType'] == 'Feedback treadmill':
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
                            
        if IMU[f]['trialType'] == '2MWT' or IMU[f]['trialType'] == 'L-test':
            # Detect gait events overground vicon data
            OMCS_gait_events[f] = gaiteventdetection(OMCS[f], OMCS[f]['Sample frequency marker data'], algorithmtype ='velocity', trialtype='overground')
            
            # Determine spatiotemporal parameters from vicon data
            try:
                OMCS_gait_characteristics[f] = spatiotemporals(OMCS[f], OMCS_gait_events[f], sample_frequency = OMCS[f]['Sample frequency marker data'], trialtype='overground')
            except:
                print('Cannot calculate OMCS based spatiotemporals for trial ', f)
    
    return OMCS, OMCS_gait_events, OMCS_gait_characteristics



def OMCS_calculate_sacrum_acceleration(OMCS):

    # Prepare datastructure
    sacrum = dict()
    OMCS_POS_Sacrum = dict()
    OMCS_ACC_Sacrum = dict()

    for f in OMCS:
        # Create a dictionary for this trial
        sacrum[f] = dict()
        OMCS_POS_Sacrum[f] = dict()
        OMCS_ACC_Sacrum[f] = dict()

        try:
            markerdata = OMCS[f]
            fs_markerdata = OMCS[f]['Sample frequency marker data']

            # Set second-order low-pass butterworth filter;
            # Cut-off frequency: 15Hz
            fc = 15  # Cut-off frequency of the filter
            omega = fc / (fs_markerdata / 2) # Normalize the frequency
            N = 2 # Order of the butterworth filter
            filter_type = 'lowpass' # Type of the filter
            b, a = signal.butter(N, omega, filter_type)   
                 
            # Define markers to process
            keys_to_modify = {"LANK", "LASI", "LHEE", "LPSI", "LTOE", 
                              "RANK", "RASI", "RHEE", "RPSI", "RTOE"}

            # Filtering specified markers
            for key in keys_to_modify.intersection(markerdata.keys()):               
                markerdataX = filtfilt_omitNaN(b, a, markerdata[key][:,0])
                markerdataY = filtfilt_omitNaN(b, a, markerdata[key][:,1])
                markerdataZ = filtfilt_omitNaN(b, a, markerdata[key][:,2])
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

            # Calculate velocity
            sacrum_vel = np.append(np.array([[np.nan, np.nan, np.nan]]), np.diff(sacrum_pos, axis=0), axis=0) * fs_markerdata    

            # Calculate acceleration
            sacrum_acc = np.append(np.array([[np.nan, np.nan, np.nan]]), np.diff(sacrum_vel, axis=0), axis=0) * fs_markerdata     
            sacrum[f]['Acceleration Sacrum X'] = sacrum_acc[:,0]
            sacrum[f]['Acceleration Sacrum Y'] = sacrum_acc[:,1]
            sacrum[f]['Acceleration Sacrum Z'] = sacrum_acc[:,2]

            OMCS_ACC_Sacrum[f] = np.column_stack((sacrum[f]['Acceleration Sacrum X'], sacrum[f]['Acceleration Sacrum Y'], sacrum[f]['Acceleration Sacrum Z'])) 
            
        except:
            print('Cannot calculate OMCS based acceleration for trial ', f) 

    return OMCS_POS_Sacrum, OMCS_ACC_Sacrum



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



def filtfilt_omitNaN(b, a, data):

    # Step 1: Remove NaNs temporarily and apply the filter
    data_filtered_without_nan = signal.filtfilt(b, a, data[~np.isnan(data)])
   
    
    # Step 2: Reintroduce NaNs at the original positions
    # Create a new array to store the filtered values with NaNs reintroduced
    data_filtered = np.full_like(data, np.nan)
    
    # Assign the filtered data to the valid indices
    data_filtered[~np.isnan(data)] = data_filtered_without_nan

    
    # Step 3: Extend NaNs to ±10 points
    nan_indices = np.where(np.isnan(data))[0]  # Get indices of NaNs
    for idx in nan_indices:
        data_filtered[max(0, idx-10) : min(len(data), idx+11)] = np.nan  # Set ±10 points to NaN

    return data_filtered



def APaccelerationLumbar(gaitevents, gaitcharacteristics, APacceleration, **kwargs):
    """
    Calculate Peak + Impulse values of Acceleration Sacrum/Lumbar
    """
    # Function was based on:
    # Deffeyes, J. E., & Peters, D. M. (2021). Time-integrated propulsive and braking impulses do not depend on walking speed. Gait & posture, 88, 258-263.
    # DOI: https://doi.org/10.1016/j.gaitpost.2021.06.012
    
    # Set defaults
    sample_frequency = 100 # Sample frequency of the marker data
    bodyweight = 1
    th_crossings = 0 # Set threshold_crossings at 0 m/s^2 to identify crossings in acceleration in AP direction
    th_crosssteps = -10 * 0.90 * bodyweight # Set threshold_crosssteps at 10 times 90% of the bodyweight to identify cross steps and deem artefact
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
    # Zeni, J. A., Jr, Richards, J. G., & Higginson, J. S. (2008).
    # Two simple methods for determining gait events during treadmill and overground walking using kinematic data.
    # Gait & posture, 27(4), 710–714. https://doi.org/10.1016/j.gaitpost.2007.07.007
    # Fourth-order low-pass butterworth filter;
    # Cut-off frequency: 20Hz
    fc = 20  # Cut-off frequency of the filter
    omega = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter (scipy.signal.filtfilt is a forwrd-backward linear filter meaning the Nth-order*2 is applied)
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
       
    APacceleration_filtered = filtfilt_omitNaN(b, a, APacceleration)
    
    # Very low pass filter for first segmentation of stance in braking and propulsion areas
    fc = 5  # Cut-off frequency of the filter
    omega = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter (scipy.signal.filtfilt is a forwrd-backward linear filter meaning the Nth-order*2 is applied)
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
    
    sacrum_acc_y = filtfilt_omitNaN(b, a, APacceleration)
               
    # First determine stance phase from IC till TC according to vicon data,
    # Deem cross steps as faulty stance phases to calculate propulsive force,
    # Then find the local minimum,
    # Last find zero crossing around local minima as start and stop of propulsion.
    
    # Left side
    gaitcharacteristics['Stance left index numbers'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Sacrum left start'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Sacrum left stop'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Sacrum left start'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Sacrum left stop'] = np.array([], dtype=int)
        
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
            
            # Find local maximum peak in AP Acceleration Sacrum (= braking force)
            maxpeaks = signal.find_peaks(sacrum_acc_y[start+5:stop-10])[0] + start+5
            if len(maxpeaks)>0:
                localmax = np.argmax(sacrum_acc_y[maxpeaks])
                localmax = int(maxpeaks[localmax])
                if sacrum_acc_y[localmax] < th_crossings: # all data is negative and thus propulsion, (no braking was generated)
                    localmax = False
            else: # no braking peaks
                localmax = False
            
            # Find local minimum peak in AP Acceleration Sacrum (= forward force) after the maximum braking force
            minpeaks = signal.find_peaks(-sacrum_acc_y[start+10:stop-5])[0] + start+10
            if type(localmax) == int:
                minpeaks = minpeaks[minpeaks>localmax]
            elif localmax == False:
                minpeaks = minpeaks[minpeaks>start]
            if len(minpeaks)>0:
                localmin = np.argmin(sacrum_acc_y[minpeaks])
                localmin = int(minpeaks[localmin])
                if sacrum_acc_y[localmin] > th_crossings: # all data is positive and thus braking, (no propulsion was generated)
                        localmin = False
            else: # no propulsion peaks
                localmin = False

            
            # Find approximate braking to propulsion point at first positive to negative zero crossing in highly filtered signal
            if type(localmin) == int and type(localmax) == int: # both braking and propulsion
                braking_to_propulsion = np.argwhere(sacrum_acc_y[localmax:localmin] < th_crossings) +localmax
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
                # gaitevents['AP Deceleration Sacrum left stop'] = np.append(gaitevents['AP Deceleration Sacrum left stop'], true_braking_to_propulsion)
                # gaitevents['AP Acceleration Sacrum left start'] = np.append(gaitevents['AP Acceleration Sacrum left start'], true_braking_to_propulsion)
            
            elif type(braking_to_propulsion) == bool: # no braking-to-propulsion transition
                if type(localmin) == int and localmax == False: # No braking
                    true_braking_to_propulsion = int(start)
                    # gaitevents['AP Deceleration Sacrum left stop'] = np.append(gaitevents['AP Deceleration Sacrum left stop'], start)
                    # gaitevents['AP Acceleration Sacrum left start'] = np.append(gaitevents['AP Acceleration Sacrum left start'], start)
                elif localmin == False and type(localmax) == int: # No propulsion
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Sacrum left stop'] = np.append(gaitevents['AP Deceleration Sacrum left stop'], stop)
                    # gaitevents['AP Acceleration Sacrum left start'] = np.append(gaitevents['AP Acceleration Sacrum left start'], stop)
                elif type(localmin) == int and type(localmax) == int:
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Sacrum left stop'] = np.append(gaitevents['AP Deceleration Sacrum left stop'], stop)
                    # gaitevents['AP Acceleration Sacrum left start'] = np.append(gaitevents['AP Acceleration Sacrum left start'], stop)
                    
            # Find approximate start of braking at "almost zero-crossing" in highly filtered signal    
            if type(localmax) == int:
                signs = np.sign(((sacrum_acc_y/bodyweight)-0.01)[start-10 : localmax])
                crossings = np.argwhere(np.diff(signs)>1) + int(start-10)
                if len(crossings) > 0:
                    start_brake = crossings[-1]
                else:
                    start_brake = np.argmin(((sacrum_acc_y/bodyweight)-0.01)[start : localmax]) + start
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
                
            # gaitevents['AP Deceleration Sacrum left start'] = np.append(gaitevents['AP Deceleration Sacrum left start'], true_start_brake)
            
            # Find approximate stop of propulsion at "almost zero-crossing" in highly filtered signal
            if type(localmin) == int:
                signs = np.sign(((sacrum_acc_y/bodyweight)+0.01)[localmin : stop +10])
                crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positve direction
                if len(crossings) > 0:
                    stop_prop = crossings[-1]
                else:
                    stop_prop = np.argmax(((sacrum_acc_y/bodyweight)+0.01)[localmin:stop]) + localmin
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

            # gaitevents['AP Acceleration Sacrum left stop'] = np.append(gaitevents['AP Acceleration Sacrum left stop'], true_stop_prop)
            
            if (true_braking_to_propulsion is not None and
                true_start_brake is not None and
                true_stop_prop is not None and
                isinstance(true_braking_to_propulsion, int) and
                isinstance(true_start_brake, int) and
                isinstance(true_stop_prop, int) and
                true_start_brake < true_braking_to_propulsion < true_stop_prop):
            
                gaitevents['AP Deceleration Sacrum left start'] = np.append(
                    gaitevents['AP Deceleration Sacrum left start'], true_start_brake)
                gaitevents['AP Deceleration Sacrum left stop'] = np.append(
                    gaitevents['AP Deceleration Sacrum left stop'], true_braking_to_propulsion)
            
                gaitevents['AP Acceleration Sacrum left start'] = np.append(
                    gaitevents['AP Acceleration Sacrum left start'], true_braking_to_propulsion)
                gaitevents['AP Acceleration Sacrum left stop'] = np.append(
                    gaitevents['AP Acceleration Sacrum left stop'], true_stop_prop)

                if debugplot == True:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(start+5, stop-10), sacrum_acc_y[start+5:stop-10], label='Acceleration', color='black')
                    plt.plot(localmax, sacrum_acc_y[localmax], 'r*', label='Local Max')  # Red stars
                    plt.plot(localmin, sacrum_acc_y[localmin], 'g*', label='Local Min')  # Green stars
                    # plt.plot(braking_to_propulsion, sacrum_acc_y[braking_to_propulsion], 'bs', label='Braking → Propulsion')  # Blue squares
                    plt.plot(true_braking_to_propulsion, sacrum_acc_y[true_braking_to_propulsion], 'bs', label='True Braking → Propulsion')  # Blue squares
                    # plt.plot(int(start_brake), sacrum_acc_y[start_brake], 'mo', label='Start Brake')  # Magenta circles
                    plt.plot(true_start_brake, sacrum_acc_y[true_start_brake], 'mo', label='True Start Brake')  # Yellow circles
                    # plt.plot(int(stop_prop), sacrum_acc_y[stop_prop], 'co', label='Stop Propulsion')  # Cyan circles
                    plt.plot(true_stop_prop, sacrum_acc_y[true_stop_prop], 'co', label='True Stop Propulsion')  # Black circles
                    plt.grid(True)
                    plt.legend(loc='upper right', fontsize='small', frameon=True)
                    plt.title('Stance Phase with Gait Events - left')
                    plt.xlabel('Time (samples)')
                    plt.ylabel('Acceleration (sacrum y-axis)')
                    plt.tight_layout()
                    plt.show()

            else:
                # print(f"Skipping stance {i}: invalid brake/propulsion pair.")
                continue
            
        except:
            pass
                
    # Right side
    gaitcharacteristics['Stance right index numbers'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Sacrum right start'] = np.array([], dtype=int)
    gaitevents['AP Acceleration Sacrum right stop'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Sacrum right start'] = np.array([], dtype=int)
    gaitevents['AP Deceleration Sacrum right stop'] = np.array([], dtype=int)
    
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
            maxpeaks = signal.find_peaks(sacrum_acc_y[start+5:stop-10])[0] + start+5
            if len(maxpeaks)>0:
                localmax = np.argmax(sacrum_acc_y[maxpeaks])
                localmax = int(maxpeaks[localmax])
                if sacrum_acc_y[localmax] < th_crossings: # all data is negative and thus propulsion, (no braking force was generated)
                    localmax = False
            else: # no braking peaks
                localmax = False
            
            # Find local minimum peak in strongly filtered Y force (= forward force) after the maximum braking force
            minpeaks = signal.find_peaks(-sacrum_acc_y[start+10:stop-5])[0] + start+10
            if type(localmax) == int:
                minpeaks = minpeaks[minpeaks>localmax]
            elif localmax == False:
                minpeaks = minpeaks[minpeaks>start]
            if len(minpeaks)>0:
                localmin = np.argmin(sacrum_acc_y[minpeaks])
                localmin = int(minpeaks[localmin])
                if sacrum_acc_y[localmin] > th_crossings: # all data is positive and thus braking, (no propulsive forcef was generated)
                        localmin = False
            else: # no propulsion peaks
                localmin = False

            
            # Find approximate braking to propulsion point at first positive to negative zero crossing in highly filtered signal
            if type(localmin) == int and type(localmax) == int: # both braking and propulsion
                braking_to_propulsion = np.argwhere(sacrum_acc_y[localmax:localmin] < th_crossings) +localmax
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
                # gaitevents['AP Deceleration Sacrum right stop'] = np.append(gaitevents['AP Deceleration Sacrum right stop'], true_braking_to_propulsion)
                # gaitevents['AP Acceleration Sacrum right start'] = np.append(gaitevents['AP Acceleration Sacrum right start'], true_braking_to_propulsion)
            
            elif type(braking_to_propulsion) == bool: # no braking-to-propulsion transition
                if type(localmin) == int and localmax == False: # No braking
                    true_braking_to_propulsion = int(start)
                    # gaitevents['AP Deceleration Sacrum right stop'] = np.append(gaitevents['AP Deceleration Sacrum right stop'], start)
                    # gaitevents['AP Acceleration Sacrum right start'] = np.append(gaitevents['AP Acceleration Sacrum right start'], start)
                elif localmin == False and type(localmax) == int: # No propulsion
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Sacrum right stop'] = np.append(gaitevents['AP Deceleration Sacrum right stop'], stop)
                    # gaitevents['AP Acceleration Sacrum right start'] = np.append(gaitevents['AP Acceleration Sacrum right start'], stop)
                elif type(localmin) == int and type(localmax) == int:
                    true_braking_to_propulsion = int(stop)
                    # gaitevents['AP Deceleration Sacrum right stop'] = np.append(gaitevents['AP Deceleration Sacrum right stop'], stop)
                    # gaitevents['AP Acceleration Sacrum right start'] = np.append(gaitevents['AP Acceleration Sacrum right start'], stop)
                    
            # Find approximate start of braking at "almost zero-crossing" in highly filtered signal    
            if type(localmax) == int:
                signs = np.sign(((sacrum_acc_y/bodyweight)-0.01)[start-10 : localmax])
                crossings = np.argwhere(np.diff(signs)>1) + int(start-10)
                if len(crossings) > 0:
                    start_brake = crossings[-1]
                else:
                    start_brake = np.argmin(((sacrum_acc_y/bodyweight)-0.01)[start : localmax]) + start
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
                
            # gaitevents['AP Deceleration Sacrum right start'] = np.append(gaitevents['AP Deceleration Sacrum right start'], true_start_brake)
            
            # Find approximate stop of propulsion at "almost zero-crossing" in highly filtered signal
            if type(localmin) == int:
                signs = np.sign(((sacrum_acc_y/bodyweight)+0.01)[localmin : stop +10])
                crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positve direction
                if len(crossings) > 0:
                    stop_prop = crossings[-1]
                else:
                    stop_prop = np.argmax(((sacrum_acc_y/bodyweight)+0.01)[localmin:stop]) + localmin
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

            # gaitevents['AP Acceleration Sacrum right stop'] = np.append(gaitevents['AP Acceleration Sacrum right stop'], true_stop_prop)
            
            if (true_braking_to_propulsion is not None and
                true_start_brake is not None and
                true_stop_prop is not None and
                isinstance(true_braking_to_propulsion, int) and
                isinstance(true_start_brake, int) and
                isinstance(true_stop_prop, int) and
                true_start_brake < true_braking_to_propulsion < true_stop_prop):
            
                gaitevents['AP Deceleration Sacrum right start'] = np.append(
                    gaitevents['AP Deceleration Sacrum right start'], true_start_brake)
                gaitevents['AP Deceleration Sacrum right stop'] = np.append(
                    gaitevents['AP Deceleration Sacrum right stop'], true_braking_to_propulsion)
            
                gaitevents['AP Acceleration Sacrum right start'] = np.append(
                    gaitevents['AP Acceleration Sacrum right start'], true_braking_to_propulsion)
                gaitevents['AP Acceleration Sacrum right stop'] = np.append(
                    gaitevents['AP Acceleration Sacrum right stop'], true_stop_prop)

                if debugplot == True:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(start+5, stop-10), sacrum_acc_y[start+5:stop-10], label='Acceleration', color='black')
                    plt.plot(localmax, sacrum_acc_y[localmax], 'r*', label='Local Max')  # Red stars
                    plt.plot(localmin, sacrum_acc_y[localmin], 'g*', label='Local Min')  # Green stars
                    # plt.plot(braking_to_propulsion, sacrum_acc_y[braking_to_propulsion], 'bs', label='Braking → Propulsion')  # Blue squares
                    plt.plot(true_braking_to_propulsion, sacrum_acc_y[true_braking_to_propulsion], 'bs', label='True Braking → Propulsion')  # Blue squares
                    # plt.plot(int(start_brake), sacrum_acc_y[start_brake], 'mo', label='Start Brake')  # Magenta circles
                    plt.plot(true_start_brake, sacrum_acc_y[true_start_brake], 'mo', label='True Start Brake')  # Yellow circles
                    # plt.plot(int(stop_prop), sacrum_acc_y[stop_prop], 'co', label='Stop Propulsion')  # Cyan circles
                    plt.plot(true_stop_prop, sacrum_acc_y[true_stop_prop], 'co', label='True Stop Propulsion')  # Black circles
                    plt.grid(True)
                    plt.legend(loc='upper right', fontsize='small', frameon=True)
                    plt.title('Stance Phase with Gait Events - right')
                    plt.xlabel('Time (samples)')
                    plt.ylabel('Acceleration (sacrum y-axis)')
                    plt.tight_layout()
                    plt.show()

            else:
                # print(f"Skipping stance {i}: invalid brake/propulsion pair.")
                continue
            
        except:
            pass
    
    
    # Remove propulsion start/stop events in first 10 seconds of trial
    gaitevents['AP Acceleration Sacrum left start'] = gaitevents['AP Acceleration Sacrum left start'][gaitevents['AP Acceleration Sacrum left start'] > 10*sample_frequency]
    try:
        gaitevents['AP Acceleration Sacrum left stop'] = gaitevents['AP Acceleration Sacrum left stop'][gaitevents['AP Acceleration Sacrum left stop'] >= gaitevents['AP Acceleration Sacrum left start'][0]]
    except IndexError:
        gaitevents['AP Acceleration Sacrum left stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Acceleration Sacrum left start'] = gaitevents['AP Acceleration Sacrum left start'][gaitevents['AP Acceleration Sacrum left start'] <= gaitevents['AP Acceleration Sacrum left stop'][-1]]
    except IndexError:
        gaitevents['AP Acceleration Sacrum left start'] = np.array([], dtype=int)
    
    gaitevents['AP Deceleration Sacrum left start'] = gaitevents['AP Deceleration Sacrum left start'][gaitevents['AP Deceleration Sacrum left start'] > 10*sample_frequency]
    try:
        gaitevents['AP Deceleration Sacrum left stop'] = gaitevents['AP Deceleration Sacrum left stop'][gaitevents['AP Deceleration Sacrum left stop'] >= gaitevents['AP Deceleration Sacrum left start'][0]]
    except IndexError:
        gaitevents['AP Deceleration Sacrum left stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Deceleration Sacrum left start'] = gaitevents['AP Deceleration Sacrum left start'][gaitevents['AP Deceleration Sacrum left start'] <= gaitevents['AP Deceleration Sacrum left stop'][-1]]
    except IndexError:
        gaitevents['AP Deceleration Sacrum left start'] = np.array([], dtype=int)
    
    gaitevents['AP Acceleration Sacrum right start'] = gaitevents['AP Acceleration Sacrum right start'][gaitevents['AP Acceleration Sacrum right start'] > 10*sample_frequency]
    try:
        gaitevents['AP Acceleration Sacrum right stop'] = gaitevents['AP Acceleration Sacrum right stop'][gaitevents['AP Acceleration Sacrum right stop'] >= gaitevents['AP Acceleration Sacrum right start'][0]]
    except IndexError:
        gaitevents['AP Acceleration Sacrum right stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Acceleration Sacrum right start'] = gaitevents['AP Acceleration Sacrum right start'][gaitevents['AP Acceleration Sacrum right start'] <= gaitevents['AP Acceleration Sacrum right stop'][-1]]
    except IndexError:
        gaitevents['AP Acceleration Sacrum right start'] = np.array([], dtype=int)
    
    gaitevents['AP Deceleration Sacrum right start'] = gaitevents['AP Deceleration Sacrum right start'][gaitevents['AP Deceleration Sacrum right start'] > 10*sample_frequency]
    try:
        gaitevents['AP Deceleration Sacrum right stop'] = gaitevents['AP Deceleration Sacrum right stop'][gaitevents['AP Deceleration Sacrum right stop'] >= gaitevents['AP Deceleration Sacrum right start'][0]]
    except IndexError:
        gaitevents['AP Deceleration Sacrum right stop'] = np.array([], dtype=int)
    try:
        gaitevents['AP Deceleration Sacrum right start'] = gaitevents['AP Deceleration Sacrum right start'][gaitevents['AP Deceleration Sacrum right start'] <= gaitevents['AP Deceleration Sacrum right stop'][-1]]
    except IndexError:
        gaitevents['AP Deceleration Sacrum right start'] = np.array([], dtype=int)
    
    
    # Peak breaking and propulsive forces
    gaitevents['Peak AP Acceleration Sacrum left'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Acceleration Sacrum left start'])):
        try:
            idxmin = np.argmin(APacceleration_filtered [gaitevents['AP Acceleration Sacrum left start'][i] : gaitevents['AP Acceleration Sacrum left stop'][i]])
            gaitevents['Peak AP Acceleration Sacrum left'] = np.append(gaitevents['Peak AP Acceleration Sacrum left'], gaitevents['AP Acceleration Sacrum left start'][i]+idxmin)
        except ValueError:
            pass
    gaitevents['Peak AP Deceleration Sacrum left'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Deceleration Sacrum left start'])):
        try:
            idxmax = np.argmax(APacceleration_filtered [gaitevents['AP Deceleration Sacrum left start'][i] : gaitevents['AP Deceleration Sacrum left stop'][i]])
            gaitevents['Peak AP Deceleration Sacrum left'] = np.append(gaitevents['Peak AP Deceleration Sacrum left'], gaitevents['AP Deceleration Sacrum left start'][i]+idxmax)
        except ValueError:
            pass
    gaitevents['Peak AP Acceleration Sacrum right'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Acceleration Sacrum right start'])):
        try:
            idxmin = np.argmin(APacceleration_filtered [gaitevents['AP Acceleration Sacrum right start'][i] : gaitevents['AP Acceleration Sacrum right stop'][i]])
            gaitevents['Peak AP Acceleration Sacrum right'] = np.append(gaitevents['Peak AP Acceleration Sacrum right'], gaitevents['AP Acceleration Sacrum right start'][i]+idxmin)
        except ValueError:
            pass
    gaitevents['Peak AP Deceleration Sacrum right'] = np.array([], dtype=int)
    for i in range(len(gaitevents['AP Deceleration Sacrum right start'])):
        try:
            idxmax = np.argmax(APacceleration_filtered [gaitevents['AP Deceleration Sacrum right start'][i] : gaitevents['AP Deceleration Sacrum right stop'][i]])
            gaitevents['Peak AP Deceleration Sacrum right'] = np.append(gaitevents['Peak AP Deceleration Sacrum right'], gaitevents['AP Deceleration Sacrum right start'][i]+idxmax)
        except ValueError:
            pass
    
    # Debug plot
    if debugplot == True:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        axs[0].set_title(title, fontsize=20)
        # Left
        axs[0].plot(APacceleration_filtered/bodyweight, 'k', label='Acceleration Sacrum Y left')
        axs[0].plot(sacrum_acc_y/bodyweight, 'grey', label='Filtered Acceleration Sacrum Y left')
        # axs[0].plot(markerdata['Acceleration Sacrum Z filtered']/bodyweight, 'orange', label='Acceleration Sacrum Z left')
        axs[0].plot(gaitevents['Index numbers initial contact left'], APacceleration_filtered[gaitevents['Index numbers initial contact left']]/bodyweight, 'r.')
        axs[0].plot(gaitevents['Index numbers terminal contact left'], APacceleration_filtered[gaitevents['Index numbers terminal contact left']]/bodyweight, 'g.')
        # axs[0].plot(gaitevents['AP Deceleration Sacrum left start'], APacceleration_filtered[gaitevents['AP Deceleration Sacrum left start']]/bodyweight, 'kx', label='Braking start')
        axs[0].vlines(x=gaitevents['AP Deceleration Sacrum left start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='red')
        axs[0].vlines(x=gaitevents['AP Acceleration Sacrum left start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='grey')
        axs[0].vlines(x=gaitevents['AP Acceleration Sacrum left stop'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='green')
        # axs[0].plot(gaitevents['AP Acceleration Sacrum left stop'], APacceleration_filtered[gaitevents['AP Acceleration Sacrum left stop']]/bodyweight, 'kx', label='Propulsion stop')
        axs[0].plot(gaitevents['Peak AP Acceleration Sacrum left'], APacceleration_filtered[gaitevents['Peak AP Acceleration Sacrum left']]/bodyweight, 'gx', label='AP Acceleration Sacrum peak')
        axs[0].plot(gaitevents['Peak AP Deceleration Sacrum left'], APacceleration_filtered[gaitevents['Peak AP Deceleration Sacrum left']]/bodyweight, 'rx', label='AP Deceleration Sacrum peak')
        axs[0].hlines(xmin=0, xmax=len(APacceleration_filtered), y=0, color='grey')
        
        for i in range(0, len(gaitevents['AP Acceleration Sacrum left start'])):
            axs[0].fill_between(x=np.arange(gaitevents['AP Acceleration Sacrum left start'][i], gaitevents['AP Acceleration Sacrum left stop'][i]), y1=APacceleration_filtered[gaitevents['AP Acceleration Sacrum left start'][i] : gaitevents['AP Acceleration Sacrum left stop'][i]]/bodyweight, y2=0, color='lightgreen')
        for i in range(0, len(gaitevents['AP Deceleration Sacrum left start'])):
            axs[0].fill_between(x=np.arange(gaitevents['AP Deceleration Sacrum left start'][i], gaitevents['AP Deceleration Sacrum left stop'][i]), y1=APacceleration_filtered[gaitevents['AP Deceleration Sacrum left start'][i] : gaitevents['AP Deceleration Sacrum left stop'][i]]/bodyweight, y2=0, color='pink')
                
        #Right
        axs[1].plot(APacceleration_filtered/bodyweight, 'k', label='Acceleration Sacrum Y right')
        axs[1].plot(sacrum_acc_y/bodyweight, 'grey', label='Filtered Acceleration Sacrum Y right')
        # axs[1].plot(markerdata['Acceleration Sacrum Z filtered']/bodyweight, 'orange', label='Acceleration Sacrum Z')
        axs[1].plot(gaitevents['Index numbers initial contact right'], APacceleration_filtered[gaitevents['Index numbers initial contact right']]/bodyweight, 'r.', label = 'IC')
        axs[1].plot(gaitevents['Index numbers terminal contact right'], APacceleration_filtered[gaitevents['Index numbers terminal contact right']]/bodyweight, 'g.', label = 'TC')
        # axs[1].plot(gaitevents['AP Acceleration Sacrum right start'], APacceleration_filtered[gaitevents['AP Acceleration Sacrum right start']]/bodyweight, 'gv', label='Propulsion start')
        # axs[1].plot(gaitevents['AP Acceleration Sacrum right stop'], APacceleration_filtered[gaitevents['AP Acceleration Sacrum right stop']]/bodyweight, 'rv', label='Propulsion stop')
        axs[1].vlines(x=gaitevents['AP Deceleration Sacrum right start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='red')
        axs[1].vlines(x=gaitevents['AP Acceleration Sacrum right start'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='grey')
        axs[1].vlines(x=gaitevents['AP Acceleration Sacrum right stop'], ymin=np.min(APacceleration_filtered/bodyweight), ymax=np.max(APacceleration_filtered/bodyweight), color='green')
        axs[1].plot(gaitevents['Peak AP Acceleration Sacrum right'], APacceleration_filtered[gaitevents['Peak AP Acceleration Sacrum right']]/bodyweight, 'gx', label='AP Acceleration Sacrum peak')
        axs[1].plot(gaitevents['Peak AP Deceleration Sacrum right'], APacceleration_filtered[gaitevents['Peak AP Deceleration Sacrum right']]/bodyweight, 'rx', label='AP Deceleration Sacrum peak')
        axs[1].hlines(xmin=0, xmax=len(APacceleration_filtered), y=0, color='grey')
        
        for i in range(0, len(gaitevents['AP Acceleration Sacrum right start'])):
            axs[1].fill_between(x=np.arange(gaitevents['AP Acceleration Sacrum right start'][i], gaitevents['AP Acceleration Sacrum right stop'][i]), y1=APacceleration_filtered[gaitevents['AP Acceleration Sacrum right start'][i] : gaitevents['AP Acceleration Sacrum right stop'][i]]/bodyweight, y2=0, color='lightgreen')
        
        for i in range(0, len(gaitevents['AP Deceleration Sacrum right start'])):
            axs[1].fill_between(x=np.arange(gaitevents['AP Deceleration Sacrum right start'][i], gaitevents['AP Deceleration Sacrum right stop'][i]), y1=APacceleration_filtered[gaitevents['AP Deceleration Sacrum right start'][i] : gaitevents['AP Deceleration Sacrum right stop'][i]]/bodyweight, y2=0, color='pink')
        axs[1].legend()


        
    # Left side
    # AP Acceleration Sacrum = area under the negative curve
    gaitcharacteristics['AP Acceleration Sacrum left'] = np.zeros(shape=(len(gaitevents['AP Acceleration Sacrum left start']),3)) *np.nan
    for i in range(len(gaitevents['AP Acceleration Sacrum left start'])):
        gaitcharacteristics['AP Acceleration Sacrum left'][i,0] = gaitevents['AP Acceleration Sacrum left start'][i]
        gaitcharacteristics['AP Acceleration Sacrum left'][i,1] = gaitevents['AP Acceleration Sacrum left stop'][i]
        # Sacrumpute the area using the Sacrumposite trapezoidal rule.
        this_propulsion = APacceleration_filtered[gaitevents['AP Acceleration Sacrum left start'][i]:gaitevents['AP Acceleration Sacrum left stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Acceleration Sacrum left'][i,2] = forward_acceleration - backward_acceleration
        if gaitcharacteristics['AP Acceleration Sacrum left'][i,2] < 0:
            gaitcharacteristics['AP Acceleration Sacrum left'][i,2]= np.nan
    # Peak AP Acceleration Sacrum
    gaitcharacteristics['Peak AP Acceleration Sacrum left'] = np.zeros(shape=(len(gaitevents['Peak AP Acceleration Sacrum left']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Acceleration Sacrum left'])):
        gaitcharacteristics['Peak AP Acceleration Sacrum left'][i,0] = gaitevents['Peak AP Acceleration Sacrum left'][i]
        gaitcharacteristics['Peak AP Acceleration Sacrum left'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Acceleration Sacrum left'][i]])/bodyweight
    # AP Deceleration Sacrum = area under the curve
    gaitcharacteristics['AP Deceleration Sacrum left'] = np.zeros(shape=(len(gaitevents['AP Deceleration Sacrum left start']),3)) *np.nan
    for i in range(len(gaitevents['AP Deceleration Sacrum left start'])):
        gaitcharacteristics['AP Deceleration Sacrum left'][i,0] = gaitevents['AP Deceleration Sacrum left start'][i]
        gaitcharacteristics['AP Deceleration Sacrum left'][i,1] = gaitevents['AP Deceleration Sacrum left stop'][i]
        # Sacrumpute the area using the Sacrumposite trapezoidal rule.
        this_brake = APacceleration_filtered[gaitevents['AP Deceleration Sacrum left start'][i]:gaitevents['AP Deceleration Sacrum left stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_brake[this_brake<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_brake[this_brake>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Deceleration Sacrum left'][i,2] = backward_acceleration - forward_acceleration
        if gaitcharacteristics['AP Deceleration Sacrum left'][i,2] < 0:
            gaitcharacteristics['AP Deceleration Sacrum left'][i,2]= np.nan
    # Peak AP Deceleration Sacrum
    gaitcharacteristics['Peak AP Deceleration Sacrum left'] = np.zeros(shape=(len(gaitevents['Peak AP Deceleration Sacrum left']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Deceleration Sacrum left'])):
        gaitcharacteristics['Peak AP Deceleration Sacrum left'][i,0] = gaitevents['Peak AP Deceleration Sacrum left'][i]
        gaitcharacteristics['Peak AP Deceleration Sacrum left'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Deceleration Sacrum left'][i]])/bodyweight
    
    # Right side
    # AP Acceleration Sacrum = area under the negative curve
    gaitcharacteristics['AP Acceleration Sacrum right'] = np.zeros(shape=(len(gaitevents['AP Acceleration Sacrum right start']),3)) *np.nan
    for i in range(len(gaitevents['AP Acceleration Sacrum right start'])):
        gaitcharacteristics['AP Acceleration Sacrum right'][i,0] = gaitevents['AP Acceleration Sacrum right start'][i]
        gaitcharacteristics['AP Acceleration Sacrum right'][i,1] = gaitevents['AP Acceleration Sacrum right stop'][i]
        # Sacrumpute the area using the Sacrumposite trapezoidal rule.
        this_propulsion = APacceleration_filtered[gaitevents['AP Acceleration Sacrum right start'][i]:gaitevents['AP Acceleration Sacrum right stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_propulsion[this_propulsion>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Acceleration Sacrum right'][i,2] = forward_acceleration - backward_acceleration
        if gaitcharacteristics['AP Acceleration Sacrum right'][i,2] < 0:
            gaitcharacteristics['AP Acceleration Sacrum right'][i,2]= np.nan
    # Peak AP Acceleration Sacrum
    gaitcharacteristics['Peak AP Acceleration Sacrum right'] = np.zeros(shape=(len(gaitevents['Peak AP Acceleration Sacrum right']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Acceleration Sacrum right'])):
        gaitcharacteristics['Peak AP Acceleration Sacrum right'][i,0] = gaitevents['Peak AP Acceleration Sacrum right'][i]
        gaitcharacteristics['Peak AP Acceleration Sacrum right'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Acceleration Sacrum right'][i]])/bodyweight
    # AP Deceleration Sacrum = area under the curve
    gaitcharacteristics['AP Deceleration Sacrum right'] = np.zeros(shape=(len(gaitevents['AP Deceleration Sacrum right start']),3)) *np.nan
    for i in range(len(gaitevents['AP Deceleration Sacrum right start'])):
        gaitcharacteristics['AP Deceleration Sacrum right'][i,0] = gaitevents['AP Deceleration Sacrum right start'][i]
        gaitcharacteristics['AP Deceleration Sacrum right'][i,1] = gaitevents['AP Deceleration Sacrum right stop'][i]
        # Sacrumpute the area using the Sacrumposite trapezoidal rule.
        this_brake = APacceleration_filtered[gaitevents['AP Deceleration Sacrum right start'][i]:gaitevents['AP Deceleration Sacrum right stop'][i]]
        forward_acceleration = (np.abs(np.trapz(this_brake[this_brake<0])) *1/sample_frequency)/bodyweight
        backward_acceleration = (np.abs(np.trapz(this_brake[this_brake>0])) *1/sample_frequency)/bodyweight
        gaitcharacteristics['AP Deceleration Sacrum right'][i,2] = backward_acceleration - forward_acceleration
        if gaitcharacteristics['AP Deceleration Sacrum right'][i,2] < 0:
            gaitcharacteristics['AP Deceleration Sacrum right'][i,2]= np.nan
    # Peak AP Deceleration Sacrum
    gaitcharacteristics['Peak AP Deceleration Sacrum right'] = np.zeros(shape=(len(gaitevents['Peak AP Deceleration Sacrum right']),2)) *np.nan
    for i in range(len(gaitevents['Peak AP Deceleration Sacrum right'])):
        gaitcharacteristics['Peak AP Deceleration Sacrum right'][i,0] = gaitevents['Peak AP Deceleration Sacrum right'][i]
        gaitcharacteristics['Peak AP Deceleration Sacrum right'][i,1] = (APacceleration_filtered[gaitevents['Peak AP Deceleration Sacrum right'][i]])/bodyweight
    
    
    
    # Replace zeros with NaN values in impulse parameters
    impulse_parameters = [
        'AP Deceleration Sacrum left', 'AP Deceleration Sacrum right',
        'AP Acceleration Sacrum left', 'AP Acceleration Sacrum right'
    ]
    for param in impulse_parameters:
        for i in range(len(gaitcharacteristics[param])):  
            if gaitcharacteristics[param][i, 2] == 0:       # Check if third column is 0
                gaitcharacteristics[param][i, 2] = np.nan   # Replace with NaN
    
    
       
    return gaitevents, gaitcharacteristics, APacceleration


