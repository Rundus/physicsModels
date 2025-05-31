# --- downsample_IntegratedPotential_data.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take the ACESII integrated data, re-grid it onto the simulation and average it together by "N" datapoints

# --- bookkeeping ---
# !/usr/bin/env python
__author__ = "Connor Feltman"
__date__ = "2022-08-22"
__version__ = "1.0.0"
import time
start_time = time.time()
# --- --- --- --- ---

# --- --- --- ---
# --- TOGGLES ---
# --- --- --- ---

# --- Select the Rocket ---
# 4 -> ACES II High Flier
# 5 -> ACES II Low Flier
wRocket = [5]

# --- OutputData ---
outputData = True



# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
import spaceToolsLib as stl
from glob import glob
import numpy as np
from copy import deepcopy
from src.ionosphere_modelling.data_preparation.data_preparation_toggles import DataPreparationToggles
from scipy.interpolate import CubicSpline
from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
from spacepy import pycdf

fliers = ['high','low']
rocketID = ['36359','36364']


#######################
# --- MAIN FUNCTION ---
#######################
def downsample_IntegratedPotential_data(wRocket):

    # --- Load the Data ---
    stl.prgMsg(f'Loading data')
    data_dict_potential = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\science\integrated_potential\{fliers[wRocket-4]}\*integrated_potential.cdf*')[0])
    data_dict_eepaa = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\L2\{fliers[wRocket-4]}\*eepaa_fullCal.cdf*')[0])
    data_dict_spatial = stl.loadDictFromFile(glob(rf'C:\Data\physicsModels\ionosphere\spatial_environment\spatial_environment.cdf')[0])
    data_dict_traj_high =stl.loadDictFromFile('C:\Data\ACESII\coordinates\Lshell\high\ACESII_36359_Lshell.cdf')
    stl.Done(start_time)

    # --- prepare the output ---
    data_dict_output = {}

    ###########################################################
    # --- INTERPOLATE THE POTENTIAL ONTO THE EEPAA TIMEBASE ---
    ###########################################################
    Epoch_eepaa_tt2200 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_eepaa['Epoch'][0]])
    Epoch_potential_tt2000 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_potential['Epoch'][0]])

    for key in data_dict_potential.keys():
        if key not in ['Epoch']:
            cs = CubicSpline(Epoch_potential_tt2000, data_dict_potential[key][0])
            data_dict_potential[key][0] = cs(Epoch_eepaa_tt2200)

    data_dict_potential['Epoch'][0] = deepcopy(data_dict_eepaa['Epoch'][0])

    # --- --- --- --- --- --- ---
    # --- DOWNSAMPLE THE DATA ---
    # --- --- --- --- --- --- ---
    stl.prgMsg('Downsampling Data')

    dlen = len(data_dict_potential['Epoch'][0])
    if len(data_dict_potential['Epoch'][0])% DataPreparationToggles.N_avg != 0:
        dlen -= dlen%DataPreparationToggles.N_avg

    # shorten the tail of the data by this much
    for key in data_dict_potential.keys():
        data_dict_potential[key][0] = data_dict_potential[key][0][:dlen]

    # store the output
    data_dict_output = {
        **data_dict_output,
        **data_dict_potential
    }

    # --- Downsample the Epoch ---
    # Epoch_chunked = np.split(data_dict_potential['Epoch'][0], round(len(data_dict_potential['Epoch'][0]) / DataPreparationToggles.N_avg))
    # data_dict_potential['Epoch'][0] = np.array([Epoch_chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(Epoch_chunked))])

    # --- Downsample the single-dimension variables---
    for key in data_dict_potential.keys():
        chunked = np.split(data_dict_potential[key][0], round(len(data_dict_potential[key][0]) / DataPreparationToggles.N_avg))
        data_dict_potential[key][0] = deepcopy(np.array([chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(chunked))]))

    # --- Since the integrated potential is time-limited to ONLY the integration region, limit the downsampled-data to that same region ---

    # find where the high flyer is above the threshold
    L_shell_region = data_dict_traj_high['L-Shell'][0][np.where(data_dict_traj_high['Alt'][0] >= SpatialToggles.altThresh)[0]]

    low_idx, high_idx = np.abs(data_dict_output['L-Shell'][0] - L_shell_region[0]).argmin(),np.abs(data_dict_output['L-Shell'][0] - L_shell_region[-1]).argmin()

    for key in data_dict_output.keys():
        data_dict_output[key][0] = data_dict_output[key][0][low_idx+1:high_idx+1]

    # --- --- --- --- --- --- ---
    # --- WRITE OUT THE DATA ---
    # --- --- --- --- --- --- ---
    if outputData:

        stl.prgMsg('Creating output file')
        fileoutName = f'ACESII_{rocketID[wRocket-4]}_integrated_potential_downsampled_{DataPreparationToggles.N_avg}.cdf'
        outputPath = f'C:\Data\physicsModels\ionosphere\data_inputs\integrated_potential\\{fliers[wRocket-4]}\\' + fileoutName
        stl.outputCDFdata(outputPath, data_dict_output)
        stl.Done(start_time)





# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---

for idx in wRocket:
    downsample_IntegratedPotential_data(idx)

