# --- collect_langmuir_density_altitude_statistics.py---
# Use the ACES-II LP data on both pa

# --- bookkeeping ---
# !/usr/bin/env python
__author__ = "Connor Feltman"
__date__ = "2022-08-22"
__version__ = "1.0.0"

import matplotlib.pyplot as plt
import numpy as np
import time
start_time = time.time()
# --- --- --- --- ---

# --- Select the Rocket ---
# 0 -> Integration High Flier
# 1 -> Integration Low Flier
# 2 -> TRICE II High Flier
# 3 -> TRICE II Low Flier
# 4 -> ACES II High Flier
# 5 -> ACES II Low Flier
wRocket = 5


# --- OutputData ---
outputData = True

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
import spaceToolsLib as stl
stl.setupPYCDF()
from spacepy import pycdf
from copy import deepcopy



def collect_langmuir_density_altitude_statistics(wflyer):
    rocket_ID = ['36359', '36364'][wflyer]
    ACES_data_folder = r'C:\Data\\ACESII\\'

    stl.prgMsg(f'Loading ACES-II {rocket_ID} Data')


    # load the specific data_dict
    wFlyer = ['high', 'low'][wflyer]


    data_dict_eepaa = stl.loadDictFromFile(ACES_data_folder + f'\\L2\\{wFlyer}\\ACESII_{rocket_ID}_l2_eepaa_fullCal.cdf')
    data_dict_energy_flux = stl.loadDictFromFile(ACES_data_folder + f'\\L3\\Energy_Flux\\{wFlyer}\\ACESII_{rocket_ID}_l3_eepaa_Flux.cdf')
    data_dict_LP = stl.loadDictFromFile(ACES_data_folder + f'\\L3\\Langmuir\\{wFlyer}\\ACESII_{rocket_ID}_l3_langmuir_fixed.cdf')
    data_dict_attitude = stl.loadDictFromFile(ACES_data_folder + f'\\attitude\\{wFlyer}\\ACESII_{rocket_ID}_Attitude_Solution.cdf')
    data_dict_LShell = stl.loadDictFromFile(ACES_data_folder + f'\\science\\L_shell\\{wFlyer}\\ACESII_{rocket_ID}_Lshell.cdf')
    stl.Done(start_time)

    # store the EEPAA data
    stl.prgMsg('Downsampling Data')

    # downsample the LP data to the eepaa data - use TT2000 as the search algorithim is MUCH more time-efficient
    LP_Epoch_tt2000 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_LP['Epoch'][0]])
    EEPAA_Epoch_tt2000 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_eepaa['Epoch'][0]])
    downsampled_indicies = np.array([np.abs(LP_Epoch_tt2000 - val).argmin() for val in EEPAA_Epoch_tt2000])
    data_dict_LP['ni'][0] = data_dict_LP['ni'][0][downsampled_indicies]

    # downsample the attitude data to the eepaa Epoch
    Aittutde_Epoch_tt2000 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_attitude['Epoch'][0]])
    downsampled_indicies = np.array([np.abs(Aittutde_Epoch_tt2000 - val).argmin() for val in EEPAA_Epoch_tt2000])
    for key in ['Alt', 'ILat', 'ILong', 'Lat', 'Long']:
        data_dict_attitude[key][0] = data_dict_attitude[key][0][downsampled_indicies]
    stl.Done(start_time)

    # --- activity vs. quiet ---
    # Determine the ILats which are active with auroral particles vs quiet
    stl.prgMsg('Collecting Quiet vs Active Region Data')

    # use the parallel downward energy flux to determine quiet vs non-quiet regions
    #   find all the regions where the is no eepaa data --> these are quiet regions
    downward_Phi_E_flux = data_dict_energy_flux['Phi_E'][0]
    quiet_indicies = np.array([idx for idx, val in enumerate(downward_Phi_E_flux) if val != val or val <= 1E12])
    active_indices = np.array([idx for idx, val in enumerate(downward_Phi_E_flux) if val == val or val >= 1E12])
    # quiet_indicies = np.array([idx for idx, slice in enumerate(downward_varPhi_flux) if np.all(slice == np.zeros(shape=(len(slice)))) or np.isnan(np.min(slice))])
    # active_indices = np.array([idx for idx, slice in enumerate(downward_varPhi_flux) if np.all(slice != np.zeros(shape=(len(slice)))) or not np.isnan(np.min(slice))])

    # get the quiet statistics
    quiet_ni = data_dict_LP['ni'][0][quiet_indicies]
    quiet_LShells = data_dict_LShell['L-Shell'][0][quiet_indicies]
    quiet_alts = data_dict_attitude['Alt'][0][quiet_indicies]
    quiet_Epoch = data_dict_eepaa['Epoch'][0][quiet_indicies]

    # get the active statistics
    active_ni = data_dict_LP['ni'][0][active_indices]
    active_LShells = data_dict_LShell['L-Shell'][0][active_indices]
    active_alts = data_dict_attitude['Alt'][0][active_indices]
    active_Epoch = data_dict_eepaa['Epoch'][0][active_indices]
    stl.Done(start_time)

    #########################
    # --- OUTPUT THE DATA ---
    #########################
    output_folder = r'C:\Data\physicsModels\ionosphere\plasma_environment\ACESII_ni_spectrum\\'

    data_dict_output = {
        'quiet_ni': [quiet_ni, deepcopy(data_dict_LP['ni'][1])],
        'active_ni': [active_ni, deepcopy(data_dict_LP['ni'][1])],
        'quiet_alts': [quiet_alts, deepcopy(data_dict_attitude['Alt'][1])],
        'active_alts': [active_alts, deepcopy(data_dict_attitude['Alt'][1])],
        'quiet_LShells': [quiet_LShells, deepcopy(data_dict_LShell['L-Shell'][1])],
        'active_LShells': [active_LShells, deepcopy(data_dict_LShell['L-Shell'][1])],
        'quiet_Epoch': [quiet_Epoch, deepcopy(data_dict_eepaa['Epoch'][1])],
        'active_Epoch': [active_Epoch, deepcopy(data_dict_eepaa['Epoch'][1])],
    }

    for key in data_dict_output.keys():
        data_dict_output[key][1]['DEPEND_0'] = None

    if outputData:
        # output the High Flyer data
        stl.outputCDFdata(outputPath=output_folder + f'{wFlyer}\\ACESII_{rocket_ID}_langmuir_ni_statistics.cdf',
                          data_dict=data_dict_output)


# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---
collect_langmuir_density_altitude_statistics(wRocket-4)