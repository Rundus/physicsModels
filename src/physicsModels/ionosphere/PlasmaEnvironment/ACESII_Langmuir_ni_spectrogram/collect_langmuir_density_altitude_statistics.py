# --- collect_langmuir_density_altitude_statistics.py---
# Use the ACES-II LP data on both pa

# --- bookkeeping ---
# !/usr/bin/env python
__author__ = "Connor Feltman"
__date__ = "2022-08-22"
__version__ = "1.0.0"

import matplotlib.pyplot as plt
import numpy as np


from src.my_imports import *
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
wRocket = 4


# --- OutputData ---
outputData = False

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from src.mission_attributes import *
from src.data_paths import DataPaths
stl.setupPYCDF()
from spacepy import pycdf


def collect_langmuir_density_altitude_statistics(wflyer):
    rocket_ID = ACESII.payload_IDs[wflyer]
    stl.prgMsg(f'Loading ACES-II {rocket_ID} Data')

    # load the specific data_dict
    wFlyer = ACESII.fliers[wflyer]
    data_dict_eepaa = stl.loadDictFromFile(DataPaths.ACES_data_folder + f'\\L2\\{wFlyer}\\ACESII_{rocket_ID}_l2_eepaa_fullCal.cdf')
    data_dict_energy_flux = stl.loadDictFromFile(DataPaths.ACES_data_folder + f'\\L3\\Energy_Flux\\{wFlyer}\\ACESII_{rocket_ID}_eepaa_Energy_Flux.cdf')
    data_dict_LP = stl.loadDictFromFile(DataPaths.ACES_data_folder + f'\\L3\\Langmuir\\{wFlyer}\\ACESII_{rocket_ID}_l3_langmuir_fixed.cdf')
    data_dict_attitude = stl.loadDictFromFile(DataPaths.ACES_data_folder + f'\\attitude\\{wFlyer}\\ACESII_{rocket_ID}_Attitude_Solution.cdf')
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
    downward_energy_flux = data_dict_energy_flux['Energy_Flux_Downward'][0]
    quiet_indicies = np.array([idx for idx, val in enumerate(downward_energy_flux) if val != val or val == 0.0])
    active_indices = np.array([idx for idx, val in enumerate(downward_energy_flux) if val == val and val != 0.0])

    # get the quiet statistics
    quiet_ni = data_dict_LP['ni'][0][quiet_indicies]
    quiet_ILats = data_dict_attitude['ILat'][0][quiet_indicies]
    quiet_alts = data_dict_attitude['Alt'][0][quiet_indicies]
    quiet_Epoch = data_dict_eepaa['Epoch'][0][quiet_indicies]

    # get the active statistics
    active_ni = data_dict_LP['ni'][0][active_indices]
    active_ILats = data_dict_attitude['ILat'][0][active_indices]
    active_alts = data_dict_attitude['Alt'][0][active_indices]
    active_Epoch = data_dict_eepaa['Epoch'][0][active_indices]
    stl.Done(start_time)

    #########################
    # --- OUTPUT THE DATA ---
    #########################
    output_folder = r'C:\Data\ACESII\science\Langmuir\\'

    data_dict_output = {
        'quiet_ni': [quiet_ni, deepcopy(data_dict_LP['ni'][1])],
        'active_ni': [active_ni, deepcopy(data_dict_LP['ni'][1])],
        'quiet_alts': [quiet_alts, deepcopy(data_dict_attitude['Alt'][1])],
        'active_alts': [active_alts, deepcopy(data_dict_attitude['Alt'][1])],
        'quiet_ILats': [quiet_ILats, deepcopy(data_dict_attitude['ILat'][1])],
        'active_ILats': [active_ILats, deepcopy(data_dict_attitude['ILat'][1])],
        'quiet_Epoch': [quiet_Epoch, deepcopy(data_dict_eepaa['Epoch'][1])],
        'active_Epoch': [active_Epoch, deepcopy(data_dict_eepaa['Epoch'][1])],
    }

    for key in data_dict_output.keys():
        data_dict_output[key][1]['DEPEND_0'] = None

    if outputData:
        # output the High Flyer data
        stl.outputCDFdata(outputPath=output_folder + f'{ACESII.fliers[wflyer]}\\ACESII_{rocket_ID}_langmuir_ni_statistics.cdf',
                          data_dict=data_dict_output)


# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---
collect_langmuir_density_altitude_statistics(wRocket-4)