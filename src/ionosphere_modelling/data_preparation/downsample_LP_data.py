# --- downsample_LP_data.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take the ACESII energy flux EEPAA data and average it together by "N" datapoints

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
wRocket = [4, 5]

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

fliers = ['high','low']
rocketID = ['36359','36364']


#######################
# --- MAIN FUNCTION ---
#######################
def downsample_LP_data(wRocket):

    # --- Load the Data ---
    stl.prgMsg(f'Loading data')
    data_dict_LP = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\L3\Langmuir\\{fliers[wRocket-4]}\*langmuir_fixed.cdf*')[0])
    stl.Done(start_time)

    # --- prepare the output ---
    data_dict_output = {}
    #
    # ############################################################
    # # --- INTERPOLATE THE LP DENSITY ONTO THE EEPAA TIMEBASE ---
    # ############################################################
    # Epoch_eepaa_tt2200 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_eepaa['Epoch'][0]])
    # Epoch_potential_tt2000 = np.array([pycdf.lib.datetime_to_tt2000(val) for val in data_dict_potential['Epoch'][0]])
    #
    # for key in data_dict_potential.keys():
    #     if key not in ['Epoch']:
    #         cs = CubicSpline(Epoch_potential_tt2000, data_dict_potential[key][0])
    #         data_dict_potential[key][0] = cs(Epoch_eepaa_tt2200)
    #
    # data_dict_potential['Epoch'][0] = deepcopy(data_dict_eepaa['Epoch'][0])
    #
    # # --- --- --- --- --- --- ---
    # # --- DOWNSAMPLE THE DATA ---
    # # --- --- --- --- --- --- ---
    # stl.prgMsg('Downsampling Data')
    #
    # dlen = len(data_dict_LP['Epoch'][0])
    # if len(data_dict_LP['Epoch'][0])% DataPreparationToggles.N_avg != 0:
    #     dlen -= dlen%DataPreparationToggles.N_avg
    #
    # # shorten the tail of the data by this much
    # for key in data_dict_LP.keys():
    #     if key not in ['Pitch_Angle','Energy']:
    #         data_dict_LP[key][0] = data_dict_LP[key][0][:dlen]
    #
    # # --- Downsample the Epoch ---
    # Epoch_chunked = np.split(data_dict['Epoch'][0], round(len(data_dict['Epoch'][0]) / DataPreparationToggles.N_avg))
    # data_dict['Epoch'][0] = np.array([Epoch_chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(Epoch_chunked))])
    #
    # # --- Downsample the single-dimension variables---
    # for key in data_dict.keys():
    #     if key not in ['Pitch_Angle','Energy','Epoch']:
    #         chunked = np.split(data_dict[key][0], round(len(data_dict[key][0]) / DataPreparationToggles.N_avg))
    #         data_dict[key][0] = deepcopy(np.array([chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(chunked))]))
    #
    # # store the output
    # data_dict_output = {
    #                     **data_dict_output,
    #                     **data_dict
    #                     }
    #
    # # --- --- --- --- --- --- ---
    # # --- WRITE OUT THE DATA ---
    # # --- --- --- --- --- --- ---
    # if outputData:
    #
    #     stl.prgMsg('Creating output file')
    #     fileoutName = f'ACESII_{rocketID[wRocket-4]}_eepaa_flux_downsampled_{DataPreparationToggles.N_avg}.cdf'
    #     outputPath = f'C:\Data\physicsModels\ionosphere\data_inputs\energy_flux\\{fliers[wRocket-4]}\\' + fileoutName
    #     stl.outputCDFdata(outputPath, data_dict_output)
    #     stl.Done(start_time)





# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---

for idx in wRocket:
    downsample_EnergyFlux_data(idx)

