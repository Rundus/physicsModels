# --- L2_to_L3_downsampled_data.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take the ACESII EEPAA data and average it together by "N" datapoints

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
def downsample_EEPAA_data(wRocket):

    # --- Load the Data ---
    stl.prgMsg(f'Loading data')
    data_dict_eepaa = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\L2\{fliers[wRocket-4]}\*eepaa_fullCal.cdf*')[0], wKeys_Load=['Epoch',
                                                                                                                               'Differential_Energy_Flux',
                                                                                                                               'Differential_Number_Flux',
                                                                                                                               'Energy',
                                                                                                                               'Pitch_Angle',
                                                                                                                               'Lat_geom',
                                                                                                                               'Long_geom',
                                                                                                                               'Alt_geom'])
    data_dict_Lshell = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\coordinates\Lshell\{fliers[wRocket-4]}\*_Lshell.cdf*')[0], wKeys_Load=[
                                                                                                                                          'L-Shell',
                                                                                                                                          'Alt',
                                                                                                                                          'Lat',
                                                                                                                                          'Long'])
    stl.Done(start_time)

    # --- prepare the output ---
    data_dict_output = {}

    # --- --- --- --- --- --- ---
    # --- DOWNSAMPLE THE DATA ---
    # --- --- --- --- --- --- ---
    stl.prgMsg('Downsampling Data')

    dlen = len(data_dict_eepaa['Epoch'][0])
    if len(data_dict_eepaa['Epoch'][0])% DataPreparationToggles.N_avg != 0:
        dlen -= dlen%DataPreparationToggles.N_avg

    # shorten the tail of the data by this much
    for key in data_dict_eepaa.keys():
        data_dict_eepaa[key][0] = data_dict_eepaa[key][0][:dlen]
    for key in data_dict_Lshell.keys():
        data_dict_Lshell[key][0] = data_dict_Lshell[key][0][:dlen]

    # --- Downsample the Epoch ---
    Epoch_chunked = np.split(data_dict_eepaa['Epoch'][0], round(len(data_dict_eepaa['Epoch'][0]) / DataPreparationToggles.N_avg))
    Epoch_ds = np.array([Epoch_chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(Epoch_chunked))])
    data_dict_eepaa['Epoch'][0] = Epoch_ds

    # --- downsample Lat_geom ---
    eepaa_keys = ['Lat_geom', 'Long_geom', 'Alt_geom']
    for key in eepaa_keys:
        chunked = np.split(data_dict_eepaa[key][0], round(len(data_dict_eepaa[key][0]) / DataPreparationToggles.N_avg))
        ds = np.array([chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(chunked))])
        data_dict_eepaa[key][0] = deepcopy(ds)

    # --- Downsample the Lat, Long, Alt ---
    Lshell_keys = ['Lat', 'Long', 'Alt', 'L-Shell']
    for key in Lshell_keys:
        chunked = np.split(data_dict_Lshell[key][0], round(len(data_dict_Lshell[key][0]) / DataPreparationToggles.N_avg))
        ds = np.array([chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(chunked))])
        data_dict_Lshell[key][0] = deepcopy(ds)

    # Lat_chunked = np.split(data_dict_Lshell['Lat'][0], round(len(data_dict_Lshell['Lat'][0]) / DataPreparationToggles.N_avg))
    # Lat_ds = np.array([Lat_chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(Lat_chunked))])
    #
    # Long_chunked = np.split(data_dict_Lshell['Long'][0], round(len(data_dict_Lshell['Long'][0]) / DataPreparationToggles.N_avg))
    # Long_ds = np.array([Long_chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(Long_chunked))])
    #
    # Alt_chunked = np.split(data_dict_Lshell['Alt'][0], round(len(data_dict_Lshell['Alt'][0]) / DataPreparationToggles.N_avg))
    # Alt_ds = np.array([Alt_chunked[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(Alt_chunked))])
    #
    # LShell_chunck = np.split(data_dict_Lshell['L-Shell'][0], round(len(data_dict_Lshell['L-Shell'][0]) / DataPreparationToggles.N_avg))
    # LShell_ds = np.array([LShell_chunck[i][int((DataPreparationToggles.N_avg - 1) / 2)] for i in range(len(LShell_chunck))])

    # --- Downsample the multi-dimensional data ---
    Pitch_Angle = data_dict_eepaa['Pitch_Angle'][0]
    Energy = data_dict_eepaa['Energy'][0]
    diffNFlux = deepcopy(data_dict_eepaa['Differential_Number_Flux'][0])
    diffEFlux = deepcopy(data_dict_eepaa['Differential_Energy_Flux'][0])
    diffEFlux_avg = np.zeros(shape=(len(Epoch_ds), len(Pitch_Angle), len(Energy)))
    diffNFlux_avg = np.zeros(shape=(len(Epoch_ds), len(Pitch_Angle), len(Energy)))

    for loopIdx, pitchValue in enumerate(Pitch_Angle):

        diffEFlux_chunked = np.split(diffEFlux[:, loopIdx, :], round(len(diffEFlux[:, loopIdx, :]) / DataPreparationToggles.N_avg))
        diffNFlux_chunked = np.split(diffNFlux[:, loopIdx, :], round(len(diffNFlux[:, loopIdx, :]) / DataPreparationToggles.N_avg))

        # --- Average the chunked data ---
        diffEFlux_temp = np.zeros(shape=(len(diffEFlux_chunked), len(Energy)))
        diffNFlux_temp = np.zeros(shape=(len(diffNFlux_chunked), len(Energy)))

        for i in range(len(Epoch_chunked)):

            # average the diffNFlux data by only choosing data which is valid
            diffEFlux_chunked[i][diffEFlux_chunked[i] < 0] = np.NaN
            diffEFlux_temp[i] = np.nanmean(diffEFlux_chunked[i], axis=0)

            # average the diffNFlux data by only choosing data which is valid
            diffNFlux_chunked[i][diffNFlux_chunked[i] < 0] = np.NaN
            diffNFlux_temp[i] = np.nanmean(diffNFlux_chunked[i], axis=0)

        diffEFlux_avg[:, loopIdx, :] = diffEFlux_temp
        diffNFlux_avg[:, loopIdx, :] = diffNFlux_temp

    # Store the outputs
    data_dict_eepaa['Differential_Number_Flux'][0] = diffNFlux_avg
    data_dict_eepaa['Differential_Energy_Flux'][0] = diffEFlux_avg
    data_dict_output = {**data_dict_output, **data_dict_eepaa}
    data_dict_output = {**data_dict_output,**data_dict_Lshell}

    # data_dict_output = {
    #                     'Differential_Number_Flux':[diffNFlux_avg, deepcopy(data_dict_eepaa['Differential_Number_Flux'][1])],
    #                     'Differential_Energy_Flux': [diffEFlux_avg, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
    #                     'Energy': [Energy, deepcopy(data_dict_eepaa['Energy'][1])],
    #                     'Pitch_Angle': [Pitch_Angle, deepcopy(data_dict_eepaa['Pitch_Angle'][1])],
    #                     'Alt': deepcopy(data_dict_Lshell['Alt']),
    #                     'Lat': deepcopy(data_dict_Lshell['Lat']),
    #                     'Long': deepcopy(data_dict_Lshell['Long']),
    #                     'Epoch': [Epoch_ds, deepcopy(data_dict_eepaa['Epoch'][1])],
    #                     'L-Shell': deepcopy(data_dict_Lshell['L-Shell'])
    #                     }

    # --- --- --- --- --- --- ---
    # --- WRITE OUT THE DATA ---
    # --- --- --- --- --- --- ---
    if outputData:

        stl.prgMsg('Creating output file')
        fileoutName = f'ACESII_{rocketID[wRocket-4]}_eepaa_downsampled_{DataPreparationToggles.N_avg}.cdf'
        outputPath = f'C:\Data\physicsModels\ionosphere\data_inputs\eepaa\\{fliers[wRocket-4]}\\' + fileoutName
        stl.outputCDFdata(outputPath, data_dict_output)
        stl.Done(start_time)
        print('\n')





# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---

for idx in wRocket:
    downsample_EEPAA_data(idx)

