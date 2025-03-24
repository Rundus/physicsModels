# --- ACESII_model_slice.py ---
# Description: compare the model results to the REAL ACES-II data
from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_classes import *
import time
start_time = time.time()
def ACESII_model_slice():

    # --- imports ---
    from src.physicsModels.ionosphere.conductivity.conductivity_toggles import conductivityToggles
    import numpy as np
    from copy import deepcopy
    from spaceToolsLib.tools.CDF_output import outputCDFdata
    from tqdm import tqdm

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################

    # get the spatial data dict
    data_dict_conductivity = stl.loadDictFromFile(rf'{conductivityToggles.outputFolder}\conductivity.cdf')

    # get the ACES-II L-SHell data
    data_dict_LShell_high = stl.loadDictFromFile(r'C:\Data\ACESII\science\L_shell\high\ACESII_36359_Lshell.cdf')
    data_dict_LShell_low = stl.loadDictFromFile(r'C:\Data\ACESII\science\L_shell\low\ACESII_36364_Lshell.cdf')

    # prepare the output
    data_dict_output = {'ni_slice_high': [np.zeros(shape=(len(data_dict_LShell_high['L-Shell'][0]))), {'DEPEND_0': 'Epoch_high'}],
                        'Epoch_high': data_dict_LShell_high['Epoch'],
                        'ni_slice_low': [np.zeros(shape=(len(data_dict_LShell_low['L-Shell'][0]))), {'DEPEND_0': 'Epoch_low'}],
                        'Epoch_low': data_dict_LShell_low['Epoch'],
                        }

    ####################################
    # --- SAMPLE THE MODEL AT ACESII ---
    ####################################

    stl.prgMsg('Sampling High Flyer')
    for idx1, tme in enumerate(data_dict_LShell_high['Epoch'][0]):

        # find where in the simulation data each point is closest
        Lshell_idx = np.abs(data_dict_conductivity['simLShell'][0] - data_dict_LShell_high['L-Shell'][0][idx1]).argmin()
        alt_idx = np.abs(data_dict_conductivity['simAlt'][0] - data_dict_LShell_high['Alt'][0][idx1]).argmin()
        data_dict_output['ni_slice_high'][0][idx1] = data_dict_conductivity['ne_total'][0][Lshell_idx][alt_idx]
    stl.Done(start_time)


    stl.prgMsg('Sampling Low Flyer')
    for idx1, tme in enumerate(data_dict_LShell_low['Epoch'][0]):
        # find where in the simulation data each point is closest
        Lshell_idx = np.abs(data_dict_conductivity['simLShell'][0] - data_dict_LShell_low['L-Shell'][0][idx1]).argmin()
        alt_idx = np.abs(data_dict_conductivity['simAlt'][0] - data_dict_LShell_low['Alt'][0][idx1]).argmin()
        data_dict_output['ni_slice_low'][0][idx1] = data_dict_conductivity['ne_total'][0][Lshell_idx][alt_idx]
    stl.Done(start_time)



    #####################
    # --- OUTPUT DATA ---
    #####################

    # --- Construct the Data Dict ---
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': None, 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}


    # update the data dict attrs
    for key, val in data_dict_output.items():
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict_output[key][1].items():
            newAttrs[subKey] = subVal

        data_dict_output[key][1] = newAttrs

    outputPath = rf'C:\Data\physicsModels\ionosphere\special\ACESII_model_slice.cdf'
    outputCDFdata(outputPath, data_dict_output)

ACESII_model_slice()
