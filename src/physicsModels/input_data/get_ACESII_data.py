# --- get_ACESII_data.py ---
# description: get the data from ACES-II and put it in input_data folder under physics_models
# Downsample/trim/change the data as needed


# --- imports ---
from copy import deepcopy
import spaceToolsLib as stl
import numpy as np
from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import helperFuncs
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import primaryBeamToggles


def get_ACESII_data():

    ########################
    # --- DIFFNFLUX DATA ---
    ########################

    # get the data
    data_dict_diffFlux = stl.loadDictFromFile('C:\Data\ACESII\L2\high\ACESII_36359_l2_eepaa_fullCal.cdf',
                                           wKeys_Load=['ILat',
                                                        'Epoch',
                                                         'Alt',
                                                         'Energy',
                                                         'Pitch_Angle',
                                                         'Differential_Number_Flux',
                                                         'Differential_Energy_Flux',
                                                         'Differential_Number_Flux_stdDev'])


    HF_iLat_at_50km = [69.267, 73.915]
    low_idx, high_idx = np.abs(data_dict_diffFlux['ILat'][0] - HF_iLat_at_50km[0]).argmin(), np.abs(data_dict_diffFlux['ILat'][0] - HF_iLat_at_50km[1]).argmin()

    # trim  and downsample the dataset to only data above 50km
    EpochFitData, ILatFitData, AltFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(
        data_dict_diffFlux=deepcopy(data_dict_diffFlux),
        targetTimes=[data_dict_diffFlux['Epoch'][0][low_idx], data_dict_diffFlux['Epoch'][0][high_idx]],
        N_avg=primaryBeamToggles.numToAverageOver)

    # output the data
    data_dict_output = {
                        'Alt': [AltFitData, deepcopy(data_dict_diffFlux['Alt'][1])],
                        'Epoch': [EpochFitData, deepcopy(data_dict_diffFlux['Epoch'][1])],
                        'ILat': [ILatFitData, deepcopy(data_dict_diffFlux['ILat'][1])],
                        'Pitch_Angle': deepcopy(data_dict_diffFlux['Pitch_Angle']),
                        'Energy': deepcopy(data_dict_diffFlux['Energy']),
                        'diffNFlux_avg': [diffNFlux_avg, deepcopy(data_dict_diffFlux['Differential_Number_Flux'][1])]
                        }

    # --- OUTPUT DATA ---
    output_folder = 'C:\Data\physicsModels\input_data\ACESII'
    outputPath = rf'{output_folder}\ACESII_36359_eepaa_avg.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)


get_ACESII_data()