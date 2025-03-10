# --- Langmuir_ni_spectrogram.py ---
# Description: Take in the HF/LF statistics data I
# from collect_langmuir_densitY_altitude_statistics.py and
# make a Alt vs. ILat vs density map from the ACES-II data
# on the EEPAA epoch
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



# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from scipy.optimize import curve_fit
import spaceToolsLib as stl
from copy import deepcopy
from src.physicsModels.ionosphere.spatial_environment.spatial_toggles import SpatialToggles


#################
# --- TOGGLES ---
#################
fitting_Ilat_bin_rez = 0.25 # in degrees

# --- OutputData ---
bool_show_num_points_plot = False
bool_output_data = True


def linear(x,a,b):
    return a*x+b


def langmuir_ni_spectrogram():

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_HF_statistics = stl.loadDictFromFile('C:\Data\ACESII\science\Langmuir\high\ACESII_36359_langmuir_ni_statistics.cdf')
    data_dict_LF_statistics = stl.loadDictFromFile('C:\Data\ACESII\science\Langmuir\low\ACESII_36364_langmuir_ni_statistics.cdf')
    data_dicts = [data_dict_HF_statistics,  data_dict_LF_statistics]

    ######################
    # --- BIN THE DATA ---
    ######################
    alt_bins = SpatialToggles.simAlt
    simILats = SpatialToggles.simILat
    max_val = max(data_dict_HF_statistics['quiet_ILats'][0])
    min_val = min(data_dict_HF_statistics['quiet_ILats'][0])
    ilat_bins = np.linspace(min_val, max_val, int((max_val - min_val)/fitting_Ilat_bin_rez + 1))
    fit_array = [[[] for alt in alt_bins] for ilat in ilat_bins]

    # populate the fitting array with both flyer's data
    for wflyer in range(2):
        for tmeIdx in range(len(data_dicts[wflyer]['quiet_Epoch'][0])):
            ilat_idx = np.abs(ilat_bins - data_dicts[wflyer]['quiet_ILats'][0][tmeIdx]).argmin()
            alt_idx = np.abs(alt_bins - data_dicts[wflyer]['quiet_alts'][0][tmeIdx]).argmin()
            fit_array[ilat_idx][alt_idx].append(data_dicts[wflyer]['quiet_ni'][0][tmeIdx])

    fit_num_points = np.zeros(shape=(len(fit_array),len(fit_array[0])))
    for iLat_idx in range(len(fit_array)):
        for alt_idx in range(len(fit_array[0])):
            fit_num_points[iLat_idx][alt_idx] = len(fit_array[iLat_idx][alt_idx])

    if bool_show_num_points_plot:
        fig, ax = plt.subplots()
        cmap = ax.pcolormesh(ilat_bins, alt_bins/stl.m_to_km, fit_num_points.T, norm='log')
        ax.set_ylabel('Alt [km]')
        ax.set_xlabel('ILat (150km) [deg]')
        ax.set_ylim(0, 420)
        ax.set_xlim(69, 74)
        cbar = plt.colorbar(cmap)
        cbar.set_label('Num of Points')
        ax.grid(which='both')
        ax.set_yticks(alt_bins/stl.m_to_km)
        ax.set_xticks(ilat_bins)
        plt.show()

    ############################
    # --- FIT THE DATA ARRAY ---
    ############################
    # for each altitude in the fit_array, fit a linear line to ni = m (ILat) + b
    # evaluate that line over all latitude values, and choose altitudes up to the maximum value of the high flyer
    density_spectrum_T = np.zeros(shape=(len(alt_bins), len(simILats)))

    for alt_idx in range(len(alt_bins)):

        # get the data to fit
        density_vals = [fit_array[i][alt_idx] for i in range(len(ilat_bins))]
        ilat_vals = [[ilat_bins[idx] for val in range(len(lst))] for idx, lst in enumerate(density_vals) if len(lst) >= 1]
        density_vals = [val for sublist in density_vals for val in sublist]

        if len(ilat_vals) == 1:
            density_spectrum_T[alt_idx] = [np.mean(density_vals) for i in range(len(simILats))]
        elif len(ilat_vals) > 1:
            ilat_vals = [val for sublist in ilat_vals for val in sublist]

            # fit the data at this specific altitude
            params, cov = curve_fit(linear, ilat_vals, density_vals)

            # evaluate the fit at a specific region
            density_spectrum_T[alt_idx] = linear(simILats, *params)

    density_spectrum_T[np.where(density_spectrum_T == 0.0)[0]] = 51395
    density_spectrum = density_spectrum_T.T


    if bool_output_data:
        data_dict_output = {
            'ni_spectrum': [density_spectrum, deepcopy(data_dict_HF_statistics['quiet_ni'][1])],
            'ilat_bins': [simILats, deepcopy(data_dict_HF_statistics['quiet_ILats'][1])],
            'alt_bins': [alt_bins, deepcopy(data_dict_HF_statistics['quiet_alts'][1])]
        }

        for key in data_dict_output.keys():
            data_dict_output[key][1]['DEPEND_0'] = None
            data_dict_output[key][1]['var_type'] = 'data'

        data_dict_output['ni_spectrum'][1]['DEPEND_0'] = 'ilat_bins'
        data_dict_output['ni_spectrum'][1]['DEPEND_1'] = 'alt_bins'
        data_dict_output['alt_bins'][1]['UNITS'] = 'm'

        output_folder = 'C:\Data\physicsModels\ionosphere\plasmaEnvironment'
        stl.outputCDFdata(outputPath=output_folder + '\\ACESII_ni_spectrum.cdf',
                          data_dict=data_dict_output)




langmuir_ni_spectrogram()