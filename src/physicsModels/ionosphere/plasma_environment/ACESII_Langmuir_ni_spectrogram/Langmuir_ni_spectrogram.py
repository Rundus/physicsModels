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
from scipy.interpolate import CubicSpline
import spaceToolsLib as stl
from copy import deepcopy
from src.physicsModels.ionosphere.spatial_environment.spatial_toggles import SpatialToggles


#################
# --- TOGGLES ---
#################
fitting_LShell_bin_rez = 0.25 # in degrees

# --- OutputData ---
bool_show_num_points_plot = False
bool_output_data = True


def linear(x,a,b):
    return a*x+b


def langmuir_ni_spectrogram():

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_HF_statistics = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\plasma_environment\ACESII_ni_spectrum\high\ACESII_36359_langmuir_ni_statistics.cdf')
    data_dict_LF_statistics = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\plasma_environment\ACESII_ni_spectrum\low\ACESII_36364_langmuir_ni_statistics.cdf')
    data_dicts = [data_dict_HF_statistics,  data_dict_LF_statistics]
    data_dict_spatial = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\spatial_environment\spatial_environment.cdf')

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    simAlt = SpatialToggles.simAlt
    simLShell = SpatialToggles.simLShell

    data_dict_output = {
        'ni_spectrum': [np.zeros(shape=(len(simLShell),len(simAlt))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'cm!A-3!N', 'LABLAXIS': 'ni_spectrum'}],
        'simLShell': deepcopy(data_dict_spatial['simLShell']),
        'simAlt': deepcopy(data_dict_spatial['simAlt']),
    }

    ######################
    # --- BIN THE DATA ---
    ######################
    max_val = max(data_dict_HF_statistics['quiet_LShells'][0])
    min_val = min(data_dict_HF_statistics['quiet_LShells'][0])
    LShell_bins = np.linspace(min_val, max_val, int((max_val - min_val)/fitting_LShell_bin_rez + 1))
    fit_array = [[[] for alt in simAlt] for LShell in LShell_bins]

    # populate the fitting array with both flyer's data
    for wflyer in range(2):
        for tmeIdx in range(len(data_dicts[wflyer]['quiet_Epoch'][0])):
            LShell_idx = np.abs(LShell_bins - data_dicts[wflyer]['quiet_LShells'][0][tmeIdx]).argmin()
            alt_idx = np.abs(simAlt - data_dicts[wflyer]['quiet_alts'][0][tmeIdx]).argmin()
            fit_array[LShell_idx][alt_idx].append(data_dicts[wflyer]['quiet_ni'][0][tmeIdx])

    fit_num_points = np.zeros(shape=(len(fit_array), len(fit_array[0])))
    for iLat_idx in range(len(fit_array)):
        for alt_idx in range(len(fit_array[0])):
            fit_num_points[iLat_idx][alt_idx] = len(fit_array[iLat_idx][alt_idx])

    if bool_show_num_points_plot:
        fig, ax = plt.subplots()
        cmap = ax.pcolormesh(LShell_bins, simAlt/stl.m_to_km, fit_num_points.T, norm='log')
        ax.set_ylabel('Alt [km]')
        ax.set_xlabel('L Shell')
        ax.set_ylim(0, 420)
        ax.set_xlim(6.7, 10.5)
        cbar = plt.colorbar(cmap)
        cbar.set_label('Num of Points')
        ax.grid(which='both')
        ax.set_yticks(simAlt/stl.m_to_km)
        ax.set_xticks(LShell_bins)
        plt.show()

    ############################
    # --- FIT THE DATA ARRAY ---
    ############################
    stl.prgMsg('Linearly Fitting each altitude')
    # for each altitude in the fit_array, fit a linear line to ni = m (ILat) + b
    # evaluate that line over all latitude values, and choose altitudes up to the maximum value of the high flyer

    density_fitGrid_T = np.zeros(shape=(len(simAlt),len(simLShell)))

    for alt_idx in range(len(simAlt)):

        # get the data to fit
        density_vals = [fit_array[i][alt_idx] for i in range(len(LShell_bins))]
        LShell_vals = [[LShell_bins[idx] for val in range(len(lst))] for idx, lst in enumerate(density_vals) if len(lst) >= 1]
        density_vals = [val for sublist in density_vals for val in sublist]

        if len(LShell_vals) == 1:
            density_fitGrid_T[alt_idx] = [np.mean(density_vals) for i in range(len(simLShell))]
        elif len(LShell_vals) > 1:
            LShell_vals = [val for sublist in LShell_vals for val in sublist]

            # fit the data at this specific altitude
            params, cov = curve_fit(linear, LShell_vals, density_vals)

            # evaluate the fit at a specific region
            density_fitGrid_T[alt_idx] = linear(simLShell, *params)

    density_fitGrid_T[np.where(density_fitGrid_T == 0.0)[0]] = np.nan
    density_fitGrid = density_fitGrid_T.T
    stl.Done(start_time)

    ###################################
    # --- INTERPOLATE OVER ALTITUDE ---
    ###################################
    # For each L-Shell, cubic interpolate over altitude onto the simulation altitude range
    # This handles anywhere where there's data dropout. Effective if 1-2 datapoints missing
    stl.prgMsg('Interpolating over altitude')

    for idx1, LShell in enumerate(simLShell):
        density = density_fitGrid[idx1]
        good_indicies = np.where(density>0)[0]
        yData = density[good_indicies]
        xData = simAlt[good_indicies]
        cs = CubicSpline(xData, yData)
        data_dict_output['ni_spectrum'][0][idx1] = cs(simAlt)
    #
    #     fig, ax = plt.subplots()
    #     ax.scatter(xData/1000, yData, s=100)
    #     ax.scatter(simAlt/1000, cs(simAlt), color='red', s=30)
    #     ax.plot(simAlt / 1000, cs(simAlt))
    #     ax.set_yscale('log')
    #     plt.show()

    stl.Done(start_time)


    if bool_output_data:

        for key in data_dict_output.keys():
            data_dict_output[key][1]['DEPEND_0'] = None
            data_dict_output[key][1]['var_type'] = 'data'

        output_folder = 'C:\Data\physicsModels\ionosphere\plasma_environment\ACESII_ni_spectrum'
        stl.outputCDFdata(outputPath=output_folder + '\\ACESII_ni_spectrum.cdf', data_dict=data_dict_output)


langmuir_ni_spectrogram()