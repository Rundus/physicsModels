# --- --- --- --- --- --- --
# --- DERPA_Te_spectrum ---
# --- --- --- --- --- --- --=
# Description: get the DERPA data and fit/interpolate it over the simulation range
import matplotlib.pyplot as plt

# --- IMPORTS ---
import numpy as np
import spaceToolsLib as stl
import datetime as dt
from scipy.signal import savgol_filter
from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
from glob import glob
from src.ionosphere_modelling.sim_toggles import SimToggles
from scipy.interpolate import CubicSpline
from copy import deepcopy
from scipy.stats import binned_statistic

#################
# --- TOGGLES ---
#################
outputData = False
plot_fit_data = False
plot_EISCAT_background = False
bool_show_num_points_plot = True


def DERPA_Te_spectrum():

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_LP_cal_high = stl.loadDictFromFile(glob(f'C:/Data/ACESII/calibration/LP/postFlight_calibration/high/*.cdf*')[0])
    data_dict_LP_cal_low = stl.loadDictFromFile(glob(f'C:/Data/ACESII/calibration/LP/postFlight_calibration/low/*.cdf*')[0])
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\geomagneticField\*.cdf*')[0])

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    simAlt = (SpatialToggles.simAlt)/stl.m_to_km
    simLShell = SpatialToggles.simLShell
    data_dict_output = {}

    ##########################################
    # --- REDUCE DERPA TO SIMULATION RANGE ---
    ##########################################
    data_dicts = [data_dict_LP_cal_high, data_dict_LP_cal_low]

    #################################################
    # --- AVERAGE THE DERPA TEMPERATURES TOGETHER ---
    #################################################
    temps_avg = [np.nanmean(np.array([data_dicts[rkt_idx]['Te_DERPA1'][0],data_dicts[rkt_idx]['Te_DERPA2'][0]]).T,axis=1) for rkt_idx in range(2)]

    if plot_fit_data:
        fig, ax = plt.subplots()
        ax.scatter(x=data_dicts[0]['L-Shell'][0], y=data_dicts[0]['Alt'][0], c=temps_avg[0],s=80)
        ax.scatter(x=data_dicts[1]['L-Shell'][0], y=data_dicts[1]['Alt'][0], c=temps_avg[1],s=80)
        ax.set_ylabel('Alt [km]', fontsize=20)
        ax.set_xlabel('L Shell', fontsize=20)
        ax.set_ylim(130, 420)
        ax.set_xlim(8.1, 9.65)
        ax.grid()
        plt.show()

    #####################################
    # --- CONSTRUCT EISCAT BACKGROUND ---
    #####################################

    # find the peak in the altitude --> we will only use the upleg
    peak_alt_point = [np.argmax(data_dicts[i]['Alt'][0]) for i in range(2)]
    peak_alt_idxs = [np.abs(data_dicts[i]['Alt'][0][0:peak_alt_point[i]]-310).argmin() for i in range(2)] # only get data up to ~300km

    # reduce EISCAT data to upleg region below 300 km
    upleg_alt = [data_dicts[i]['Alt'][0][0:peak_alt_idxs[i]] for i in range(2)]
    upleg_Ti = [data_dicts[i]['Ti'][0][0:peak_alt_idxs[i]] for i in range(2)]
    upleg_Te = [data_dicts[i]['Te'][0][0:peak_alt_idxs[i]] for i in range(2)]
    upleg_ne = [data_dicts[i]['ne'][0][0:peak_alt_idxs[i]] for i in range(2)]
    upleg_Tr = [data_dicts[i]['Tr'][0][0:peak_alt_idxs[i]] for i in range(2)]

    # smooth each of these curves and evaluate them on the simulation altitudes
    def running_mean_filter(data):
        k = 5
        N = len(data)
        new_data = np.zeros_like(data)
        for i in range(k,N-k):
            new_data[i] = np.mean(data[i-k:i+k+1])
        return new_data

    if plot_EISCAT_background:

        fig, ax = plt.subplots(4)
        colors= ['tab:blue','tab:red']
        for i in range(2):
            ax[0].plot(upleg_alt[i], running_mean_filter(upleg_Ti[i]),label='Ti',color=colors[i])
            ax[1].plot(upleg_alt[i], running_mean_filter(upleg_Te[i]), label='Te',color=colors[i])
            ax[2].plot(upleg_alt[i], running_mean_filter(upleg_ne[i]), label='ne',color=colors[i])
            ax[3].plot(upleg_alt[i], running_mean_filter(upleg_Tr[i]), label='Tr',color=colors[i])

        for j in range(4):
            ax[j].legend()
        plt.show()

    # Construct the simulation output background variables from the High Flyer LP calibration files
    cs_Te = CubicSpline(upleg_alt[0],upleg_Te[0])
    cs_Ti = CubicSpline(upleg_alt[0], upleg_Ti[0])
    cs_Tr = CubicSpline(upleg_alt[0], upleg_Tr[0])
    cs_ne = CubicSpline(upleg_alt[0], upleg_ne[0])

    Te_background_grid = np.zeros(shape=(len(simLShell),len(simAlt)))
    Ti_background_grid = np.zeros(shape=(len(simLShell), len(simAlt)))
    Tr_background_grid = np.zeros(shape=(len(simLShell), len(simAlt)))
    ne_background_grid = np.zeros(shape=(len(simLShell), len(simAlt)))

    for i in range(len(simLShell)):
        Te_background_grid[i] = cs_Te(simAlt)
        Ti_background_grid[i] = cs_Ti(simAlt)
        Tr_background_grid[i] = cs_Tr(simAlt)
        ne_background_grid[i] = cs_ne(simAlt)

    ######################################
    # --- CONSTRUCT DERPA PERTURBATION ---
    ######################################

    # reduce data to only simulation region
    for wflyer in range(2):
        low_idx, high_idx = np.abs(data_dicts[wflyer]['L-Shell'][0] - simLShell[0]).argmin(), np.abs(data_dicts[wflyer]['L-Shell'][0] - simLShell[-1]).argmin()
        for key, val in data_dicts[wflyer].items():
            if key not in ['L-Shell']:
                data_dicts[wflyer][key][0] = deepcopy(data_dicts[wflyer][key][0][low_idx:high_idx+1])
        data_dicts[wflyer]['L-Shell'][0] = deepcopy(data_dicts[wflyer]['L-Shell'][0][low_idx:high_idx+1])

    # Construct an L-shell vs altitude array for fitting to the grid
    alt_grid = np.linspace(70, 420, 2)
    fit_array = [[[] for alt in alt_grid] for LShell in simLShell]

    # populate the fitting array with both flyer's data
    for wflyer in range(2):
        for tmeIdx in range(len(data_dicts[wflyer]['Epoch'][0])):
            LShell_idx = np.abs(simLShell - data_dicts[wflyer]['L-Shell'][0][tmeIdx]).argmin()
            alt_idx = np.abs(alt_grid - data_dicts[wflyer]['Alt'][0][tmeIdx]).argmin()
            fit_array[LShell_idx][alt_idx].append(np.nanmean([data_dicts[wflyer]['Te_DERPA1'][0][tmeIdx],data_dicts[wflyer]['Te_DERPA2'][0][tmeIdx]]))

    fit_num_points = np.zeros(shape=(len(simLShell),len(alt_grid)))
    for iLat_idx in range(len(fit_array)):
        for alt_idx in range(len(fit_array[0])):
            fit_num_points[iLat_idx][alt_idx] = len(fit_array[iLat_idx][alt_idx])

    if bool_show_num_points_plot:
        fig, ax = plt.subplots()
        cmap = ax.pcolormesh(simLShell, alt_grid, fit_num_points.T, norm='log',vmin=1,vmax=100)
        ax.set_ylabel('Alt [km]', fontsize=20)
        ax.set_xlabel('L Shell', fontsize=20)
        ax.set_ylim(70, 420)
        ax.set_xlim(simLShell[0], simLShell[-1])
        cbar = plt.colorbar(cmap)
        cbar.set_label('Num of Points')
        ax.grid()
        plt.show()


    #####################
    # --- OUTPUT DATA ---
    #####################
    if outputData:
        data_dict_output = {
            'Te_background':[(stl.q0/stl.kB)*np.array(Te_background_grid), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'K', 'LABLAXIS': 'Te'}],
            'Ti_background': [(stl.q0/stl.kB)*np.array(Ti_background_grid), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'K', 'LABLAXIS': 'Ti'}],
            'Tr_background': [np.array(Tr_background_grid), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'Tr'}],
            'ne_background': [np.array(ne_background_grid), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm^-3', 'LABLAXIS': 'ne'}],
            'simLShell': deepcopy(data_dict_Bgeo['simLShell']),
            'simAlt':deepcopy(data_dict_Bgeo['simAlt'])
        }
        output_path = 'C:/Data/physicsModels/ionosphere/plasma_environment/DERPA_Te_spectrum/DERPA_Te_spectrum.cdf'
        stl.outputDataDict(outputPath=output_path, data_dict=data_dict_output)

DERPA_Te_spectrum()