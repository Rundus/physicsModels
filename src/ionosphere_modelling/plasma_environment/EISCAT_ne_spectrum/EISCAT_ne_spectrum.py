# --- --- --- --- --- --- --
# --- EISCAT_ne_spectrum ---
# --- --- --- --- --- --- --

# Description: Sample the EISCAT data at a specific region and get the average value vs.altitude and variance for
# each point. Filter and cubic spline the average curve in order to make an average background density.


# --- IMPORTS ---
import numpy as np
import spaceToolsLib as stl
import datetime as dt
from copy import deepcopy
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles


#################
# --- TOGGLES ---
#################
plot_average_ne_curve = True

# time_targets = [
#     dt.datetime(2022,11,20,17,4,10),
#     dt.datetime(2022,11,20,17,16,10)
# ]

time_targets = [
    dt.datetime(2022,11,20,16,20,00),
    dt.datetime(2022,11,20,16,50,00)
]


def EISCAT_ne_spectrum():

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_EISCAT = stl.loadDictFromFile(r'C:\Data\ACESII\science\EISCAT\tromso\UHF\MAD6400_2022-11-20_beata_ant@uhfa.cdf')
    data_dict_spatial = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\spatial_environment\spatial_environment.cdf')

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    simAlt = (SpatialToggles.simAlt)/stl.m_to_km
    simLShell = SpatialToggles.simLShell

    data_dict_output = {
        'ne_spectrum': [np.zeros(shape=(len(simLShell), len(simAlt))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'cm!A-3!N', 'LABLAXIS': 'ne_spectrum'}],
        'simLShell': deepcopy(data_dict_spatial['simLShell']),
        'simAlt': deepcopy(data_dict_spatial['simAlt']),
    }

    ##################################################
    # --- REDUCE EISCAT TO SPECIFIC ALT/TIME RANGE ---
    ##################################################
    tme_idx_low, tme_idx_high = np.abs(data_dict_EISCAT['Epoch'][0] - time_targets[0]).argmin(),np.abs(data_dict_EISCAT['Epoch'][0] - time_targets[1]).argmin()
    alt_idx_low, alt_idx_high = np.abs(data_dict_EISCAT['range'][0] - 70).argmin(),np.abs(data_dict_EISCAT['range'][0] - 300).argmin()
    EISCAT_density = data_dict_EISCAT['ne'][0][tme_idx_low:tme_idx_high+1][alt_idx_low:alt_idx_high+1]
    ISR_ne_mean = np.nanmean(EISCAT_density, axis=0)
    ISR_ne_std = np.nanstd(EISCAT_density, axis=0)
    ISR_alt = data_dict_EISCAT['range'][0]

    ##########################################################
    # --- BIN THE EISCAT DATA INTO THE SIMULATION ALTITUDE ---
    ##########################################################
    bin_indicies =np.digitize(x=ISR_alt, bins=simAlt, right=False)
    ne_avg = [[] for i in range(len(simAlt))]
    for i in range(len(ISR_alt)):
        if not np.isnan(ISR_ne_mean[i]):
            ne_avg[bin_indicies[i]-1].append(ISR_ne_mean[i])

    binned_ne_std = np.array([np.nanstd(arr) for arr in ne_avg])
    binned_ne_avg = np.array([np.nanmean(arr) for arr in ne_avg])

    #############################################
    # --- SAVGOL FILTER DATA TO SMOOTH IT OUT ---
    #############################################
    good_avg = binned_ne_avg[np.where(np.isnan(binned_ne_avg) == False)[0]]
    good_std = binned_ne_std[np.where(np.isnan(binned_ne_std) == False)[0]]
    good_alts = simAlt[np.where(np.isnan(binned_ne_avg) == False)[0]]
    filtered_binned_ne_avg = savgol_filter(good_avg, window_length=12, polyorder=3)

    ########################################
    # --- INTERPOLATE SAVGOL ONTO SIMALT ---
    ########################################
    cs = CubicSpline(good_alts, filtered_binned_ne_avg)
    ne_background = 1E-6*np.array(cs(simAlt)) # convert to cm^-3
    ne_background[ne_background < 0] = 0

    # find where data is below 70 km and set it equal to zero (or 1, which is basically zero but works on a log plot)
    zero_idx = np.abs(simAlt - 70).argmin()
    ne_background[:zero_idx] = 0

    ############################
    # --- FORMAT OUTPUT DATA ---
    ############################
    for idx in range(len(simLShell)):
        data_dict_output['ne_spectrum'][0][idx] = ne_background

    #####################
    # --- OUTPUT DATA ---
    #####################
    output_folder = 'C:\Data\physicsModels\ionosphere\plasma_environment\EISCAT_ne_spectrum'
    stl.outputCDFdata(outputPath=output_folder + '\\EISCAT_ne_spectrum.cdf', data_dict=data_dict_output)

    ###################
    # --- PLOT DATA ---
    ###################

    if plot_average_ne_curve:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # ax.scatter(ISR_alt, ISR_ne_mean,s=20, color='tab:blue')
        ax.errorbar(simAlt, 1E-6*binned_ne_avg, yerr=1E-6*binned_ne_std, capsize=2, color='tab:blue')
        ax.scatter(simAlt, 1E-6*binned_ne_avg,s=15,color='tab:blue')
        ax.plot(simAlt, ne_background,color='tab:red')
        ax.set_yscale('log')
        ax.set_ylim(1E2, 1E6)
        ax.set_xlim(50, 320)
        ax.set_xlabel('Alt [km]')
        ax.set_ylabel('ne [cm$^{-3}$]')
        plt.savefig('C:\Data\physicsModels\ionosphere\plasma_environment\EISCAT_ne_spectrum\EISCAT_ne_spectrum.png')

EISCAT_ne_spectrum()