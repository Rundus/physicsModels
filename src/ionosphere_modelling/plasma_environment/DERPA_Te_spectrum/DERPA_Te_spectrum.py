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


#################
# --- TOGGLES ---
#################
outputData = False
plot_fit_data = False
plot_EISCAT_fit = True

def DERPA_Te_spectrum():

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_ERPA1_high = stl.loadDictFromFile(r'C:\Data\ACESII\L2\high\ACESII_36359_l2_ERPA1.cdf')
    data_dict_ERPA2_high = stl.loadDictFromFile(r'C:\Data\ACESII\L2\high\ACESII_36359_l2_ERPA2.cdf')
    data_dict_ERPA1_low = stl.loadDictFromFile(r'C:\Data\ACESII\L2\low\ACESII_36364_l2_ERPA1.cdf')
    data_dict_ERPA2_low = stl.loadDictFromFile(r'C:\Data\ACESII\L2\low\ACESII_36364_l2_ERPA2.cdf')
    data_dict_EISCAT = stl.loadDictFromFile(r'C:\Data\ACESII\science\EISCAT\tromso\UHF\MAD6400_2022-11-20_beata_ant@uhfa.cdf')

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    simAlt = (SpatialToggles.simAlt)/stl.m_to_km
    simLShell = SpatialToggles.simLShell
    data_dict_output = {}

    ##########################################
    # --- REDUCE DERPA TO SIMULATION RANGE ---
    ##########################################
    data_dicts = [data_dict_ERPA1_high,data_dict_ERPA2_high, data_dict_ERPA1_low,data_dict_ERPA2_low]
    for idx, dddict in enumerate(data_dicts):
        low_idx = np.abs(dddict['L-Shell'][0] - simLShell[0]).argmin()
        high_idx = np.abs(dddict['L-Shell'][0]- simLShell[-1]).argmin()
        for key in ['Epoch', 'L-Shell', 'temperature', 'Alt']:
            data_dicts[idx][f"{key}"][0] = data_dicts[idx][f"{key}"][0][low_idx:high_idx+1]

    ###########################################
    # --- AVERAGE THE TEMPERATURES TOGETHER ---
    ###########################################
    temp_high = np.zeros(shape=len(data_dict_ERPA1_high['Epoch'][0]))
    for tme in range(len(data_dict_ERPA1_high['Epoch'][0])):
        temp_high[tme] = np.nanmean([data_dict_ERPA1_high['temperature'][0][tme], data_dict_ERPA2_high['temperature'][0][tme]])

    temp_low = np.zeros(shape=len(data_dict_ERPA1_low['Epoch'][0]))
    for tme in range(len(data_dict_ERPA1_low['Epoch'][0])):
        temp_low[tme] = np.nanmean([data_dict_ERPA1_low['temperature'][0][tme], data_dict_ERPA2_low['temperature'][0][tme]])

    if plot_fit_data:
        fig, ax = plt.subplots()
        ax.scatter(x=data_dict_ERPA1_high['L-Shell'][0], y=data_dict_ERPA1_high['Alt'][0]/stl.m_to_km, c=temp_high,s=80)
        ax.scatter(x=data_dict_ERPA1_low['L-Shell'][0], y=data_dict_ERPA1_low['Alt'][0]/stl.m_to_km, c=temp_low,s=80)
        ax.set_ylabel('Alt [km]', fontsize=20)
        ax.set_xlabel('L Shell', fontsize=20)
        ax.set_ylim(130, 420)
        ax.set_xlim(8.1, 9.65)
        ax.grid()
        plt.show()

    #############################
    # --- GET THE EISCAT DATA ---
    #############################
    # Description: The EISCAT data can be used to get the Te during the parts where ACES-II did NOT measure it <170 km

    # EISCAT alternating time jumps 45s, 75sec
    target_times = [dt.datetime(2022,11,20,17,21,25),
                    # dt.datetime(2022,11,20,17,21,25)+dt.timedelta(seconds=45),
                    dt.datetime(2022,11,20,17,21,25)+dt.timedelta(seconds=1*75+1*45),
                    # dt.datetime(2022,11,20,17,21,25)+dt.timedelta(seconds=2*45+1*75),
                    dt.datetime(2022,11,20,17,21,25)+dt.timedelta(seconds=2*45+2*75),
                    # dt.datetime(2022,11,20,17,21,25)+dt.timedelta(seconds=3*45+2*75),
                    dt.datetime(2022,11,20,17,21,25)+dt.timedelta(seconds=3*45+3*75)
                    ]
    # target_times = [dt.datetime(2022, 11, 20, 17, 21, 25)]

    Te = data_dict_EISCAT['tr'][0] *data_dict_EISCAT['ti'][0]
    idxs = [np.abs(data_dict_EISCAT['Epoch'][0] - tme).argmin() for tme in target_times]
    Te = Te[idxs]

    if plot_EISCAT_fit:
        fig, ax = plt.subplots()
        Te_mean = np.nanmean(Te,axis=0)
        Te_mean_nonan = []
        Range_nonan = []
        for idx in range(len(Te_mean)):
            if np.isnan(Te_mean[idx]) == False:
                Te_mean_nonan.append(Te_mean[idx])
                Range_nonan.append(data_dict_EISCAT['range'][0][idx])
        Te_mean = Te_mean_nonan
        print(np.shape(Te_mean))
        ax.scatter(x=Range_nonan, y=Te_mean, s=30)
        filtered_data = savgol_filter(x=Te_mean,window_length=150,polyorder=3)
        filtfilt_data = savgol_filter(x=filtered_data, window_length=160, polyorder=3)
        ax.plot(Range_nonan,filtfilt_data,color='red')
        ax.set_xlabel('Alt [km]', fontsize=20)
        ax.set_ylabel('Te', fontsize=20)
        ax.set_xlim(60, 400)
        ax.set_ylim(0,3000)
        ax.grid()
        plt.show()


    #####################
    # --- OUTPUT DATA ---
    #####################
    if outputData:
        output_folder = 'C:\Data\physicsModels\ionosphere\plasma_environment\DERPA_Te_spectrum\DERPA_Te_spectrum'
        stl.outputCDFdata(outputPath=output_folder + '\\DERPA_Te_spectrum.cdf', data_dict=data_dict_output)

DERPA_Te_spectrum()