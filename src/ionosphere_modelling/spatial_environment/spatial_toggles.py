import datetime as dt
import numpy as np
import spaceToolsLib as stl
from copy import deepcopy
from src.ionosphere_modelling.data_preparation.data_preparation_toggles import DataPreparationToggles

class SpatialToggles:

    # some minimum times
    target_time = dt.datetime(2022, 11, 20, 17, 20)

    # minimum L-Shell
    minimum_L_shell = 8.4

    # --- Altitude Grid ---
    sim_alt_low = 80 * stl.m_to_km  # low altitude (in meters)
    sim_alt_high = 300 * stl.m_to_km  # high altitude (in meters)
    alt_rez = 2 * stl.m_to_km  # number of points in the altitude grid
    simAlt = np.linspace(sim_alt_low, sim_alt_high, int((sim_alt_high - sim_alt_low) / alt_rez + 1))  # in METERS

    # --- LShell Grid ---
    # Description: USE the HF L-shell attitude data to generate an L-Shell grid. Choose all L-SHells above a threshold altitude
    altThresh = 210*stl.m_to_km # get the HF attitude data for altitudes above this value [in km]
    data_dict_eepaa_high_ds = stl.loadDictFromFile(f'C:\Data\physicsModels\ionosphere\data_inputs\eepaa\high\ACESII_36359_eepaa_downsampled_{DataPreparationToggles.N_avg}.cdf')
    min_idx = np.abs(data_dict_eepaa_high_ds['L-Shell'][0] - minimum_L_shell).argmin()
    data_dict_eepaa_high_ds['L-Shell'][0] = deepcopy(data_dict_eepaa_high_ds['L-Shell'][0][min_idx:])
    data_dict_eepaa_high_ds['Alt'][0] = deepcopy(data_dict_eepaa_high_ds['Alt'][0][min_idx:])
    simLShell = deepcopy(data_dict_eepaa_high_ds['L-Shell'][0][np.where(data_dict_eepaa_high_ds['Alt'][0] >= altThresh)[0]])

    # --- geomagnetic Longitude Grid ---
    # Used to ensure the simulated R.O.I.is about right
    simGeomLong = deepcopy(data_dict_eepaa_high_ds['Long_geom'][0][np.where(data_dict_eepaa_high_ds['Alt'][0] >= altThresh)[0]])
    outputFolder = 'C:\Data\physicsModels\ionosphere\spatial_environment'