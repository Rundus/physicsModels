import datetime as dt
import numpy as np
import spaceToolsLib as stl
from copy import deepcopy

class SpatialToggles:

    # some minimum times
    target_time = dt.datetime(2022, 11, 20, 17, 20)
    target_time_simulation_minimum = dt.datetime(2022, 11, 20, 17, 24, 2,664000) # minimum time of the simulation (determines minimum L-Shell)

    # --- Altitude Grid ---
    sim_alt_low = 80 * stl.m_to_km  # low altitude (in meters)
    sim_alt_high = 300 * stl.m_to_km  # high altitude (in meters)
    # alt_rez = 1 * stl.m_to_km  # number of points in the altitude grid
    alt_rez = 2 * stl.m_to_km  # number of points in the altitude grid
    simAlt = np.linspace(sim_alt_low, sim_alt_high, int((sim_alt_high - sim_alt_low) / alt_rez + 1))  # in METERS

    # --- LShell Grid ---
    # Description: USE the HF L-shell attitude data to generate an L-Shell grid. Choose all L-SHells above a threshold altitude
    altThresh = 250*stl.m_to_km # get the HF attitude data for altitudes above this value [in km]
    data_dict_eepaa_high_ds = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\data_inputs\eepaa\high\ACESII_36359_eepaa_downsampled_3.cdf')
    Epoch_min_idx = np.abs(data_dict_eepaa_high_ds['Epoch'][0] - target_time_simulation_minimum).argmin()
    data_dict_eepaa_high_ds['L-Shell'][0] = deepcopy(data_dict_eepaa_high_ds['L-Shell'][0][Epoch_min_idx:])
    data_dict_eepaa_high_ds['Alt'][0] = deepcopy(data_dict_eepaa_high_ds['Alt'][0][Epoch_min_idx:])
    simLShell = deepcopy(data_dict_eepaa_high_ds['L-Shell'][0][np.where(data_dict_eepaa_high_ds['Alt'][0] >= altThresh)[0]])

    # --- geomagnetic Longitude Grid ---
    # Used to ensure the simulated R.O.I.is about right
    simGeomLong = deepcopy(data_dict_eepaa_high_ds['Long_geom'][0][np.where(data_dict_eepaa_high_ds['Alt'][0] >= altThresh)[0]])
    outputFolder = 'C:\Data\physicsModels\ionosphere\spatial_environment'