import datetime as dt
import numpy as np
import spaceToolsLib as stl
from copy import deepcopy
from src.ionosphere_modelling.data_preparation.data_preparation_toggles import DataPreparationToggles
from src.ionosphere_modelling.sim_toggles import SimToggles



class SpatialToggles:

    # some minimum times
    target_time = dt.datetime(2022, 11, 20, 17, 20)

    # --- Altitude Grid ---
    sim_alt_low = 70 * stl.m_to_km  # low altitude (in meters)
    sim_alt_high = 300 * stl.m_to_km  # high altitude (in meters)
    alt_rez = 2 * stl.m_to_km  # number of points in the altitude grid
    simAlt = np.linspace(sim_alt_low, sim_alt_high, int((sim_alt_high - sim_alt_low) / alt_rez + 1))  # in METERS

    # --- LShell Grid ---
    # Description: USE the HF L-shell attitude data to generate an L-Shell grid. Choose all L-SHells above a threshold altitude
    altThresh = 350*stl.m_to_km # get the HF attitude data for altitudes above this value [in km]
    data_dict_eepaa_high_ds = stl.loadDictFromFile(f'{SimToggles.sim_root_path}/data_inputs/eepaa/high/ACESII_36359_eepaa_downsampled_{DataPreparationToggles.N_avg}.cdf')
    idxs = np.where(data_dict_eepaa_high_ds['Alt'][0] > altThresh)[0]
    low_idx, high_idx = idxs[0], idxs[-1]
    simLShell = deepcopy(data_dict_eepaa_high_ds['L-Shell'][0][low_idx:high_idx+1])

    # --- geomagnetic Longitude Grid ---
    # Used to ensure the simulated R.O.I.is about right
    simGeomLong = deepcopy(data_dict_eepaa_high_ds['Long_geom'][0][low_idx:high_idx+1])
    outputFolder = f'{SimToggles.sim_root_path}/spatial_environment'