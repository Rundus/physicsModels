import datetime as dt
import numpy as np
import spaceToolsLib as stl
from copy import deepcopy

class SpatialToggles:

    target_time = dt.datetime(2022, 11, 20, 17, 20)

    # --- Altitude Grid ---
    sim_alt_low = 80 * stl.m_to_km  # low altitude (in meters)
    sim_alt_high = 300 * stl.m_to_km  # high altitude (in meters)
    # alt_rez = 1 * stl.m_to_km  # number of points in the altitude grid
    alt_rez = 2 * stl.m_to_km  # number of points in the altitude grid
    simAlt = np.linspace(sim_alt_low, sim_alt_high, int((sim_alt_high - sim_alt_low) / alt_rez + 1))  # in METERS

    # --- LShell Grid ---
    # Description: USE the HF L-shell attitude data to generate an L-Shell grid. Choose all L-SHells above a threshold altitude
    altThresh = 300*stl.m_to_km # get the HF attitude data for altitudes above this value [in km]
    # sim_Lshell_Low = 6.8
    # sim_Lshell_High = 10.5
    # LShell_rez = 0.002 # there are 8659 records between 70ILat to 73.5 ILat on the HF. Choose an appropriate resolution.
    # LShell_rez = 0.02  # there are 8659 records between 70ILat to 73.5 ILat on the HF. Choose an appropriate resolution.
    # simLShell = np.linspace(sim_Lshell_Low, sim_Lshell_High, int((sim_Lshell_High - sim_Lshell_Low) / LShell_rez + 1))  # unitless
    data_dict_HF_eepaa = stl.loadDictFromFile('C:\Data\ACESII\L2\high\ACESII_36359_l2_eepaa_fullCal.cdf')
    data_dict_HF_LShell = stl.loadDictFromFile('C:\Data\ACESII\coordinates\Lshell\high\ACESII_36359_Lshell.cdf')
    simLShell = deepcopy(data_dict_HF_LShell['L-Shell'][0][np.where(data_dict_HF_eepaa['Alt'][0] >= altThresh)[0]])


    # geomagnetic Longitude Grid - Used to ensure the simulated R.O.I.is about right
    # sim_geomLong_low = 111.828
    # sim_geomLong_high = 117.1
    # simGeomLong = np.linspace(sim_geomLong_low, sim_geomLong_high, int((sim_geomLong_high - sim_geomLong_low) / LShell_rez + 1))  # in Degrees
    simGeomLong = deepcopy(data_dict_HF_eepaa['Long_geom'][0][np.where(data_dict_HF_eepaa['Alt'][0] >= altThresh)[0]])
    outputFolder = 'C:\Data\physicsModels\ionosphere\spatial_environment'


# data_dict_HF_eepaa = stl.loadDictFromFile('C:\Data\ACESII\L2\high\ACESII_36359_l2_eepaa_fullCal.cdf')
# data_dict_HF_LShell = stl.loadDictFromFile('C:\Data\ACESII\science\L_shell\high\ACESII_36359_Lshell.cdf')
# indices = np.where(data_dict_HF_eepaa['Alt'][0] >= 300000)[0]
#
# from spacepy import pycdf
# Epoch_tt2000 = np.array([pycdf.lib.datetime_to_tt2000(val)/1E9 for val in data_dict_HF_eepaa['Epoch'][0]])
# # arr = Epoch_tt2000
# arr = SpatialToggles.simLShell
# diff = np.array([arr[i+1] - arr[i] for i in range(len(arr)-2) ])
# print(SpatialToggles.simLShell)
# print(diff)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter( [i for i in range(len(diff))], diff)
# # ax.set_ylim(0.04,0.06)
# # ax.scatter([i for i in range(len(SpatialToggles.simLShell))],SpatialToggles.simLShell)
# plt.show()