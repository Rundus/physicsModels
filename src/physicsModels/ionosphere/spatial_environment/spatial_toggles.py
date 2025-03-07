import datetime as dt
import numpy as np
from spaceToolsLib import m_to_km

class SpatialToggles:

    target_time = dt.datetime(2022, 11, 20, 17, 20)

    # Altitude Grid
    sim_alt_low = 50 * m_to_km  # low altitude (in meters)
    sim_alt_high = 400 * m_to_km  # high altitude (in meters)
    alt_rez = 5 * m_to_km  # number of points in the altitude grid
    simAlt = np.linspace(sim_alt_low, sim_alt_high, int((sim_alt_high - sim_alt_low) / alt_rez + 1))  # in METERS

    # ILat Grid
    sim_Lshell_Low = 6.8
    sim_Lshell_High = 10.5
    LShell_rez = 0.002 # there are 8659 records between 70ILat to 73.5 ILat on the HF. Choose an appropriate resolution.
    simLShell = np.linspace(sim_Lshell_Low, sim_Lshell_High, int((sim_Lshell_High - sim_Lshell_Low) / LShell_rez + 1))  # unitless

    # geomagnetic Longitude Grid - Used to ensure the simulated R.O.I. is about right
    sim_geomLong_low = 111.828
    sim_geomLong_high = 117.1
    simGeomLong = np.linspace(sim_geomLong_low, sim_geomLong_high, int((sim_geomLong_high - sim_geomLong_low) / LShell_rez + 1))  # in Degrees

    outputFolder = 'C:\Data\physicsModels\ionosphere\spatial_environment'