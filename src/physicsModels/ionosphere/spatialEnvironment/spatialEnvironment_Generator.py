# --- imports ---
import numpy
from datetime import datetime
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import spaceToolsLib as stl
import numpy as np


def generate_spatialEnvironment():
    # import the toggles
    from src.physicsModels.ionosphere.simToggles_Ionosphere import SpatialToggles
    altRange = SpatialToggles.simAlt
    LShellRange = SpatialToggles.simLShell

    # prepare the output
    data_dict_output = {}

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial
    grid_lat = np.zeros(shape=(len(LShellRange),len(altRange)))
    grid_long = np.zeros(shape=(len(LShellRange),len(altRange)))
    grid_alt = np.array([altRange for i in range(len(LShellRange))])
    grid_LShell = np.array([[LShellRange[i] for k in range(len(altRange))] for i in range(len(LShellRange))])

    for Lval, idx in enumerate(LShellRange):

        # get the geomagnetic coordinate of the P.O.I. based on L-Shell
        geomagAlts = [((alt + (stl.Re * stl.m_to_km)) / (stl.Re * stl.m_to_km)) for alt in altRange]
        geomagLats = np.array([np.degrees(np.arccos(radi / Lval)) for radi in geomagAlts])
        geomagLongs = np.array([111.83 for i in range(len(altRange))])
        times = [datetime(2022, 11, 20, 17, 20, 00, 000) for i in range(len(altRange))]

        # Convert to geographic coordinates
        Pos = np.array([geomagAlts, geomagLats, geomagLongs]).transpose()
        ISOtime = [times[i].isoformat() for i in range(len(times))]
        cvals_MAG = coord.Coords(Pos, 'MAG', 'sph')
        cvals_MAG.ticks = Ticktock(ISOtime, 'ISO')
        cvals_GDZ = cvals_MAG.convert('GEO', 'sph')

        Lat_geo = cvals_GDZ.lati
        Long_geo = cvals_GDZ.long

        print(Long_geo)
        print(Long_geo)

        # store the data
        grid_lat[idx] = Lat_geo
        grid_long[idx] = Long_geo



    # --- Construct the output data dict ---
    data_dict_output = {**data_dict_output,
                        **{
                           'simAlt' : [altRange, {'UNITS': 'm', 'LABLAXIS': 'simAlt'}],
                            'simLShell': [altRange, {'UNITS': None, 'LABLAXIS': 'simLShell'}],
                           'grid_lat': [grid_lat, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'deg', 'LABLAXIS': 'latitude'}],
                            'grid_alt': [grid_alt, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'Altitude'}],
                           'grid_LShell': [grid_LShell, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'LShell'}],
                            'grid_long': [grid_long, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'deg', 'LABLAXIS': 'Longitude'}],
                           }}

    outputPath = rf'{SpatialToggles.outputFolder}\spatialEnvironment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
