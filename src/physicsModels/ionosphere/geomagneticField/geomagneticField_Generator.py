# --- imports ---
import numpy
from datetime import datetime
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import spaceToolsLib as stl
import numpy as np


def generate_GeomagneticField():

    # import the toggles
    from src.physicsModels.ionosphere.simToggles_Ionosphere import BgeoToggles, SpatialToggles
    altRange = SpatialToggles.simAlt
    LShellRange = SpatialToggles.simLShell


    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\spatialEnvironment\spatialEnvironment.cdf')

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial



    geomagAlts = [((alt + (stl.Re*stl.m_to_km)) / (stl.Re*stl.m_to_km)) for alt in altRange]
    geomagLats = np.array([np.degrees(np.arccos(radi / BgeoToggles.Lshell)) for radi in geomagAlts])
    geomagLongs = np.array([111.83 for i in range(len(altRange))])
    times = [datetime(2022, 11, 20, 17, 20, 00, 000) for i in range(len(altRange))]
    Pos = np.array([geomagAlts, geomagLats, geomagLongs]).transpose()
    ISOtime = [times[i].isoformat() for i in range(len(times))]
    cvals_MAG = coord.Coords(Pos, 'MAG', 'sph')
    cvals_MAG.ticks = Ticktock(ISOtime, 'ISO')
    cvals_GDZ = cvals_MAG.convert('GEO', 'sph')
    Lat_geo = cvals_GDZ.lati

    # Get the Chaos model
    B = stl.CHAOS(Lat_geo, [15.25 for i in range(len(altRange))], np.array(altRange) / m_to_km, times)
    Bgeo = (1E-9) * np.array([np.linalg.norm(Bvec) for Bvec in B])
    Bgrad = [(Bgeo[i + 1] - Bgeo[i]) / (altRange[i + 1] - altRange[i]) for i in range(len(Bgeo) - 1)]
    Bgrad = np.array(Bgrad + [Bgrad[-1]]) # add the high altitude value Bgrad again to model the starting point (MAYBE it SHOULD BE 0?)


    # --- Construct the Data Dict ---
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}

    data_dict_output = { **data_dict_output,
                         **{'Bgeo': [Bgeo, {'DEPEND_0':'simAlt','DEPEND_1':'simLShell', 'UNITS':'T', 'LABLAXIS': 'Bgeo'}],
                  'Bgrad': [Bgrad, {'DEPEND_0':'simAlt','DEPEND_1':'simLShell', 'UNITS':'T', 'LABLAXIS': 'Bgrad'}],
                  'simAlt': [altRange, {'DEPEND_0':'simAlt', 'UNITS':'m', 'LABLAXIS': 'simAlt'}],
                  'simLShell': [LShellRange, {'DEPEND_0':'simLShell', 'UNITS':None, 'LABLAXIS': 'LShell'}]
                 }}

    outputPath = rf'{BgeoToggles.outputFolder}\geomagneticfield.cdf'
    stl.outputCDFdata(outputPath, data_dict)
