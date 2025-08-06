def generate_spatialEnvironment():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    import datetime as dt
    import math
    from geopy import distance


    # import the toggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    altRange = SpatialToggles.simAlt
    LShellRange = SpatialToggles.simLShell
    LongGeomRange = SpatialToggles.simGeomLong

    # prepare the output
    data_dict_output = {}

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial
    grid_lat = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_long = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_alt = np.array([altRange for i in range(len(LShellRange))]) # in meters
    grid_LShell = np.array([LShellRange for k in range(len(altRange))]).T

    for idx, Lval in tqdm(enumerate(LShellRange)):

        # get the geomagnetic coordinate of the P.O.I. based on L-Shell
        geomagAlts = [((alt + stl.Re * stl.m_to_km) / (stl.Re * stl.m_to_km)) for alt in altRange]
        geomagLats = np.array([np.degrees(np.arccos(np.sqrt(radi / Lval))) for radi in geomagAlts])
        geomagLongs = np.array([LongGeomRange[idx] for i in range(len(altRange))])
        times = [SpatialToggles.target_time for i in range(len(altRange))]

        # Convert to geographic coordinates
        Pos = np.array([geomagAlts, geomagLats, geomagLongs]).transpose()
        ISOtime = [times[i].isoformat() for i in range(len(times))]
        cvals_MAG = coord.Coords(Pos, 'MAG', 'sph')
        cvals_MAG.ticks = Ticktock(ISOtime, 'ISO')
        cvals_GDZ = cvals_MAG.convert('GEO', 'sph')

        # store the data
        grid_lat[idx] = cvals_GDZ.lati
        grid_long[idx] = cvals_GDZ.long

    # --- CALCULATE spatial gradient distances (deltaX, deltaZ) on our spatial grid---
    # Note: Since the simulation is 2D, there's no deltaY
    grid_deltaZ = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_deltaX = np.zeros(shape=(len(LShellRange), len(altRange)))

    for idx, Lval in tqdm(enumerate(LShellRange)):
        for idx_z, alt in enumerate(altRange): # DeltaZ grid:

            # --- Calculate deltaAlt grid ---
            if idx_z == len(altRange)-1: # if we're at the top of the simulation region, just repeat the previous value
                # Get the points
                p1 = (grid_lat[idx][idx_z-1], grid_long[idx][idx_z-1], grid_alt[idx][idx_z-1] / stl.m_to_km)
                p2 = (grid_lat[idx][idx_z], grid_long[idx][idx_z], grid_alt[idx][idx_z] / stl.m_to_km)
            else:
                # Get the points
                p1 = (grid_lat[idx][idx_z], grid_long[idx][idx_z], grid_alt[idx][idx_z]/stl.m_to_km)
                p2 = (grid_lat[idx][idx_z+1], grid_long[idx][idx_z+1], grid_alt[idx][idx_z+1]/stl.m_to_km)

            flat_distance = distance.distance(p1[:2], p2[:2]).km
            euclidian_distance = math.sqrt(flat_distance ** 2 + (p2[2] - p1[2]) ** 2)
            grid_deltaZ[idx][idx_z] = euclidian_distance*stl.m_to_km

    for idx, Lval in enumerate(LShellRange): # DeltaX grid:
        for idx_z, alt in enumerate(altRange):
            if idx == len(LShellRange)-1:
                # Get the points
                p1 = (grid_lat[idx-1][idx_z], grid_long[idx-1][idx_z], grid_alt[idx-1][idx_z] / stl.m_to_km)
                p2 = (grid_lat[idx][idx_z], grid_long[idx][idx_z], grid_alt[idx][idx_z] / stl.m_to_km)
            else:
                # Get the points
                p1 = (grid_lat[idx][idx_z], grid_long[idx][idx_z], grid_alt[idx][idx_z] / stl.m_to_km)
                p2 = (grid_lat[idx+1][idx_z], grid_long[idx+1][idx_z], grid_alt[idx+1][idx_z] / stl.m_to_km)

            flat_distance = distance.distance(p1[:2], p2[:2]).km
            euclidian_distance = math.sqrt(flat_distance ** 2 + (p2[2] - p1[2]) ** 2)
            grid_deltaX[idx][idx_z] = euclidian_distance*stl.m_to_km

    # --- Construct the output data dict ---
    data_dict_output = {**data_dict_output,
                        **{
                            'simAlt': [altRange, {'UNITS': 'm', 'LABLAXIS': 'simAlt'}],
                            'simLShell': [LShellRange, {'UNITS': None, 'LABLAXIS': 'simLShell'}],
                            'grid_lat': [grid_lat, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'deg', 'LABLAXIS': 'latitude'}],
                            'grid_alt': [grid_alt, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'Altitude'}],
                            'grid_LShell': [grid_LShell, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'LShell'}],
                            'grid_long': [grid_long, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'deg', 'LABLAXIS': 'Longitude'}],
                            'grid_dz': [grid_deltaZ, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'Vertical Gradient'}],
                            'grid_dx': [grid_deltaX, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'Horizontal Gradient'}],
                           }}

    outputPath = rf'{SpatialToggles.outputFolder}\spatial_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
