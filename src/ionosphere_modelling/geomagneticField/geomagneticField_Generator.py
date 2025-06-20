def generate_GeomagneticField():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.geomagneticField.geomagneticField_toggles import BgeoToggles


    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    N = len(LShellRange)
    M =len(altRange)
    grid_Bgeo = np.zeros(shape=(len(LShellRange),len(altRange)))
    grid_Bgrad = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_Bgeo_ENU_E = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_Bgeo_ENU_N = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_Bgeo_ENU_U = np.zeros(shape=(len(LShellRange), len(altRange)))

    for idx, Lval in tqdm(enumerate(LShellRange)):

        lats = data_dict_spatial['grid_lat'][0][idx]
        alts = data_dict_spatial['grid_alt'][0][idx]
        longs = data_dict_spatial['grid_long'][0][idx]

        # Get the Chaos model
        B = stl.CHAOS(lats, longs, np.array(alts) / stl.m_to_km, [SpatialToggles.target_time for i in range(len(alts))])
        grid_Bgeo_ENU_E[idx] = B[:, 0]*1E-9
        grid_Bgeo_ENU_N[idx] = B[:, 1]*1E-9
        grid_Bgeo_ENU_U[idx] = B[:, 2]*1E-9
        Bgeo = (1E-9) * np.array([np.linalg.norm(Bvec) for Bvec in B])

        # store the data
        grid_Bgeo[idx] = Bgeo

    # calculate the geomagnetic gradient
    for idx in tqdm(range(N)):
        grid_Bgrad[idx] = np.gradient(grid_Bgeo[idx], deepcopy(data_dict_spatial['grid_alt'][0][idx]))


    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'Bgeo': [grid_Bgeo, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'T', 'LABLAXIS': 'Bgeo', 'VAR_TYPE': 'data'}],
                            'Bgrad': [grid_Bgrad, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'T/m', 'LABLAXIS': 'Bgrad', 'VAR_TYPE': 'data'}],
                            'B_E': [grid_Bgeo_ENU_E, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'T', 'LABLAXIS': 'B_E', 'VAR_TYPE': 'data'}],
                            'B_N': [grid_Bgeo_ENU_N, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'T', 'LABLAXIS': 'B_N','VAR_TYPE': 'data'}],
                            'B_U': [grid_Bgeo_ENU_U,{'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'T', 'LABLAXIS': 'B_U', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{BgeoToggles.outputFolder}\geomagneticfield.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
