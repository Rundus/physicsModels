def generate_JouleHeating():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.joule_heating.joule_heating_toggles import JouleHeatingToggles

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_EField = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\electricField\*.cdf*')[0])
    data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\conductivity\*.cdf*')[0])

    #################################
    # --- CALCULATE JOULE HEATING ---
    #################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    grid_Joule = data_dict_conductivity['sigma_P'][0]*np.power(data_dict_EField['E_N'][0],2)


    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'q_j': [grid_Joule, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'W/m!A-3', 'LABLAXIS': 'Joule Heating Rate', 'VAR_TYPE': 'data','SCALEMIN':1E-9,'SCALEMAX':1E-6}]}
                         }

    outputPath = rf'{JouleHeatingToggles.outputFolder}\jouleheating.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
