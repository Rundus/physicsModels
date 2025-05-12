def generate_Currents():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.currents.currents_toggles import CurrentsToggles

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_EField = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\electricField\*.cdf*')[0])
    data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\conductivity\*.cdf*')[0])

    ###########################################
    # --- GENERATE THE IONOSPHERIC CURRENTS ---
    ###########################################
    # prepare the data

    ped_current_1 = -1*data_dict_conductivity['sigma_Pedersen'][0]*data_dict_EField['deltaE_arc_normal'][0]/data_dict_spatial['grid_deltaX'][0] # - Sigma_P * Grad_perp.E_perp
    ped_current_2 = -1*data_dict_EField['E_arc_normal'][0]*data_dict_conductivity['delta_sigma_P_normal'][0]/data_dict_spatial['grid_deltaX'][0] # - E_perp . (Grad_perp Sigma_P)
    hall_current = data_dict_EField['E_arc_tangent'][0]*data_dict_conductivity['delta_sigma_H_normal'][0]/data_dict_spatial['grid_deltaX'][0] # unit_b.[Grad_perpSigma_H x E_perp]

    parallel_current = ped_current_1 + ped_current_2 + hall_current

    # --- Construct the output Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'ped_current_1': [ped_current_1, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'A/m^2', 'LABLAXIS': 'ped_current_1', 'VAR_TYPE': 'data'}],
                            'ped_current_2': [ped_current_2, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'A/m^2', 'LABLAXIS': 'ped_current_2', 'VAR_TYPE': 'data'}],
                            'hall_current': [hall_current, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'hall_current', 'VAR_TYPE': 'data'}],
                            'parallel_current': [parallel_current, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'parallel_current', 'VAR_TYPE': 'data'}]
                        }}

    outputPath = rf'{CurrentsToggles.outputFolder}\currents.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
