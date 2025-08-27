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
    from src.ionosphere_modelling.electricField.electricField_toggles import EFieldToggles
    from src.ionosphere_modelling.currents.currents_toggles import CurrentsToggles

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])

    if CurrentsToggles.filter_data:
        data_dict_EField = deepcopy(stl.loadDictFromFile(rf'{CurrentsToggles.outputFolder}\{CurrentsToggles.filter_path}\filtered_EFields_conductivity.cdf')) # collect the IRI data
        data_dict_conductivity = deepcopy(stl.loadDictFromFile(rf'{CurrentsToggles.outputFolder}\{CurrentsToggles.filter_path}\filtered_EFields_conductivity.cdf')) # collect the IRI data
    else:
        data_dict_EField = deepcopy(stl.loadDictFromFile(rf'{EFieldToggles.outputFolder}\electric_Field.cdf'))  # collect the IRI data
        data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\conductivity\*.cdf*')[0])



    #################################
    # --- CALCULATE JOULE HEATING ---
    #################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    grid_Joule_DC = data_dict_conductivity['sigma_P_DC'][0]*np.power(data_dict_EField['E_N_DC'][0], 2)
    grid_Joule_AC = data_dict_conductivity['sigma_P_AC'][0] * np.power(data_dict_EField['E_N_AC'][0], 2)

    # --- Construct the Data Dict ---
    data_dict_output = {
                        'simLShell': deepcopy(data_dict_spatial['simLShell']),
                        'simAlt': deepcopy(data_dict_spatial['simAlt']),
                         'q_j_DC': [grid_Joule_DC, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'W/m!A3', 'LABLAXIS': 'DC Joule Heating Rate', 'VAR_TYPE': 'data','SCALEMIN':1E-9,'SCALEMAX':1E-6}],
                         'q_j_AC': [grid_Joule_AC, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A3', 'LABLAXIS': 'AC Joule Heating Rate', 'VAR_TYPE': 'data', 'SCALEMIN': 1E-9, 'SCALEMAX': 1E-6}],
                         }

    outputPath = rf'{JouleHeatingToggles.outputFolder}\joule_heating.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
