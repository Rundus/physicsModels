import itertools


def generatePoyntingFlux():
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
    from src.ionosphere_modelling.electricField.electricField_toggles import EFieldToggles
    from src.ionosphere_modelling.poynting_flux.poynting_flux_toggles import poyntingFluxToggles

    #######################
    # --- LOAD THE DATA ---
    #######################
    # get the geomagnetic field data dict
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\geomagneticField\*.cdf*')[0])

    if CurrentsToggles.filter_data:
        data_dict_EField = deepcopy(stl.loadDictFromFile(rf'{CurrentsToggles.outputFolder}\{CurrentsToggles.filter_path}\filtered_EFields_conductivity.cdf')) # collect the IRI data
    else:
        data_dict_EField = deepcopy(stl.loadDictFromFile(rf'{EFieldToggles.outputFolder}\electric_Field.cdf'))  # collect the IRI data


    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    LShellRange = data_dict_Bgeo['simLShell'][0]
    altRange = data_dict_Bgeo['simAlt'][0]

    #####################################
    # --- CALCULATE THE POYNTING FLUX ---
    #####################################

    S_DC = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    S_AC = np.zeros(shape=(len(LShellRange), len(altRange), 3))

    ranges = [range(len(LShellRange)),range(len(altRange))]
    for i, j in tqdm(itertools.product(*ranges)):

        # DC Fields (N,T,p)
        B_vec = np.array([ 0, 0, data_dict_Bgeo['Bgeo'][0][i][j]])
        E_vec = np.array([data_dict_EField['E_N_DC'][0][i][j], 0, data_dict_EField['E_p_DC'][0][i][j]  ])
        S_DC[i][j] = np.cross(E_vec,B_vec)/stl.u0

        # AC Fields (N,T,p)
        B_vec = np.array([0, 0, data_dict_Bgeo['Bgeo'][0][i][j]])
        E_vec = np.array([data_dict_EField['E_N_AC'][0][i][j], 0, data_dict_EField['E_p_AC'][0][i][j]])
        S_AC[i][j] = np.cross(E_vec, B_vec) / stl.u0

    #####################
    # --- OUTPUT DATA ---
    #####################

    data_dict_output = {
                        'simLShell': deepcopy(data_dict_Bgeo['simLShell']),
                        'simAlt': deepcopy(data_dict_Bgeo['simAlt']),
                        'S_N_DC': [S_DC[:, :, 0], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A2', 'LABLAXIS': 'arc-Normal DC Poynting Flux', 'VAR_TYPE': 'data'}],
                        'S_T_DC': [S_DC[:, :, 1], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A2', 'LABLAXIS': 'arc-Normal DC Poynting Flux', 'VAR_TYPE': 'data'}],
                        'S_p_DC': [S_DC[:, :, 2], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A2', 'LABLAXIS': 'Field Aligned DC Poynting Flux', 'VAR_TYPE': 'data'}],
                        'S_N_AC': [S_DC[:, :, 0], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A2', 'LABLAXIS': 'arc-Normal AC Poynting Flux', 'VAR_TYPE': 'data'}],
                        'S_T_AC': [S_DC[:, :, 1], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A2', 'LABLAXIS': 'arc-Normal AC Poynting Flux', 'VAR_TYPE': 'data'}],
                        'S_p_AC': [S_DC[:, :, 2], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'W/m!A2', 'LABLAXIS': 'Field Aligned AC Poynting Flux', 'VAR_TYPE': 'data'}],
                        }

    outputPath = rf'{poyntingFluxToggles.outputFolder}\poynting_flux.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
