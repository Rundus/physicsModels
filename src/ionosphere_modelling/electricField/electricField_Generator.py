def generate_electricField():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.electricField.electricField_toggles import EFieldToggles
    from spacepy import pycdf
    import datetime as dt
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock  # used to determine the time I'm choosing the reference geomagentic field

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_potential = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\electrostaticPotential\electrostaticPotential.cdf')
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]

    ##############################################
    # --- CALCULATE AURORAL COORDINATE E-Field ---
    ##############################################
    grid_E_T = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_E_N = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_E_p = np.zeros(shape=(len(LShellRange), len(altRange)))

    grid_E_T_detrend = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_E_N_detrend = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_E_p_detrend = np.zeros(shape=(len(LShellRange), len(altRange)))

    potential_detrend = deepcopy(data_dict_potential['potential_detrend'][0])
    potential = deepcopy(data_dict_potential['potential'][0])

    # Normal
    for j in range(len(altRange)):
        # Calculate E = -Grad(Phi)
        grid_E_N[:, j] = -1 * np.gradient(potential[:, j], data_dict_spatial['grid_dN'][0][:, j])
        grid_E_N_detrend[:, j] = -1 * np.gradient(potential_detrend[:, j], data_dict_spatial['grid_dN'][0][:, j])

    # Vertical
    for i in range(len(LShellRange)):
        # Calculate E = -Grad(Phi)
        grid_E_p[i, :] = -1 * np.gradient(potential[i, :], data_dict_spatial['grid_dp'][0][i,:])
        grid_E_p_detrend[i, :] = -1 * np.gradient(potential_detrend[i,:], data_dict_spatial['grid_dp'][0][i,:])

    # --- Construct the Data Dict ---
    data_dict_output = {**data_dict_spatial,
                        **{
                            'E_N': [grid_E_N, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_N', 'VAR_TYPE': 'data'}],
                            'E_T': [grid_E_T, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_T', 'VAR_TYPE': 'data'}],
                            'E_p': [grid_E_p, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_p', 'VAR_TYPE': 'data'}],

                            'E_N_detrend': [grid_E_N_detrend, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_N', 'VAR_TYPE': 'data'}],
                            'E_T_detrend': [grid_E_T_detrend, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_T', 'VAR_TYPE': 'data'}],
                            'E_p_detrend': [grid_E_p_detrend, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_p', 'VAR_TYPE': 'data'}],
                        }}

    ###############################
    # --- INCLUDE NEUTRAL WINDS ---
    ###############################
    if EFieldToggles.include_neutral_winds:
        # load neutral wind data
        data_dict_neutral = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\neutral_environment\*.cdf*')[0])

        # form the ENU neutral wind vector
        neutral_wind_ENU = np.zeros(shape=(len(LShellRange), len(altRange), 3))
        neutral_wind_ENU[:, :, 0] = deepcopy(data_dict_neutral['meridional_wind'][0])
        neutral_wind_ENU[:, :, 1] = deepcopy(data_dict_neutral['zonal_wind'][0])

        # transform the ENU neutral wind data into auroral
        grid_neutral_wind_auroral = np.zeros(shape=(len(LShellRange), len(altRange), 3))
        grid_UxB_auroral = np.zeros(shape=(len(LShellRange), len(altRange), 3))
        for i in tqdm(range(len(LShellRange))):
            for j in range(len(altRange)):
                grid_neutral_wind_auroral[i][j] = np.matmul(data_dict_spatial['grid_ENU_to_Auroral_transform'][0][i][j], neutral_wind_ENU[i][j])
                grid_UxB_auroral[i][j] = np.cross(grid_neutral_wind_auroral[i][j], np.array([0,0,1]))

        data_dict_output ={**data_dict_output,
                           **{
                               'U_N': [grid_neutral_wind_auroral[:, :, 0], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'm/s', 'LABLAXIS': 'Arc-Normal Neutral Wind', 'VAR_TYPE': 'data'}],
                               'U_T': [grid_neutral_wind_auroral[:, :, 1], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'm/s', 'LABLAXIS': 'Arc-Tangent Neutral Wind', 'VAR_TYPE': 'data'}],
                               'U_p': [grid_neutral_wind_auroral[:, :, 2], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'm/s', 'LABLAXIS': 'Field Aligned Neutral Wind', 'VAR_TYPE': 'data'}],
                               'UxB_N': [grid_UxB_auroral[:, :, 0], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Arc-Normal UxB', 'VAR_TYPE': 'data'}],
                               'UxB_T': [grid_UxB_auroral[:, :, 1], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Arc-Tangent UxB', 'VAR_TYPE': 'data'}],
                               'UxB_p': [grid_UxB_auroral[:, :, 2], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Field Aligned UxB', 'VAR_TYPE': 'data'}],
                              }
                           }

    # outputPath = rf'{EFieldToggles.outputFolder}\electric_Field.cdf'
    # stl.outputCDFdata(outputPath, data_dict_output)
    #
    # # --- Construct the Data Dict ---
    # # data_dict_output = { **data_dict_spatial,
    # #                      **{
    # #                          'E_N': [grid_E_N, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_N', 'VAR_TYPE': 'data'}],
    # #                          'E_T': [grid_E_T, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_T','VAR_TYPE': 'data'}],
    # #                          'E_p': [grid_E_p, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_p', 'VAR_TYPE': 'data'}],
    # #                     }}

    if EFieldToggles.include_neutral_winds:
        data_dict_output ={**data_dict_output,
                           **{
                               'U_N': [grid_neutral_wind_auroral[:, :, 0], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'm/s', 'LABLAXIS': 'Arc-Normal Neutral Wind', 'VAR_TYPE': 'data'}],
                               'U_T': [grid_neutral_wind_auroral[:, :, 1], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'm/s', 'LABLAXIS': 'Arc-Tangent Neutral Wind', 'VAR_TYPE': 'data'}],
                               'U_p': [grid_neutral_wind_auroral[:, :, 2], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'm/s', 'LABLAXIS': 'Field Aligned Neutral Wind', 'VAR_TYPE': 'data'}],
                               'UxB_N': [grid_UxB_auroral[:, :, 0], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Arc-Normal UxB', 'VAR_TYPE': 'data'}],
                               'UxB_T': [grid_UxB_auroral[:, :, 1], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Arc-Tangent UxB', 'VAR_TYPE': 'data'}],
                               'UxB_p': [grid_UxB_auroral[:, :, 2], {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Field Aligned UxB', 'VAR_TYPE': 'data'}],
                              }
                           }

    outputPath = rf'{EFieldToggles.outputFolder}\electric_Field.cdf'
    stl.outputDataDict(outputPath, data_dict_output)
