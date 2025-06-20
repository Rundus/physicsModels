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
    from scipy.signal import savgol_filter

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_potential = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\electrostaticPotential\electrostaticPotential.cdf')

    ###########################
    # --- CALCULATE E-FIELD ---
    ###########################

    # [1] Calculate the magnitude of the E_X, E_Z components in the SIMULATED coordinate system
    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]

    # calculate the vertical electric field
    grid_EField_Z = np.zeros(shape=(len(LShellRange), len(altRange)))  # grid of E_N projected

    for idx in range(len(LShellRange)):
        # determine the iterative sum of the vertical position
        gradients = deepcopy(data_dict_spatial['grid_deltaAlt'][0][idx])
        initial_point = gradients[0]
        position_points = np.array([np.sum(gradients[0:i+1])-initial_point for i in range(len(gradients))])
        grid_EField_Z[idx] = -1*np.gradient(data_dict_potential['potential'][0][idx],position_points)

    # calculate the horizontal gradient
    grid_EField_X = np.zeros(shape=(len(LShellRange), len(altRange)))  # grid of E_N projected

    for idx in range(len(altRange)):
        # determine the iterative sum of the horizontal position
        gradients = deepcopy(data_dict_spatial['grid_deltaX'][0][:,idx])
        initial_point = gradients[0]
        position_points = np.array([np.sum(gradients[0:i + 1]) - initial_point for i in range(len(gradients))])
        grid_EField_X[:,idx] = -1*np.gradient(data_dict_potential['potential'][0][:,idx], position_points)


    # [2] Calculate the ECEF coordinates of the simulation Grid



    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{
                            'E_X': [grid_EField_X, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V/m', 'LABLAXIS': 'sim X-dir Electric Field', 'VAR_TYPE': 'data'}],
                            'E_Z': [grid_EField_Z, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'sim Z-dir Electric Field', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{EFieldToggles.outputFolder}\electric_Field.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
