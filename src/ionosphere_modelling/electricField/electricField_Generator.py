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

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    grid_EField_normal = np.zeros(shape=(len(LShellRange), len(altRange))) # grid of E_N projected
    grid_deltaE_N = np.zeros(shape=(len(LShellRange), len(altRange))) # gradient in the E_N component in the normal direction
    grid_deltaE_Alt = np.zeros(shape=(len(LShellRange), len(altRange))) # gradient in the E_N componet in the vertical direction

    # --- CALCULATE E-Field FROM POTENTIAL ---
    # TODO: Deal with this filtering/noise effect later
    # Just filter the data for now to make it look clean.
    for idx, altval in enumerate(altRange):
        temp = deepcopy(np.gradient(data_dict_potential['potential'][0][:,idx])/(-1*data_dict_spatial['grid_deltaX'][0][:,idx]))
        # filtered = savgol_filter(temp,window_length=50,polyorder=2)
        grid_EField_normal[:,idx] = temp

    # # --- Gradient in Electric Field (vertical), should be about zero ---
    # for idx, val in enumerate(LShellRange):
    #     dz= data_dict_spatial['grid_deltaAlt'][0][idx]
    #     grid_deltaE_Alt[idx] = np.gradient(grid_EField_normal[idx],dz)
    #
    # # --- Gradient in Electric Field (horizontal) ---
    # for idx, val in enumerate(altRange):
    #     dx = data_dict_spatial['grid_deltaX'][0][:,idx]
    #     grid_deltaE_N[:,idx] = np.gradient(grid_EField_normal[:,idx], dx)




    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{
                             'E_N': [grid_EField_normal, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V/m', 'LABLAXIS': 'Arc-Normal Electric Field', 'VAR_TYPE': 'data'}],
                            # 'E_T': [grid_EField_tangent, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Arc-Tangent Electric Field', 'VAR_TYPE': 'data'}],
                            'dE_N_normal': [grid_deltaE_N, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'deltaE Arc-Normal Electric Field', 'VAR_TYPE': 'data'}],
                            'dE_N_vertical': [grid_deltaE_Alt, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'deltaE Vertical Electric Field', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{EFieldToggles.outputFolder}\electric_Field.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
