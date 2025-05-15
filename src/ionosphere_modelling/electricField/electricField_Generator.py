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

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_EFI = stl.loadDictFromFile(r'C:\Data\ACESII\science\auroral_coordinates\low\ACESII_36364_E_Field_Auroral_Coordinates.cdf')


    ########################################
    # --- Filter/Smooth the E-Field Data ---
    ########################################
    deltaT = (pycdf.lib.datetime_to_tt2000(data_dict_EFI['Epoch'][0][5001]) - pycdf.lib.datetime_to_tt2000(data_dict_EFI['Epoch'][0][5000])) / 1E9

    # Filter params
    order = 4
    fs = 1 / deltaT
    freq_cut = 0.05
    filt_type = 'lowpass'
    data_dict_EFI['E_tangent'][0] = deepcopy(stl.butter_filter(
        data=data_dict_EFI['E_tangent'][0],
        lowcutoff=freq_cut,
        highcutoff=freq_cut,
        filtertype=filt_type,
        fs=fs,
        order=order
    ))

    data_dict_EFI['E_normal'][0] = deepcopy(stl.butter_filter(
        data=data_dict_EFI['E_normal'][0],
        lowcutoff=freq_cut,
        highcutoff=freq_cut,
        filtertype=filt_type,
        fs=fs,
        order=order
    ))

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    grid_EField_normal = np.zeros(shape=(len(LShellRange), len(altRange))) # grid of E_N projected
    grid_EField_tangent = np.zeros(shape=(len(LShellRange), len(altRange)))  # grid of E_N projected
    grid_deltaE_N = np.zeros(shape=(len(LShellRange), len(altRange))) # gradient in the E_N component in the normal direction
    grid_deltaE_Alt = np.zeros(shape=(len(LShellRange), len(altRange))) # gradient in the E_N componet in the vertical direction

    for idx, Lval in tqdm(enumerate(LShellRange)):
        Lshell_idx = np.abs(data_dict_EFI['L-Shell'][0] - Lval).argmin()
        grid_EField_normal[idx] = np.array([(1E-3)*data_dict_EFI['E_normal'][0][Lshell_idx] for i in range(len(altRange))])
        grid_EField_tangent[idx] = np.array([(1E-3) * data_dict_EFI['E_tangent'][0][Lshell_idx] for i in range(len(altRange))])


    # # --- Gradient in Electric Field (vertical), should be about zero ---
    # for idx, val in enumerate(LShellRange):
    #     dz= data_dict_spatial['grid_deltaAlt'][0][idx]
    #     grid_deltaE_Alt[idx] = np.gradient(grid_EField_normal[idx],dz)
    #
    # # --- Gradient in Electric Field (horizontal) ---
    # for idx, val in enumerate(altRange):
    #     dx = data_dict_spatial['grid_deltaX'][0][:,idx]
    #     grid_deltaE_N[:,idx] = np.gradient(grid_EField_normal[:,idx], dx)

    #
    # --- Gradient in Electric Field (Lshell) ---
    for idx, Lval in tqdm(enumerate(LShellRange)):
        for idx_z in range(len(SpatialToggles.simAlt)):

            # Calculate deltaE along altitude axis (*Should be all zeros at the moment)
            if idx_z == len(SpatialToggles.simAlt)-1:
                grid_deltaE_Alt[idx][idx_z] = grid_EField_normal[idx][idx_z] - grid_EField_normal[idx][idx_z-1]
            else:
                grid_deltaE_Alt[idx][idx_z] = grid_EField_normal[idx][idx_z+1] - grid_EField_normal[idx][idx_z]

            # Calculate deltaE along L-Shell axis
            if idx == len(SpatialToggles.simLShell)-1:
                grid_deltaE_N[idx][idx_z] = grid_EField_normal[idx][idx_z] - grid_EField_normal[idx-1][idx_z]
            else:
                grid_deltaE_N[idx][idx_z] = grid_EField_normal[idx+1][idx_z] - grid_EField_normal[idx][idx_z]


    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'E_N': [grid_EField_normal, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V/m', 'LABLAXIS': 'Arc-Normal Electric Field', 'VAR_TYPE': 'data'}],
                            'E_T': [grid_EField_tangent, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'Arc-Tangent Electric Field', 'VAR_TYPE': 'data'}],
                            'dE_N_normal': [grid_deltaE_N, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'deltaE Arc-Normal Electric Field', 'VAR_TYPE': 'data'}],
                            'dE_N_vertical': [grid_deltaE_Alt, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'deltaE Vertical Electric Field', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{EFieldToggles.outputFolder}\electricField.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
