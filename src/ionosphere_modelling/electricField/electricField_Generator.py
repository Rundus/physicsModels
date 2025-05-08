def generate_electricField():
    # --- imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm

    # import the toggles
    from src.ionosphere_modelling.electricField.electricField_toggles import EFieldToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\spatial_environment\spatial_environment.cdf')
    data_dict_EFI = stl.loadDictFromFile(r'C:\Data\ACESII\science\auroral_coordinates\low\ACESII_36364_E_Field_Auroral_Coordinates.cdf')

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    grid_EField = np.zeros(shape=(len(LShellRange), len(altRange)))

    for idx, Lval in tqdm(enumerate(LShellRange)):

        Lshell_idx = np.abs(data_dict_EFI['L-Shell'][0] - Lval).argmin()
        grid_EField[idx] = np.array([(1E-3)*data_dict_EFI['E_normal'][0][Lshell_idx] for i in range(len(altRange))])


    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'E_arc_normal': [grid_EField, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V/m', 'LABLAXIS': 'Arc-Normal Electric Field', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{EFieldToggles.outputFolder}\electricField.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
