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
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]
    grid_Bgeo = np.zeros(shape=(len(LShellRange),len(altRange)))
    grid_Bgrad = np.zeros(shape=(len(LShellRange), len(altRange)))








    # --- Construct the output Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'Bgeo': [grid_Bgeo, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'T', 'LABLAXIS': 'Bgeo', 'VAR_TYPE': 'data'}],
                            'Bgrad': [grid_Bgrad, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'T', 'LABLAXIS': 'Bgrad', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{CurrentsToggles.outputFolder}\currents.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
