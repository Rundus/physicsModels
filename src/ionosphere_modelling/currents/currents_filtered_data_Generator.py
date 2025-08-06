
def generate_filtered_data_for_currents():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.currents.currents_toggles import CurrentsToggles
    import gc

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_EField = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\electricField\*.cdf*')[0])
    data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\conductivity\*.cdf*')[0])
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]

    #####################################################
    # --- PERFORM THE SSA ON E-FIELD AND CONDUCTIVITY ---
    #####################################################
    E_N_DC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_N_AC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    sigma_P_DC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))
    sigma_P_AC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))

    # Loop over altitude
    for j in tqdm(range(len(altRange))):

        if j == 0: # case when you're at the bottom of the simulation i.e. where there's no varition
            E_N_DC[:, j] = deepcopy(data_dict_EField['E_N'][0][:, j])
            E_N_AC[:, j] = deepcopy(data_dict_EField['E_N'][0][:, j])
            sigma_P_DC[:, j] = deepcopy(data_dict_conductivity['sigma_P'][0][:, j])
            sigma_P_AC[:, j] = deepcopy(data_dict_conductivity['sigma_P'][0][:, j])

        else:
            # --- SSA the E-Field ---
            SSA_E = stl.SSA(tseries=data_dict_EField['E_N'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
            E_N_DC[:, j] = deepcopy(SSA_E.reconstruct(indices=CurrentsToggles.DC_components))
            E_N_AC[:, j] = deepcopy(SSA_E.reconstruct(indices=CurrentsToggles.AC_components))
            del SSA_E

            # --- SSA the conductivity ---
            SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_P'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
            sigma_P_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.DC_components))
            sigma_P_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.AC_components))
            del SSA_sigma
            gc.collect()

    data_dict_output = {**data_dict_spatial,
                      **{ 'E_N_DC':[E_N_DC, deepcopy(data_dict_EField['E_N'][1])],
                          'E_N_AC': [E_N_AC, deepcopy(data_dict_EField['E_N'][1])],
                          'sigma_P_DC':[sigma_P_DC, deepcopy(data_dict_conductivity['sigma_P'][1])],
                           'sigma_P_AC': [sigma_P_AC, deepcopy(data_dict_conductivity['sigma_P'][1])]}
                      }

    outputPath = rf'{CurrentsToggles.outputFolder}\ssa_filtered_data\ssa_filtered_data.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)

generate_filtered_data_for_currents()