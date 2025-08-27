
def generate_filtered_EField():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.currents.currents_filter_toggles import FilterToggles
    from src.ionosphere_modelling.electricField.electricField_toggles import EFieldToggles
    import gc

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_EField = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\electricField\*.cdf*')[0])
    altRange = data_dict_spatial['simAlt'][0]

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    E_N_DC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_N_AC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_p_DC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_p_AC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))

    ###############################
    # --- PERFORM BOXCAR FILTER ---
    ###############################
    # Note: ONLY provides filtered DC components
    if FilterToggles.use_boxcar:
        from scipy.signal import savgol_filter
        for j in tqdm(range(len(altRange))):
            # E_N
            E_N_DC[:, j] = np.convolve(data_dict_EField['E_N'][0][:, j], np.ones(FilterToggles.N_boxcar)/FilterToggles.N_boxcar, mode='same')

            # E_p
            E_p_DC[:, j] = np.convolve(data_dict_EField['E_p'][0][:, j], np.ones(FilterToggles.N_boxcar)/FilterToggles.N_boxcar, mode='same')

    #####################################
    # --- PERFORM SAVITZ-GOLAY FILTER ---
    #####################################
    # Note: ONLY provides filtered AC components

    if FilterToggles.use_savitz_golay:
        from scipy.signal import savgol_filter
        for j in tqdm(range(len(altRange))):
            # smooth electric field data

            # E_N
            E_N_DC[:, j] = savgol_filter(x=data_dict_EField['E_N'][0][:, j],
                                                             window_length=FilterToggles.window,
                                                             polyorder=FilterToggles.polyorder)
            # E_p
            E_p_DC[:, j] = savgol_filter(x=data_dict_EField['E_p'][0][:, j],
                                                             window_length=FilterToggles.window,
                                                             polyorder=FilterToggles.polyorder)

    ################################
    # --- PERFORM THE SSA FILTER ---
    ################################
    if FilterToggles.use_SSA_filter:

        # Loop over altitude
        for j in tqdm(range(len(altRange))):

            if j == 0: # case when you're at the bottom of the simulation i.e. where there's no variation

                # --- Electric Fields ---
                E_N_DC[:, j] = deepcopy(data_dict_EField['E_N'][0][:, j])
                E_N_AC[:, j] = deepcopy(data_dict_EField['E_N'][0][:, j])

                E_p_DC[:, j] = deepcopy(data_dict_EField['E_p'][0][:, j])
                E_p_AC[:, j] = deepcopy(data_dict_EField['E_p'][0][:, j])

            else:

                # --- SSA the E-Field(s) ---

                SSA_E = stl.SSA(tseries=data_dict_EField['E_N'][0][:, j], L=FilterToggles.wLen, mirror_percent=FilterToggles.mirror_percent)
                E_N_DC[:, j] = deepcopy(SSA_E.reconstruct(indices=FilterToggles.DC_components))
                E_N_AC[:, j] = deepcopy(SSA_E.reconstruct(indices=FilterToggles.AC_components))
                del SSA_E
                gc.collect()

                SSA_E = stl.SSA(tseries=data_dict_EField['E_p'][0][:, j], L=FilterToggles.wLen, mirror_percent=FilterToggles.mirror_percent)
                E_p_DC[:, j] = deepcopy(SSA_E.reconstruct(indices=FilterToggles.DC_components))
                E_p_AC[:, j] = deepcopy(SSA_E.reconstruct(indices=FilterToggles.AC_components))
                del SSA_E
                gc.collect()

    ############################
    # --- WRITE OUT THE DATA ---
    ############################
    data_dict_output = {**data_dict_spatial,
                      **{
                          'E_N_DC':[E_N_DC, deepcopy(data_dict_EField['E_N'][1])],
                          'E_N_AC': [E_N_AC, deepcopy(data_dict_EField['E_N'][1])],

                          'E_T_DC':deepcopy(data_dict_EField['E_T']),
                          'E_T_AC': deepcopy(data_dict_EField['E_T']),

                          'E_p_DC': [E_p_DC, deepcopy(data_dict_EField['E_p'][1])],
                          'E_p_AC': [E_p_AC, deepcopy(data_dict_EField['E_p'][1])],
                          }
                      }

    outputPath = rf'{EFieldToggles.outputFolder}\{FilterToggles.filter_path}\filtered_EFields_conductivity.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)