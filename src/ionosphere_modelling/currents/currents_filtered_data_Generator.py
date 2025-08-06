
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
    altRange = data_dict_spatial['simAlt'][0]

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    E_N_DC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_N_AC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_p_DC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    E_p_AC = np.zeros(shape=np.shape(data_dict_EField['E_N'][0]))
    sigma_P_DC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))
    sigma_P_AC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))
    sigma_H_DC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))
    sigma_H_AC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))
    sigma_D_DC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))
    sigma_D_AC = np.zeros(shape=np.shape(data_dict_conductivity['sigma_P'][0]))

    ###############################
    # --- PERFORM BOXCAR FILTER ---
    ###############################
    # Note: ONLY provides filtered DC components
    if CurrentsToggles.use_boxcar:
        from scipy.signal import savgol_filter
        for j in tqdm(range(len(altRange))):
            # E_N
            E_N_DC[:, j] = np.convolve(data_dict_EField['E_N'][0][:, j], np.ones(CurrentsToggles.N_boxcar)/CurrentsToggles.N_boxcar, mode='same')

            # E_p
            E_p_DC[:, j] = np.convolve(data_dict_EField['E_p'][0][:, j], np.ones(CurrentsToggles.N_boxcar)/CurrentsToggles.N_boxcar, mode='same')

            # sigma_P
            sigma_P_DC[:, j] = np.convolve(data_dict_conductivity['sigma_P'][0][:, j], np.ones(CurrentsToggles.N_boxcar)/CurrentsToggles.N_boxcar, mode='same')

            # sigma_H
            sigma_H_DC[:, j] = np.convolve(data_dict_conductivity['sigma_H'][0][:, j], np.ones(CurrentsToggles.N_boxcar)/CurrentsToggles.N_boxcar, mode='same')

            # sigma_D
            sigma_D_DC[:, j] = np.convolve(data_dict_conductivity['sigma_D'][0][:, j], np.ones(CurrentsToggles.N_boxcar)/CurrentsToggles.N_boxcar, mode='same')

    #####################################
    # --- PERFORM SAVITZ-GOLAY FILTER ---
    #####################################
    # Note: ONLY provides filtered AC components

    if CurrentsToggles.use_savitz_golay:
        from scipy.signal import savgol_filter
        for j in tqdm(range(len(altRange))):
            # smooth electric field data

            # E_N
            E_N_DC[:, j] = savgol_filter(x=data_dict_EField['E_N'][0][:, j],
                                                             window_length=CurrentsToggles.window,
                                                             polyorder=CurrentsToggles.polyorder)
            # E_p
            E_p_DC[:, j] = savgol_filter(x=data_dict_EField['E_p'][0][:, j],
                                                             window_length=CurrentsToggles.window,
                                                             polyorder=CurrentsToggles.polyorder)

            # smooth conductivity data
            # sigma_P
            sigma_P_AC[:, j] = savgol_filter(x=data_dict_conductivity['sigma_P'][0][:, j],
                                                                       window_length=CurrentsToggles.window,
                                                                       polyorder=CurrentsToggles.polyorder)

            # sigma_H
            sigma_H_DC[:, j] = savgol_filter(x=data_dict_conductivity['sigma_H'][0][:, j],
                                             window_length=CurrentsToggles.window,
                                             polyorder=CurrentsToggles.polyorder)

            # sigma_D
            sigma_D_DC[:, j] = savgol_filter(x=data_dict_conductivity['sigma_D'][0][:, j],
                                                                       window_length=CurrentsToggles.window,
                                                                       polyorder=CurrentsToggles.polyorder)



    ################################
    # --- PERFORM THE SSA FILTER ---
    ################################
    if CurrentsToggles.use_SSA_filter:

        # Loop over altitude
        for j in tqdm(range(len(altRange))):

            if j == 0: # case when you're at the bottom of the simulation i.e. where there's no variation

                # --- Electric Fields ---
                E_N_DC[:, j] = deepcopy(data_dict_EField['E_N'][0][:, j])
                E_N_AC[:, j] = deepcopy(data_dict_EField['E_N'][0][:, j])

                E_p_DC[:, j] = deepcopy(data_dict_EField['E_p'][0][:, j])
                E_p_AC[:, j] = deepcopy(data_dict_EField['E_p'][0][:, j])

                # --- Conductivities ---
                sigma_P_DC[:, j] = deepcopy(data_dict_conductivity['sigma_P'][0][:, j])
                sigma_P_AC[:, j] = deepcopy(data_dict_conductivity['sigma_P'][0][:, j])

                sigma_H_DC[:, j] = deepcopy(data_dict_conductivity['sigma_H'][0][:, j])
                sigma_H_AC[:, j] = deepcopy(data_dict_conductivity['sigma_H'][0][:, j])

                sigma_D_DC[:, j] = deepcopy(data_dict_conductivity['sigma_D'][0][:, j])
                sigma_D_AC[:, j] = deepcopy(data_dict_conductivity['sigma_D'][0][:, j])

            else:

                # --- SSA the E-Field(s) ---

                SSA_E = stl.SSA(tseries=data_dict_EField['E_N'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
                E_N_DC[:, j] = deepcopy(SSA_E.reconstruct(indices=CurrentsToggles.DC_components))
                E_N_AC[:, j] = deepcopy(SSA_E.reconstruct(indices=CurrentsToggles.AC_components))
                del SSA_E
                gc.collect()

                SSA_E = stl.SSA(tseries=data_dict_EField['E_p'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
                E_p_DC[:, j] = deepcopy(SSA_E.reconstruct(indices=CurrentsToggles.DC_components))
                E_p_AC[:, j] = deepcopy(SSA_E.reconstruct(indices=CurrentsToggles.AC_components))
                del SSA_E
                gc.collect()

                # --- SSA the conductivity ---
                SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_P'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
                sigma_P_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.DC_components))
                sigma_P_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.AC_components))
                del SSA_sigma
                gc.collect()

                SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_H'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
                sigma_H_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.DC_components))
                sigma_H_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.AC_components))
                del SSA_sigma
                gc.collect()

                SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_D'][0][:, j], L=CurrentsToggles.wLen, mirror_percent=CurrentsToggles.mirror_percent)
                sigma_D_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.DC_components))
                sigma_D_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=CurrentsToggles.AC_components))
                del SSA_sigma
                gc.collect()

    ############################
    # --- WRITE OUT THE DATA ---
    ############################
    data_dict_output = {**data_dict_spatial,
                      **{ 'E_N_DC':[E_N_DC, deepcopy(data_dict_EField['E_N'][1])],
                          'E_N_AC': [E_N_AC, deepcopy(data_dict_EField['E_N'][1])],

                          'E_p_DC':[E_p_DC, deepcopy(data_dict_EField['E_p'][1])],
                          'E_p_AC': [E_p_AC, deepcopy(data_dict_EField['E_p'][1])],

                          'sigma_P_DC':[sigma_P_DC, deepcopy(data_dict_conductivity['sigma_P'][1])],
                          'sigma_P_AC': [sigma_P_AC, deepcopy(data_dict_conductivity['sigma_P'][1])],

                          'sigma_H_DC': [sigma_H_DC, deepcopy(data_dict_conductivity['sigma_H'][1])],
                          'sigma_H_AC': [sigma_H_AC, deepcopy(data_dict_conductivity['sigma_H'][1])],

                          'sigma_D_DC': [sigma_H_DC, deepcopy(data_dict_conductivity['sigma_D'][1])],
                          'sigma_D_AC': [sigma_H_AC, deepcopy(data_dict_conductivity['sigma_D'][1])]
                          }
                      }

    outputPath = rf'{CurrentsToggles.outputFolder}\{CurrentsToggles.filter_path}\filtered_EFields_conductivity.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)

generate_filtered_data_for_currents()