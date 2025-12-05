
def generate_filtered_conductivity():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.filtering.filter_toggles import FilterToggles
    from src.ionosphere_modelling.conductivity.conductivity_toggles import ConductivityToggles
    import gc

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}/spatial_environment/*.cdf*')[0])
    data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}/conductivity/*.cdf*')[0])
    altRange = data_dict_spatial['simAlt'][0]

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
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
    if FilterToggles.use_boxcar:
        from scipy.signal import savgol_filter
        for j in tqdm(range(len(altRange))):
            # sigma_P
            sigma_P_DC[:, j] = np.convolve(data_dict_conductivity['sigma_P'][0][:, j], np.ones(FilterToggles.N_boxcar)/FilterToggles.N_boxcar, mode='same')

            # sigma_H
            sigma_H_DC[:, j] = np.convolve(data_dict_conductivity['sigma_H'][0][:, j], np.ones(FilterToggles.N_boxcar)/FilterToggles.N_boxcar, mode='same')

            # sigma_D
            sigma_D_DC[:, j] = np.convolve(data_dict_conductivity['sigma_D'][0][:, j], np.ones(FilterToggles.N_boxcar)/FilterToggles.N_boxcar, mode='same')

    #####################################
    # --- PERFORM SAVITZ-GOLAY FILTER ---
    #####################################
    # Note: ONLY provides filtered AC components

    if FilterToggles.use_savitz_golay:
        from scipy.signal import savgol_filter
        for j in tqdm(range(len(altRange))):
            # smooth electric field data

            # smooth conductivity data
            # sigma_P
            sigma_P_AC[:, j] = savgol_filter(x=data_dict_conductivity['sigma_P'][0][:, j],
                                                                       window_length=FilterToggles.window,
                                                                       polyorder=FilterToggles.polyorder)

            # sigma_H
            sigma_H_DC[:, j] = savgol_filter(x=data_dict_conductivity['sigma_H'][0][:, j],
                                             window_length=FilterToggles.window,
                                             polyorder=FilterToggles.polyorder)

            # sigma_D
            sigma_D_DC[:, j] = savgol_filter(x=data_dict_conductivity['sigma_D'][0][:, j],
                                                                       window_length=FilterToggles.window,
                                                                       polyorder=FilterToggles.polyorder)

    ################################
    # --- PERFORM THE SSA FILTER ---
    ################################
    if FilterToggles.use_SSA_filter:

        # Loop over altitude
        for j in tqdm(range(len(altRange))):

            if j == 0: # case when you're at the bottom of the simulation i.e. where there's no variation

                # --- Conductivities ---
                sigma_P_DC[:, j] = deepcopy(data_dict_conductivity['sigma_P'][0][:, j])
                sigma_P_AC[:, j] = deepcopy(data_dict_conductivity['sigma_P'][0][:, j])

                sigma_H_DC[:, j] = deepcopy(data_dict_conductivity['sigma_H'][0][:, j])
                sigma_H_AC[:, j] = deepcopy(data_dict_conductivity['sigma_H'][0][:, j])

                sigma_D_DC[:, j] = deepcopy(data_dict_conductivity['sigma_D'][0][:, j])
                sigma_D_AC[:, j] = deepcopy(data_dict_conductivity['sigma_D'][0][:, j])

            else:
                # --- SSA the conductivity ---
                SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_P'][0][:, j], L=FilterToggles.wLen, mirror_percent=FilterToggles.mirror_percent)
                sigma_P_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=FilterToggles.DC_components))
                sigma_P_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=FilterToggles.AC_components))
                del SSA_sigma
                gc.collect()

                SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_H'][0][:, j], L=FilterToggles.wLen, mirror_percent=FilterToggles.mirror_percent)
                sigma_H_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=FilterToggles.DC_components))
                sigma_H_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=FilterToggles.AC_components))
                del SSA_sigma
                gc.collect()

                SSA_sigma = stl.SSA(tseries=data_dict_conductivity['sigma_D'][0][:, j], L=FilterToggles.wLen, mirror_percent=FilterToggles.mirror_percent)
                sigma_D_DC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=FilterToggles.DC_components))
                sigma_D_AC[:, j] = deepcopy(SSA_sigma.reconstruct(indices=FilterToggles.AC_components))
                del SSA_sigma
                gc.collect()

    ############################
    # --- WRITE OUT THE DATA ---
    ############################
    data_dict_output = {**data_dict_spatial,
                      **{
                          'sigma_P_DC':[sigma_P_DC, deepcopy(data_dict_conductivity['sigma_P'][1])],
                          'sigma_P_AC': [sigma_P_AC, deepcopy(data_dict_conductivity['sigma_P'][1])],

                          'sigma_H_DC': [sigma_H_DC, deepcopy(data_dict_conductivity['sigma_H'][1])],
                          'sigma_H_AC': [sigma_H_AC, deepcopy(data_dict_conductivity['sigma_H'][1])],

                          'sigma_D_DC': [sigma_H_DC, deepcopy(data_dict_conductivity['sigma_D'][1])],
                          'sigma_D_AC': [sigma_H_AC, deepcopy(data_dict_conductivity['sigma_D'][1])]
                          }
                      }

    outputPath = rf'{FilterToggles.outputFolder}/{FilterToggles.filter_path}/filtered_conductivity.cdf'
    stl.outputDataDict(outputPath, data_dict_output)
          