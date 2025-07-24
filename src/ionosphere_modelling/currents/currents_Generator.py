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
    from scipy.integrate import simpson

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

    ###########################################
    # --- GENERATE THE IONOSPHERIC CURRENTS ---
    ###########################################

    if CurrentsToggles.smooth_data:
        from scipy.signal import savgol_filter

        for j in range(len(altRange)):

            if CurrentsToggles.use_savitz_golay:
                # smooth electric field data
                data_dict_EField['E_N'][0][:,j] = savgol_filter(x=data_dict_EField['E_N'][0][:,j],
                                                                window_length=CurrentsToggles.window,
                                                                polyorder=CurrentsToggles.polyorder)

                data_dict_EField['E_p'][0][:, j] = savgol_filter(x=data_dict_EField['E_p'][0][:, j],
                                                                 window_length=CurrentsToggles.window,
                                                                 polyorder=CurrentsToggles.polyorder)

                # smooth conductivity data
                data_dict_conductivity['sigma_P'][0][:, j] = savgol_filter(x=data_dict_conductivity['sigma_P'][0][:, j],
                                                                 window_length=CurrentsToggles.window,
                                                                 polyorder=CurrentsToggles.polyorder)

                data_dict_conductivity['sigma_D'][0][:, j] = savgol_filter(x=data_dict_conductivity['sigma_D'][0][:, j],
                                                                           window_length=CurrentsToggles.window,
                                                                           polyorder=CurrentsToggles.polyorder)
            elif CurrentsToggles.use_boxcar:
                data_dict_EField['E_N'][0][:, j] = np.convolve(data_dict_EField['E_N'][0][:, j], np.ones(CurrentsToggles.N_boxcar) / CurrentsToggles.N_boxcar, mode='same')
                data_dict_EField['E_p'][0][:, j] = np.convolve(data_dict_EField['E_p'][0][:, j], np.ones(CurrentsToggles.N_boxcar) / CurrentsToggles.N_boxcar, mode='same')
                data_dict_conductivity['sigma_P'][0][:, j] = np.convolve(data_dict_conductivity['sigma_P'][0][:, j], np.ones(CurrentsToggles.N_boxcar) / CurrentsToggles.N_boxcar, mode='same')
                data_dict_conductivity['sigma_D'][0][:, j] = np.convolve(data_dict_conductivity['sigma_D'][0][:, j], np.ones(CurrentsToggles.N_boxcar) / CurrentsToggles.N_boxcar, mode='same')
            elif CurrentsToggles.use_SSA_filter:
                data_dict_SSA = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\currents\filtered_data_01.cdf')
                data_dict_EField['E_N'][0] = deepcopy(data_dict_SSA['E_N_DC'][0])
                data_dict_conductivity['sigma_P'][0] = deepcopy(data_dict_SSA['sigma_P_DC'][0])

        data_dict_output ={**data_dict_output,
                           **{
                               'E_N_DC': deepcopy(data_dict_EField['E_N']),
                               'E_p_DC': deepcopy(data_dict_EField['E_p']),
                               'sigma_P_DC': deepcopy(data_dict_conductivity['sigma_P']),
                               'sigma_D_DC': deepcopy(data_dict_conductivity['sigma_D']),
                           }
                           }

    # determine the gradient in pedersen conductivity and Eletric Field
    dSigma_P_dN = np.zeros(shape=(len(LShellRange), len(altRange)))
    dE_N_dN = np.zeros(shape=(len(LShellRange), len(altRange)))
    for j in range(len(altRange)):

        # Calculate the horizontal distance
        dis = np.array([np.sum(data_dict_EField['deltaN'][0][:i + 1, j]) for i in range(len(LShellRange))])

        dSigma_P_dN[:, j] = np.gradient(data_dict_conductivity['sigma_P'][0][:,j],dis)
        dE_N_dN[:, j] = np.gradient(data_dict_EField['E_N'][0][:,j], dis)

    # Calculate the Normal Current (Pedersen)
    J_N = deepcopy(data_dict_conductivity['sigma_P'][0]) * deepcopy(data_dict_EField['E_N'][0])

    # Calculate the Tangent Current (Hall)
    J_T = deepcopy(data_dict_conductivity['sigma_H'][0]) * deepcopy(data_dict_EField['E_N'][0])

    # Calculate J_p
    J_p = deepcopy(data_dict_conductivity['sigma_D'][0]) * deepcopy(data_dict_EField['E_p'][0])

    # Calculate J_parallel through integration
    dJ_parallel_dp = -1*deepcopy(data_dict_EField['E_N'][0])*dSigma_P_dN -1*deepcopy(data_dict_conductivity['sigma_P'][0])*dE_N_dN

    # --- Construct the output Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{'J_P': [J_N, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'A/m^2', 'LABLAXIS': 'Pedersen Current', 'VAR_TYPE': 'data'}],
                            'J_H': [J_T, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'A/m^2', 'LABLAXIS': 'Hall Current', 'VAR_TYPE': 'data'}],
                            'J_p': [J_p, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'Field Aligned Current', 'VAR_TYPE': 'data'}],
                            'J_parallel_dp': [dJ_parallel_dp, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^3', 'LABLAXIS': 'Parallel Current Per Meter', 'VAR_TYPE': 'data','SCALEMIN':-1.7E-10,'SCALEMAX':1.7E-10}]
                        }}

    ###################################
    # --- HEIGHT INTEGRATE CURRENTs ---
    ###################################

    J_P_HI = np.array([simpson(y=data_dict_output['J_P'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
    J_H_HI = np.array([simpson(y=data_dict_output['J_H'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
    J_p_HI = np.array([simpson(y=data_dict_output['J_p'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
    J_parallel_HI = np.array([simpson(y=data_dict_output['J_parallel_dp'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])


    data_dict_output = {**data_dict_output,
                        **{
                            'J_p_HI': [J_p_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'Field Aligned Current (Height Integrated)', 'VAR_TYPE': 'data'}],
                            'J_P_HI': [J_P_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'Pedersen Current', 'VAR_TYPE': 'data'}],
                            'J_H_HI': [J_H_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'Hall Current', 'VAR_TYPE': 'data'}],
                            'J_parallel_HI': [J_parallel_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': 'Parallel Current', 'VAR_TYPE': 'data'}]
                           }}


    if CurrentsToggles.smooth_data:
        data_dict_output = {**data_dict_output,
                            **{'E_N_DC':deepcopy(data_dict_EField['E_N']),
                               'sigma_P_DC':deepcopy(data_dict_conductivity['sigma_P'])
                               }

                            }

    outputPath = rf'{CurrentsToggles.outputFolder}\currents.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
