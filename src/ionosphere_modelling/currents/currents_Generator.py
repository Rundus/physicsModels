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
    # Description: Get the Fields and conductivity values for the
    if CurrentsToggles.filter_data:
        if CurrentsToggles.use_boxcar:
            data_dict_filter = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\currents\savitz_golay_filtered\filtered_EFields_conductivity.cdf')
        elif CurrentsToggles.use_savitz_golay:
            data_dict_filter = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\currents\savitz_golay_filtered\filtered_EFields_conductivity.cdf')
        elif CurrentsToggles.use_SSA_filter:
            data_dict_filter = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\currents\ssa_filtered\filtered_EFields_conductivity.cdf')
        data_dict_output = {**data_dict_output,
                            **{
                            'E_N_DC': deepcopy(data_dict_filter['E_N_DC']),
                            'E_N_AC': deepcopy(data_dict_filter['E_N_AC']),

                           'E_p_DC': deepcopy(data_dict_filter['E_p_DC']),
                           'E_p_AC': deepcopy(data_dict_filter['E_p_AC']),

                           'sigma_P_DC': deepcopy(data_dict_filter['sigma_P_DC']),
                           'sigma_P_AC': deepcopy(data_dict_filter['sigma_P_AC']),

                           'sigma_H_DC': deepcopy(data_dict_filter['sigma_H_DC']),
                           'sigma_H_AC': deepcopy(data_dict_filter['sigma_H_AC']),

                           'sigma_D_DC': deepcopy(data_dict_filter['sigma_D_DC']),
                           'sigma_D_AC': deepcopy(data_dict_filter['sigma_D_AC'])
                             }
                            }
    else:
        data_dict_output = {**data_dict_output,
                            **{
                                'E_N_DC': [deepcopy(data_dict_EField['E_N'])],
                                'E_N_AC': [deepcopy(data_dict_EField['E_N'])],

                                'E_p_DC': [deepcopy(data_dict_EField['E_p'])],
                                'E_p_AC': [deepcopy(data_dict_EField['E_p'])],

                                'sigma_P_DC': [deepcopy(data_dict_conductivity['sigma_P'])],
                                'sigma_P_AC': [deepcopy(data_dict_conductivity['sigma_P'])],

                                'sigma_H_DC': [deepcopy(data_dict_conductivity['sigma_H'])],
                                'sigma_H_AC': [deepcopy(data_dict_conductivity['sigma_H'])],

                                'sigma_D_DC': [deepcopy(data_dict_conductivity['sigma_D'])],
                                'sigma_D_AC': [deepcopy(data_dict_conductivity['sigma_D'])]
                            }
                            }

    data_dict_output = {**data_dict_spatial,**data_dict_output}


    ############################
    # --- CALCULATE CURRENTS ---
    ############################
    tags = ['DC', 'AC']
    for tag in tqdm(tags):

        # [1] Calculate the Horizontal/Vertical Distances in the simulation for the Gradient
        dSigma_P_dN = np.zeros(shape=(len(LShellRange), len(altRange)))
        dE_N_dN = np.zeros(shape=(len(LShellRange), len(altRange)))

        for j in range(len(altRange)):

            # Calculate the horizontal distance
            dis = data_dict_spatial['grid_dN'][0][:,j]

            dSigma_P_dN[:, j] = np.gradient(data_dict_output[f'sigma_P_{tag}'][0][:, j], dis)
            dE_N_dN[:, j] = np.gradient(data_dict_output[f'E_N_{tag}'][0][:, j], dis)

            # [2] # Calculate the Normal Current (Pedersen)
            J_N = deepcopy(data_dict_output[f'sigma_P_{tag}'][0]) * deepcopy(data_dict_output[f'E_N_{tag}'][0])

            # Calculate the Tangent Current (Hall)
            J_T = deepcopy(data_dict_output[f'sigma_H_{tag}'][0]) * deepcopy(data_dict_output[f'E_N_{tag}'][0])

            # Calculate J_p
            J_p = deepcopy(data_dict_output[f'sigma_D_{tag}'][0]) * deepcopy(data_dict_output[f'E_p_{tag}'][0])

            # Calculate J_parallel through integration
            dJ_parallel_dp = -1*deepcopy(data_dict_output[f'E_N_{tag}'][0])*dSigma_P_dN - 1*deepcopy(data_dict_output[f'sigma_P_{tag}'][0])*dE_N_dN

        # --- Construct the output Data Dict ---
        data_dict_output = { **data_dict_output,
                             **{f'J_P_{tag}': [J_N, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'A/m^2', 'LABLAXIS': f'Pedersen Current {tag}', 'VAR_TYPE': 'data'}],
                                f'J_H_{tag}': [J_T, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'A/m^2', 'LABLAXIS': f'Hall Current {tag}', 'VAR_TYPE': 'data'}],
                                f'J_p_{tag}': [J_p, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': f'Field Aligned Current {tag}', 'VAR_TYPE': 'data'}],
                                f'J_parallel_dp_{tag}': [dJ_parallel_dp, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^3', 'LABLAXIS': f'Parallel Current Per Meter {tag}', 'VAR_TYPE': 'data','SCALEMIN':-1.7E-10,'SCALEMAX':1.7E-10}]
                            }}

        ###################################
        # --- HEIGHT INTEGRATE CURRENTS ---
        ###################################

        J_P_HI = np.array([simpson(y=data_dict_output[f'J_P_{tag}'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
        J_H_HI = np.array([simpson(y=data_dict_output[f'J_H_{tag}'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
        J_p_HI = np.array([simpson(y=data_dict_output[f'J_p_{tag}'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
        J_parallel_HI = np.array([simpson(y=data_dict_output[f'J_parallel_dp_{tag}'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])


        data_dict_output = {**data_dict_output,
                            **{
                                f'J_p_HI_{tag}': [J_p_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': f'Field Aligned Current {tag} (Height Integrated)', 'VAR_TYPE': 'data'}],
                                f'J_P_HI_{tag}': [J_P_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': f'Pedersen Current {tag} (Height Integrated)', 'VAR_TYPE': 'data'}],
                                f'J_H_HI_{tag}': [J_H_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': f'Hall Current {tag} (Height Integrated)', 'VAR_TYPE': 'data'}],
                                f'J_parallel_HI_{tag}': [J_parallel_HI, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'A/m^2', 'LABLAXIS': f'Parallel Current {tag} (Height Integrated)', 'VAR_TYPE': 'data'}]
                               }}

    outputPath = rf'{CurrentsToggles.outputFolder}\currents.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
