def generateElectrostaticPotential():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.electrostaticPotential.electrostaticPotential_toggles import ElectroStaticToggles
    from scipy.sparse import csr_matrix

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_integrated_potential = stl.loadDictFromFile(r'C:\Data\ACESII\science\integrated_potential\low\ACESII_36364_integrated_potential.cdf')
    data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\conductivity\*.cdf*')[0])

    ###############################################
    # --- RE-GRID THE POTENTIAL ONTO SIMULATION ---
    ###############################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]

    grid_potential = [[[] for j in range(len(altRange))] for i in range(len(LShellRange))]

    counter = 0
    for Lval, altVal in zip(data_dict_integrated_potential['L-Shell'][0],stl.m_to_km*data_dict_integrated_potential['Alt'][0]):

        # Find where this integrated potential best fits in the simulation grid
        LShell_idx = np.abs(LShellRange - Lval).argmin()
        Alt_idx = np.abs(altRange - altVal).argmin()

        # Place this value into the respective
        grid_potential[LShell_idx][Alt_idx].append(data_dict_integrated_potential['Potential'][0][counter])
        counter+=1

    # average all the bins with data and set empty bins to zero
    for idx1 in range(len(LShellRange)):
        for idx2 in range(len(altRange)):
            val = grid_potential[idx1][idx2]
            if len(val) == 0:
                grid_potential[idx1][idx2] = 0
            elif len(val) > 0:
                grid_potential[idx1][idx2] = np.average(val)

    grid_potential = np.array(grid_potential)

    if ElectroStaticToggles.perform_mapping:

        ###################################
        # --- CALCULATE THE COEFFICENTS ---
        ###################################

        # --- Calculate the conductivity ratio term p(x,z) = sigma_P / sigma_D ---
        p = deepcopy(data_dict_conductivity['sigma_P'][0])/deepcopy(data_dict_conductivity['sigma_D'][0])
        data_dict_output = {**data_dict_output,
                            **{'p_coeff': [np.array(p), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': None, 'LABLAXIS': 'p_coefficent', 'VAR_TYPE': 'support_data'}]}
                            }

        # --- Calculate the term g(x,z)= d ln(sigma_D)/dz ---
        delta_sigma_D = np.zeros(shape=(len(SpatialToggles.simLShell), len(SpatialToggles.simAlt)))

        for idx, Lval in enumerate(LShellRange):
            for idx_z, alt in enumerate(altRange):
                if idx_z == len(altRange)-1:
                    delta_sigma_D[idx][idx_z] = data_dict_conductivity['sigma_D'][0][idx][idx_z] - data_dict_conductivity['sigma_D'][0][idx][idx_z-1]
                else:
                    delta_sigma_D[idx][idx_z] = data_dict_conductivity['sigma_D'][0][idx][idx_z+1] - data_dict_conductivity['sigma_D'][0][idx][idx_z]

        g = (1/data_dict_conductivity['sigma_D'][0])*(delta_sigma_D/data_dict_spatial['grid_deltaAlt'][0])
        data_dict_output = {**data_dict_output,
                            **{'g_coeff': [np.array(g), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': '1/m', 'LABLAXIS': 'g_coefficent', 'VAR_TYPE': 'support_data'}]}
                            }

        # --- Calculate the term: q(x,z) = 1/sigma_D * dsigma_p /dx ---
        delta_sigma_P_x = np.zeros(shape=(len(SpatialToggles.simLShell),len(SpatialToggles.simAlt)))

        for idx, Lval in enumerate(LShellRange): # DeltaX grid:
            for idx_z, alt in enumerate(altRange):
                if idx == len(LShellRange)-1:
                    delta_sigma_P_x[idx][idx_z] = data_dict_conductivity['sigma_P'][0][idx][idx_z] - data_dict_conductivity['sigma_P'][0][idx-1][idx_z]
                else:
                    delta_sigma_P_x[idx][idx_z] = data_dict_conductivity['sigma_P'][0][idx+1][idx_z] - data_dict_conductivity['sigma_P'][0][idx][idx_z]


        q = (1/data_dict_conductivity['sigma_D'][0])*(delta_sigma_P_x/data_dict_spatial['grid_deltaX'][0])
        data_dict_output = {**data_dict_output,
                            **{'q_coeff': [np.array(q), {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'1/m', 'LABLAXIS': 'q_coefficent', 'VAR_TYPE': 'support_data'}]}
                            }
        ######################################
        # --- FORM THE PDE SOLUTION MATRIX ---
        ######################################

        # get the dimensions
        N = len(altRange)
        M = len(LShellRange)

        # Calculate all the coefficents
        deltaZ = deepcopy(data_dict_spatial['grid_deltaAlt'][0])
        deltaX = deepcopy(data_dict_spatial['grid_deltaX'][0])

        A_coef =np.array(1/np.power(deltaZ,2) + g/(deltaZ)).T
        B_coef = np.array(-1*(2/np.power(deltaZ,2) + 2*p/np.power(deltaX,2) + g/deltaZ + q/deltaX)).T
        C_coef = np.array(1/np.power(deltaZ,2)).T
        D_coef = np.array(p/np.power(deltaX,2) + q/deltaX).T
        E_coef = np.array(p/np.power(deltaX,2)).T

        # form the sparse A matrix
        A_matrix = csr_matrix((M, N))
        from sys import getsizeof
        print(getsizeof(A_matrix))

        A_matrix[0][0] = 100
        print(A_matrix[0][0])

        # # populate the coefficents of the matrix
        # counter = 0
        # for j in range(M):
        #     for i in range(N):
        #
        #         # create a matrix identical to the final solution for Phi(x,z)
        #         temp_vector = np.zeros(shape=(M,N)).T
        #
        #         # --- Populate temp_vector ---
        #         # Description: for each i,j index, only populate the temp_vector with coefficents at the relevant i,j positions,
        #         # which is dictated by the PDE discretized solution. Make special cases for the initial conditions and boundary values
        #
        #         # First term
        #         temp_vector[i][j+1] = A_coef[i][j]
        #
        #         # Second Term
        #         temp_vector[i][j] = B_coef[i][j]
        #
        #         # Third Term
        #         temp_vector[i][j-1] = C_coef[i][j]
        #
        #         # Fourth Term
        #         temp_vector[i+1][j] = D_coef[i][j]
        #
        #         # Fifth Term
        #         temp_vector[i-1][j] = E_coef[i][j]
        #
        #
        #         # --- Flatten temp_vector ---
        #         temp_vector_flatten = np.array(temp_vector).flatten()
        #
        #         # --- store particular ODE ---
        #         A_matrix[counter] = temp_vector_flatten
        #         counter += 1




    #########################
    # --- OUTPUT THE DATA ---
    #########################
    # --- Construct the output data dict ---
    for key in ['simAlt','simLShell']:
        data_dict_output = {**data_dict_output,
                            **{key:deepcopy(data_dict_spatial[key])}
                            }
    data_dict_output = { **data_dict_output,
                         **{'potential_grid': [grid_potential, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                        }
                         }

    outputPath = rf'{ElectroStaticToggles.outputFolder}\electrostaticPotential.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
