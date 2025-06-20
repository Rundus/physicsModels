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
    from scipy.sparse import csr_matrix,lil_matrix, csc_matrix
    from scipy.sparse.linalg import spsolve, splu
    from scipy.interpolate import CubicSpline
    from itertools import product

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_integrated_potential = stl.loadDictFromFile(r'C:\Data\ACESII\science\integrated_potential\low\ACESII_36364_integrated_potential.cdf')
    data_dict_conductivity = stl.loadDictFromFile(glob(f'{SimToggles.sim_root_path}\conductivity\*.cdf*')[0])

    #######################################################################
    # --- RE-GRID THE POTENTIAL ONTO SIMULATION WITH INITIAL CONDITIONS ---
    #######################################################################
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

        # Place this value into the respective bin
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



    ###############################################################
    # --- DOWNSAMPLE SIMULATION PARAMETERS IN L-SHELL DIMENSION ---
    ###############################################################
    # Description: The simulation dimensions are large, thus E-Field mapping is REALLY computationally expensive
    # Here we downsample all needed simulation parameters to a new grid. We will interpolate our results onto the simulation later.
    data_dict_output = {**data_dict_output,
                       **{
                           'sigma_P':deepcopy(data_dict_conductivity['sigma_P']),
                           'sigma_D':deepcopy(data_dict_conductivity['sigma_D']),
                           'grid_deltaAlt': deepcopy(data_dict_spatial['grid_deltaAlt']),
                           'grid_deltaX':deepcopy(data_dict_spatial['grid_deltaX']),
                           'initial_potential': [np.array(grid_potential), {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                           'simLShell':deepcopy(data_dict_spatial['simLShell']),
                           'simAlt':deepcopy(data_dict_spatial['simAlt'])
                       }
                        }

    grid_flattened = np.array([np.sum(arr)/len(np.nonzero(arr)[0]) for arr in data_dict_output['initial_potential'][0]])

    #########################
    # --- MAP THE E-FIELD ---
    #########################

    # Calculate all the coefficents - ASSUME A SINGLE VALUE, AVERAGE VALUE for deltaZ, deltaX
    deltaZ = round(np.nanmean(deepcopy(data_dict_output['grid_deltaAlt'][0])))
    deltaX = round(np.nanmean(deepcopy(data_dict_output['grid_deltaX'][0])))

    ###################################
    # --- CALCULATE THE COEFFICENTS ---
    ###################################

    # --- Calculate the conductivity ratio term p(x,z) = sigma_P / sigma_D ---
    p = data_dict_output['sigma_P'][0]/data_dict_output['sigma_D'][0]
    data_dict_output = {**data_dict_output,
                        **{'p_coeff': [np.array(p), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': None, 'LABLAXIS': 'p_coefficent', 'VAR_TYPE': 'support_data'}]}
                        }

    # --- Calculate the term g(x,z)= d ln(sigma_D)/dz ---
    g = np.zeros(shape=(len(data_dict_output['simLShell'][0]), len(SpatialToggles.simAlt)))

    for idx in range(len(data_dict_output['simLShell'][0])):
        data = deepcopy(data_dict_output['sigma_D'][0][idx,:])
        data_log = np.log(data)
        g[idx, :] = np.gradient(data_log, deltaZ)

    data_dict_output = {**data_dict_output,
                        **{'g_coeff': [np.array(g), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': '1/m', 'LABLAXIS': 'g_coefficent', 'VAR_TYPE': 'support_data'}]}
                        }

    # --- Calculate the term: q(x,z) = 1/sigma_D * dsigma_p /dx ---
    q = np.zeros(shape=(len(data_dict_output['simLShell'][0]),len(SpatialToggles.simAlt)))
    for idx_row in range(len(data_dict_output['simAlt'][0])):
        data = deepcopy(data_dict_output['sigma_P'][0][:,idx_row])
        q[:,idx_row] = np.gradient(data, deltaX)
    q = (1/data_dict_output['sigma_D'][0]) * q
    data_dict_output = {**data_dict_output,
                        **{'q_coeff': [np.array(q), {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'1/m', 'LABLAXIS': 'q_coefficent', 'VAR_TYPE': 'support_data'}]}
                        }

    ######################################
    # --- FORM THE PDE SOLUTION MATRIX ---
    ######################################

    # Description: The SOLUTION matrix "A" is a ROW-WISE matrix. Our Ax=b system has x
    # as the flattened ROW elements [PHI_row0col1, PHI_row0col2, ... etc ]
    A_coef = np.array(1 + 0.5*g*deltaZ)
    B_coef = np.array(-2*(1 + p*np.power(deltaZ/deltaX,2)))
    C_coef = np.array(1 - 0.5*g*deltaZ)
    D_coef = np.array(p*np.power(deltaZ/deltaX,2) + 0.5*q*np.power(deltaZ,2)/deltaX)
    E_coef = np.array(p*np.power(deltaZ/deltaX,2) - 0.5*q*np.power(deltaZ,2)/deltaX)

    key_nam = ['A_coef','B_coef','C_coef','D_coef','E_coef']
    key_val = [A_coef,B_coef,C_coef,D_coef,E_coef]
    for k in range(len(key_nam)):
        name = key_nam[k]
        val = key_val[k]
        data_dict_output = {**data_dict_output,
                            **{name: [np.array(val), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': None, 'LABLAXIS': f'{name}_coefficent', 'VAR_TYPE': 'support_data'}]}
                            }

    # get the Initial condition potential grid and make modifications. Store the modifications
    # initial_potential = deepcopy(data_dict_output['initial_potential'][0])
    initial_potential = np.zeros(shape=(len(data_dict_output['simLShell'][0]), len(SpatialToggles.simAlt)))
    initial_potential[:, -1] = 1 * grid_flattened  # set the Top to some initial conditions
    initial_potential[:, 0] = 0 * grid_flattened  # set the bottom to some initial conditions
    initial_potential[0] = np.zeros(shape=len(data_dict_output['simAlt'][0]))  # set the left side to zero
    initial_potential[-1] = np.zeros(shape=len(data_dict_output['simAlt'][0]))  # set the right side to zero
    data_dict_output['initial_potential'][0] = initial_potential

    # --- --- --- --- --- --- --- --- --- --- --- --
    # --- populate the coefficients of the matrix ---
    # --- --- --- --- --- --- --- --- --- --- --- --
    def checkBoundary(i, j, N, M):
        '''
        Checks input indices are at the boundary of the simulation space
        :param i: COLUMN index , relating to simLShell
        :param j: ROW indxes, relating to simAlt
        :param N: Row indices
        :param M: Column indices
        :return: Boolean
        '''
        if i == M - 1:  # Right Side Boundary
            return 1
        elif i == 0:  # Left Side Boundary
            return 1
        elif j == 0:  # Bottom side Boundary
            return 1
        elif j == N - 1:  # TopSide Boundary
            return 1
        else:
            return 0

    # [2] Construct the Array
    # [a] form the sparse A matrix
    M = len(data_dict_output['simLShell'][0])
    N = len(data_dict_output['simAlt'][0])
    dim = N * M
    A_matrix = lil_matrix((dim, dim))

    # [1] Get the col/row indices of the initial condition matrix
    IC_indicies = np.where(np.abs(initial_potential) > 0)
    IC_idxs = np.array([IC_indicies[0][i]*N + IC_indicies[1][i] for i in range(len(IC_indicies[0]))])

    # [2] Form the solution vector
    b_matrix = np.zeros(shape=(dim))

    # [3] Fill in the sparse matrix to form the coefficent matrix
    counter = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c_sub_1 = 0
    c_sub_2 = 0
    c_sub_3 = 0

    for i in tqdm(range(M)): # L-Shell variation
        for j in range(N): # Altitude variation
            pos = i*N + j # Transform the solution matrix indices to A_matrix indices: pos = column_idx + dim*row_id

            # UPDATE THE A MATRIX
            # IF on an IC point, set that SPECIFIC potential value to the IC value
            if pos in IC_idxs:
                A_matrix[counter, pos] = 1
                c1 += 1

                # UPDATE THE SOLUTION B VECTOR
                b_matrix[pos] = deepcopy(initial_potential[i][j])

            # IF on a boundary point, set that SPECIFIC potential value to zero in the sparse matrix
            elif checkBoundary(i=i,j=j,N=N,M=M)==1:
                A_matrix[counter, pos] = 1
                c2 += 1

            # if you have a non-IC point, non boundary condition then fill it with the coefficients
            else:
                pairs = [  [i, j+1],     [i, j],       [i, j-1],       [i+1, j],      [i-1, j]]
                coeff = [A_coef[i][j], B_coef[i][j], C_coef[i][j],   D_coef[i][j],  E_coef[i][j]]
                c3 += 1
                for pidx, pair in enumerate(pairs):
                    pos_idx = N * pair[0] + pair[1]  # Transform the solution matrix indices to A_matrix indices: pos = column_idx + dim*row_idx
                    A_matrix[counter, pos_idx] = coeff[pidx]

            counter += 1

    print('\n')
    print(f'Initial Conditions {c1}')
    print(f'Boundaries {c2}')
    print(f'ODEs {c3}')
    print(f'Top Points: {c_sub_1}')
    print(f'Corner Points: {c_sub_2}')
    print(f'Rightside Points: {c_sub_3}')

    # [4] convert lil_matrix to a csr_matrix
    A_matrix = csr_matrix(A_matrix)

    # [5] Solve the Ax=B problem and reshape into potential grid
    solved_potential = spsolve(A=A_matrix, b=b_matrix).reshape((M, N))

    # --- store the outputs  ---
    data_dict_output = {**data_dict_output,
                        **{'potential': [solved_potential, {'DEPEND_0':'simLShell','DEPEND_1':'simAlt', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                           'b_matrix': [b_matrix.reshape((M,N)), {'DEPEND_0':'simLShell','DEPEND_1':'simAlt', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                            'sigma_D_sigma_P_ratio_sqrt': [np.sqrt(deepcopy(data_dict_output['sigma_D'][0])/deepcopy(data_dict_output['sigma_P'][0])), {'DEPEND_0':'simLShell','DEPEND_1':'simAlt', 'UNITS': None, 'LABLAXIS': 'SigmaD/sigmaP', 'VAR_TYPE': 'data'}]
                        }
                        }

    #########################
    # --- OUTPUT THE DATA ---
    #########################

    outputPath = rf'{ElectroStaticToggles.outputFolder}\electrostaticPotential.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)