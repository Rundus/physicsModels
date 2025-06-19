
import numpy as np
def checkBoundary(i, j, N, M):
    '''
    Checks input indices are at the boundary of the simulation space
    :param i: COLUMN index , relating to simLShell
    :param j: ROW indxes, relating to simAlt
    :param N: Row indices
    :param M: Column indices
    :return: Boolean
    '''
    if i == M-1: # Right Side Boundary
        return 1
    elif i == 0: # Left Side Boundary
        return 1
    elif j==0: # Bottom side Boundary
        return 1
    elif j == N-1: # TopSide Boundary
        return 1
    else:
        return 0


# def getBoundary_transformed_points(matrix):
#     mask = np.ones(matrix.shape,dtype=bool)
#     mask[matrix.ndim*(slice(1,-1),)] = False
#     col_idxs, row_idxs = np.where(mask==True)
#     return np.array([ val[0]+ np.shape(matrix)[1]*val[1] for val in zip(row_idxs,col_idxs)])




def checkInitialConditionBoundary(coo, targets):
    shape = (6, 7)
    c_ravel = np.ravel_multi_index(coo.T, shape)
    t_ravel = np.ravel_multi_index(targets.T, shape)
    boolean_list = np.isin(c_ravel, t_ravel)
    if True in boolean_list:
        return True
    else:
        return False



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
                           'grid_potential': [np.array(grid_potential), {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                           'simLShell':deepcopy(data_dict_spatial['simLShell']),
                           'simAlt':deepcopy(data_dict_spatial['simAlt'])
                       }
                        }

    for key, cal in data_dict_output.items():

        if key not in ['simAlt']:

            data = data_dict_output[key][0]

            # reduce the data by the modulus amount
            dlen = len(data)
            if len(data) % ElectroStaticToggles.N_avg != 0:
                dlen -= dlen % ElectroStaticToggles.N_avg
            data = data[:dlen]

            # break up the data into chunks
            chunked = np.split(data, round(len(data) / ElectroStaticToggles.N_avg))

            # average data if it's a conductivity or potential
            if key in ['sigma_P', 'sigma_D', 'grid_potential', 'grid_deltaAlt','simLShell']:
                data = np.array([np.nanmean(chunked[i], axis=0) for i in range(len(chunked))])
            elif key in ['grid_deltaX']:
                # increase the grid spacing if it's a distance
                data = np.array([np.nansum(chunked[i], axis=0) for i in range(len(chunked))])


            # ONLY allow a single value for the electrostatic potential at each L-Shell
            if key in ['grid_potential']:
                for i in range(len(data)): # loop over L-Shell
                    temp = deepcopy(data[i])
                    maxVal = np.max(np.abs(temp))
                    temp[np.where(np.abs(temp)<maxVal)] = 0
                    data[i] = temp

            data_dict_output[key][0] = data

    grid_flattened = np.array([1.5*np.sum(arr) for arr in data_dict_output['grid_potential'][0]])

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
        g[idx,:] = np.gradient(data_log, deltaZ)

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

    grid_potential_ds = deepcopy(data_dict_output['grid_potential'][0])


    if ElectroStaticToggles.perform_mapping:
        # --- --- --- --- --- --- --- --- --- --- --- --
        # --- populate the coefficients of the matrix ---
        # --- --- --- --- --- --- --- --- --- --- --- --
        # [2] Construct the Array
        # [a] form the sparse A matrix
        M = len(data_dict_output['simLShell'][0])
        N = len(data_dict_output['simAlt'][0])
        dim = N * M
        A_matrix = lil_matrix((dim, dim))

        # [e] apply the zero boundary condition values to the solution
        grid_potential_ds[:, -1] = 1.5*grid_flattened  # set the Top to some initial conditions
        grid_potential_ds[:, 0] = 0.1*grid_flattened  # set the Top to some initial conditions
        grid_potential_ds[0] = np.zeros(shape=N)  # set the left side to zero
        grid_potential_ds[-1] = np.zeros(shape=N)  # set the right side to zero

        # [1] Get the col/row indices of the initial condition matrix
        IC_indicies = np.where(np.abs(grid_potential_ds) > 0)
        IC_idxs = np.array([IC_indicies[0][i]*N + IC_indicies[1][i] for i in range(len(IC_indicies[0]))])

        # [] Form the solution vector
        b_matrix = np.zeros(shape=(dim))

        # [b] Fill in the sparse matrix to form the coefficent matrix
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
                    b_matrix[pos] = deepcopy(grid_potential_ds[i][j])

                # IF on a boundary point, set that SPECIFIC potential value to zero in the sparse matrix
                elif checkBoundary(i=i,j=j,N=N,M=M)==1:

                    # # TOP ROW (NO CORNERS)
                    # if j == N-1 and i not in [0, M-1]:
                    #     c_sub_1+=1
                    #
                    #     A_coef_temp = (1 - 2*p*np.power(deltaZ/deltaX,2) + g*deltaZ)
                    #     B_coef_temp = -1*(2 + 0.5*g*deltaZ)
                    #     C_coef_temp = np.ones(shape=np.shape(g))
                    #     D_coef_temp = p*np.power(deltaZ/deltaX,2) + 0.5*q*np.power(deltaZ,2)/deltaX
                    #     E_coef_temp =p*np.power(deltaZ/deltaX,2) - 0.5*q*np.power(deltaZ,2)/deltaX
                    #     pairs = [[i, j], [i, j-1], [i, j - 2], [i + 1, j], [i - 1, j]]
                    #     coeff = [A_coef_temp[i][j], B_coef_temp[i][j], C_coef_temp[i][j], D_coef_temp[i][j], E_coef_temp[i][j]]
                    #     for pidx, pair in enumerate(pairs):
                    #         pos_idx = N * pair[0] + pair[1]  # Transform the solution matrix indices to A_matrix indices: pos = column_idx + dim*row_idx
                    #         A_matrix[counter, pos_idx] = coeff[pidx]
                    #
                    #
                    # # TOP RIGHT CORNER
                    # elif j == N-1 and i == M-1:
                    #     c_sub_2 += 1
                    #     A_coef_temp = (1 + p *np.power(deltaZ/deltaX,2) + g*deltaZ + q * np.power(deltaZ,2)/deltaX)
                    #     B_coef_temp = -1 * (2 +  g * deltaZ)
                    #     C_coef_temp = np.ones(shape=np.shape(g))
                    #     D_coef_temp = -2*p * np.power(deltaZ / deltaX, 2) - q * np.power(deltaZ, 2) / deltaX
                    #     E_coef_temp = p * np.power(deltaZ / deltaX, 2)
                    #     pairs = [[i, j], [i, j - 1], [i, j - 2], [i - 1, j], [i - 2, j]]
                    #     coeff = [A_coef_temp[i][j], B_coef_temp[i][j], C_coef_temp[i][j], D_coef_temp[i][j], E_coef_temp[i][j]]
                    #     for pidx, pair in enumerate(pairs):
                    #         pos_idx = N * pair[0] + pair[1]  # Transform the solution matrix indices to A_matrix indices: pos = column_idx + dim*row_idx
                    #         A_matrix[counter, pos_idx] = coeff[pidx]
                    #
                    # # RIGHT SIDE (NO CORNERS)
                    # elif i == M-1 and j not in [0,N-1]:
                    #     c_sub_3 += 1
                    #     A_coef_temp = (1 +  0.5*g * deltaZ )
                    #     B_coef_temp = (-2 + p*np.power(deltaZ/deltaX,2) + q * np.power(deltaZ,2)/deltaX)
                    #     C_coef_temp = (1 - 0.5*g*deltaZ)
                    #     D_coef_temp = (-2*p * np.power(deltaZ/deltaX,2) - q * np.power(deltaZ,2)/deltaX)
                    #     E_coef_temp = p * np.power(deltaZ / deltaX, 2)
                    #     pairs = [[i, j+1], [i, j ], [i, j - 1], [i - 1, j], [i - 2, j]]
                    #     coeff = [A_coef_temp[i][j], B_coef_temp[i][j], C_coef_temp[i][j], D_coef_temp[i][j], E_coef_temp[i][j]]
                    #     for pidx, pair in enumerate(pairs):
                    #         pos_idx = N * pair[0] + pair[1]  # Transform the solution matrix indices to A_matrix indices: pos = column_idx + dim*row_idx
                    #         A_matrix[counter, pos_idx] = coeff[pidx]
                    #
                    # else: # ENTIRE LEFT & BOTTOM SIDE
                    #     A_matrix[counter, pos] = 1

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

        # [d] convert lil_matrix to a csr_matrix
        A_matrix = csr_matrix(A_matrix)

        # [g] Solve the Ax=B problem and reshape into potential grid
        solved_potential = spsolve(A=A_matrix, b=b_matrix).reshape((M, N))

        # --- store the outputs  ---
        data_dict_output = {**data_dict_output,
                            **{'solved_potential': [solved_potential, {'DEPEND_0':'simLShell','DEPEND_1':'simAlt', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                               'b_matrix': [b_matrix.reshape((M,N)), {'DEPEND_0':'simLShell','DEPEND_1':'simAlt', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
                                # 'coeff_matrix': [A_matrix.toarray().reshape((dim, dim)), {'DEPEND_0': None, 'DEPEND_1': None, 'UNITS': 'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}]
                            }
                            }



        # --- Interpolate Solved potential onto simulation grid ---
        from scipy.interpolate import CubicSpline
        potential = np.zeros(shape=(len(data_dict_spatial['simLShell'][0]),len(data_dict_spatial['simAlt'][0])))

        for idx in range(len(data_dict_spatial['simAlt'][0])):
            cs = CubicSpline(data_dict_output['simLShell'][0],solved_potential[:,idx])
            potential[:,idx] = cs(data_dict_spatial['simLShell'][0])

        data_dict_output = {**data_dict_output,
                            **{
                                'potential':[potential,deepcopy(data_dict_output['solved_potential'][1])],
                                'simLShell_nds': deepcopy(data_dict_spatial['simLShell']),
                                'simAlt_nds': deepcopy(data_dict_spatial['simAlt'])
                               }
                            }
        data_dict_output['potential'][1]['DEPEND_0'] = 'simLShell_nds'
        data_dict_output['potential'][1]['DEPEND_1'] = 'simAlt_nds'

    #########################
    # --- OUTPUT THE DATA ---
    #########################

    outputPath = rf'{ElectroStaticToggles.outputFolder}\electrostaticPotential.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)

generateElectrostaticPotential()