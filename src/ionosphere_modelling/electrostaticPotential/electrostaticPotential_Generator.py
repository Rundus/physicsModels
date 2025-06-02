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
    from scipy.sparse import csr_matrix,lil_matrix
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




    ###############################################################
    # --- DOWNSAMPLE SIMULATION PARAMETERS IN L-SHELL DIMENSION ---
    ###############################################################

    # Description: The simulation dimensions are large, thus E-Field mapping is REALLY computationally expensive
    # Here we downsample all needed simulation parameters to a new grid. We will interpolate our results onto the simulation later.
    # Parameters: simLShell, simAlt, L-Shell, grid_Potential, Alt, grid_deltaAlt, grid_deltaL,sigma_P, sigma_D
    data_dict_output = {**data_dict_output,
                       **{
                           'sigma_P':deepcopy(data_dict_conductivity['sigma_P']),
                           'sigma_D':deepcopy(data_dict_conductivity['sigma_D']),
                           'grid_deltaAlt': deepcopy(data_dict_spatial['grid_deltaAlt']),
                           'grid_deltaX':deepcopy(data_dict_spatial['grid_deltaX']),
                           'grid_potential': [grid_potential, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],
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
            if key in ['sigma_P','sigma_D','grid_potential','grid_deltaAlt']:
                data = np.array([np.nanmean(chunked[i], axis=0) for i in range(len(chunked))])
            elif key in ['grid_deltaX']:
                # increase the grid spacing if it's a distance
                data = np.array([np.nansum(chunked[i], axis=0) for i in range(len(chunked))])
            elif key in ['simLShell']:
                data = np.array([np.nanmean(chunked[i], axis=0) for i in range(len(chunked))])

            data_dict_output[key][0] = data


    #########################
    # --- MAP THE E-FIELD ---
    #########################

    ###################################
    # --- CALCULATE THE COEFFICENTS ---
    ###################################

    # --- Calculate the conductivity ratio term p(x,z) = sigma_P / sigma_D ---
    p = data_dict_output['sigma_P'][0]/data_dict_output['sigma_D'][0]
    data_dict_output = {**data_dict_output,
                        **{'p_coeff': [np.array(p), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': None, 'LABLAXIS': 'p_coefficent', 'VAR_TYPE': 'support_data'}]}
                        }

    # --- Calculate the term g(x,z)= d ln(sigma_D)/dz ---
    delta_sigma_D = np.zeros(shape=(len(data_dict_output['simLShell'][0]), len(SpatialToggles.simAlt)))

    for idx, Lval in enumerate(data_dict_output['simLShell'][0]):
        for idx_z, alt in enumerate(altRange):
            if idx_z == len(altRange)-1:
                delta_sigma_D[idx][idx_z] = data_dict_output['sigma_D'][0][idx][idx_z] - data_dict_output['sigma_D'][0][idx][idx_z-1]
            else:
                delta_sigma_D[idx][idx_z] = data_dict_output['sigma_D'][0][idx][idx_z+1] - data_dict_output['sigma_D'][0][idx][idx_z]

    g = (1/data_dict_output['sigma_D'][0])*(delta_sigma_D/data_dict_output['grid_deltaAlt'][0])
    data_dict_output = {**data_dict_output,
                        **{'g_coeff': [np.array(g), {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': '1/m', 'LABLAXIS': 'g_coefficent', 'VAR_TYPE': 'support_data'}]}
                        }

    # --- Calculate the term: q(x,z) = 1/sigma_D * dsigma_p /dx ---
    delta_sigma_P_x = np.zeros(shape=(len(data_dict_output['simLShell'][0]),len(SpatialToggles.simAlt)))

    for idx, Lval in enumerate(data_dict_output['simLShell'][0]): # DeltaX grid:
        for idx_z, alt in enumerate(altRange):
            if idx == len(data_dict_output['simLShell'][0])-1:
                delta_sigma_P_x[idx][idx_z] = data_dict_output['sigma_P'][0][idx][idx_z] - data_dict_output['sigma_P'][0][idx-1][idx_z]
            else:
                delta_sigma_P_x[idx][idx_z] = data_dict_output['sigma_P'][0][idx+1][idx_z] - data_dict_output['sigma_P'][0][idx][idx_z]


    q = (1/data_dict_output['sigma_D'][0])*(delta_sigma_P_x/data_dict_output['grid_deltaX'][0])
    data_dict_output = {**data_dict_output,
                        **{'q_coeff': [np.array(q), {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'1/m', 'LABLAXIS': 'q_coefficent', 'VAR_TYPE': 'support_data'}]}
                        }
    ######################################
    # --- FORM THE PDE SOLUTION MATRIX ---
    ######################################

    # Calculate all the coefficents
    deltaZ = deepcopy(data_dict_output['grid_deltaAlt'][0])
    deltaX = deepcopy(data_dict_output['grid_deltaX'][0])

    A_coef =np.array(1/np.power(deltaZ,2) + g/(deltaZ)).T
    B_coef = np.array(-1*(2/np.power(deltaZ,2) + 2*p/np.power(deltaX,2) + g/deltaZ + q/deltaX)).T
    C_coef = np.array(1/np.power(deltaZ,2)).T
    D_coef = np.array(p/np.power(deltaX,2) + q/deltaX).T
    E_coef = np.array(p/np.power(deltaX,2)).T

    if ElectroStaticToggles.perform_mapping:
        # --- --- --- --- --- --- --- --- --- --- --- --
        # --- populate the coefficients of the matrix ---
        # --- --- --- --- --- --- --- --- --- --- --- --

        # [1] Get the col/row indices of the initial condition matrix
        IC_indicies = np.where(grid_potential > 0)
        IC_idxs = np.array([IC_indicies[0], IC_indicies[1]]).T

        # [2] Construct the Array

        # [a] form the sparse A matrix
        N = len(data_dict_output['simAlt'][0])
        M = len(data_dict_output['simLShell'][0])
        dim = N * M
        A_matrix = lil_matrix((dim, dim))

        # [b] Fill in the sparse matrix
        def checkBoundary(i,j,N,M):
            '''
            Checks input indices are at the boundary of the simulation space
            :param i:
            :param j:
            :param N: Row indices
            :param M: Column indices
            :return: Boolean
            '''

            if j == M - 1: # Right Side Boundary
                if i == N - 1: # Upper right corner
                    return 1
                elif i ==0: # Lower Right corner
                    return 1
                else: # Right Edge
                    return 1
            elif j == 0: # Left Side Boundary
                if i == N-1: # upper left corner
                    return 1
                elif i == 0: # lower left corner
                    return 1
                else: # left boundary
                    return 1
            elif i == 0: # Bottom Boundary
                return 1
            elif i == N-1: # Top Boundary
                return 1
            else: # if you're not at a boundary
                return 0


        counter = 0
        for i, j in tqdm(product(*[range(1,N-1),range(1,M-1)])):
            pairs = [[i,j+1],[i,j],[i,j-1],[i+1,j],[i-1,j]]
            coeff = [A_coef[i][j],B_coef[i][j],C_coef[i][j],D_coef[i][j],E_coef[i][j]]
            for pidx, pair in enumerate(pairs):
                idx1, idx2 =pair[0],pair[1]
                pos = idx2 + M * idx1  # Transform the solution matrix indices to A_matrix indices: pos = column_idx + dim*row_idx
                if checkBoundary(idx1,idx2,N=N,M=M):
                    A_matrix[counter, pos] = 1
                elif [idx1, idx2] in IC_idxs:
                    A_matrix[counter, pos] = 1
                else:
                    A_matrix[counter, pos] = coeff[pidx]
            counter+=1
            # # FIRST TERM
            # idx1, idx2 = i, j+1
            # pos = idx1+dim*idx2 # Transform the solution matrix indices to A_matrix indices
            # if checkBoundary(idx1,idx2,N,M):
            #     A_matrix[counter, pos] = 0
            # elif [idx1, idx2] in IC_idxs:
            #     A_matrix[counter, pos] = 1
            # else:
            #     A_matrix[counter, pos] = A_coef[i][j]
            #
            # # SECOND TERM
            # idx1, idx2 = i, j
            # pos = idx1+dim*idx2 # Transform the solution matrix indices to A_matrix indices
            # if checkBoundary(idx1,idx2,N,M):
            #     A_matrix[counter, pos] = 0
            # elif [idx1, idx2] in IC_idxs:
            #     A_matrix[counter, pos] = 1
            # else:
            #     A_matrix[counter, pos] = B_coef[i][j]
            #
            # # THIRD TERM
            # idx1, idx2 = i, j - 1
            # pos = idx1+dim*idx2 # Transform the solution matrix indices to A_matrix indices
            # if checkBoundary(idx1,idx2,N,M):
            #     A_matrix[counter, pos] = 0
            # elif [idx1, idx2] in IC_idxs:
            #     A_matrix[counter, pos] = 1
            # else:
            #     A_matrix[counter, pos] = C_coef[i][j]
            #
            # # FOURTH TERM
            # idx1, idx2 = i+1, j
            # pos = idx1+dim*idx2 # Transform the solution matrix indices to A_matrix indices
            # if checkBoundary(idx1,idx2,N,M):
            #     A_matrix[counter, pos] = 0
            # elif [idx1, idx2] in IC_idxs:
            #     A_matrix[counter, pos] = 1
            # else:
            #     A_matrix[counter, pos] = D_coef[i][j]
            #
            # # FIFTH TERM
            # idx1, idx2 = i-1, j
            # pos = idx1+dim*idx2 # Transform the solution matrix indices to A_matrix indices
            # if checkBoundary(idx1,idx2,N,M):
            #     A_matrix[counter, pos] = 0
            # elif [idx1, idx2] in IC_idxs:
            #     A_matrix[counter,pos] = 1
            # else:
            #     A_matrix[counter,pos] = E_coef[i][j]


        # [c] convert lil_matrix to a csr_matrix
        A_matrix = csr_matrix(A_matrix)

        # check the determinant
        lu = splu(A_matrix)
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        d = diagL.prod() * diagU.prod()
        print(d)

        # [d] apply the zero boundary condition values to the solution
        grid_potential_ds = data_dict_output['grid_potential'][0]
        grid_potential_ds[0] = np.zeros(shape=len(data_dict_output['simAlt'][0]))
        grid_potential_ds[-1] = np.zeros(shape=len(data_dict_output['simAlt'][0]))

        # [e] Flatten the b matrix solution
        b = grid_potential_ds.flatten()

        # [f] Solve the Ax=B problem and reshape into potential grid
        solved_potential_ds = spsolve(A=A_matrix, b=b).reshape((M,N))

        # [g] Interpolate solution onto simulation grid
        # solved_potential = np.zeros(shape=(len(data_dict_spatial['simAlt'][0]), len(data_dict_spatial['simLShell'][0])))
        # xRange = deepcopy(data_dict_spatial['simLShell'][0])
        # yRange = deepcopy(data_dict_spatial['simAlt'][0])
        #
        # for idx, row in enumerate(solved_potential):
        #
        #     cs = CubicSpline(,row)


        # --- store the outputs  ---
        data_dict_output = {**data_dict_output,
                            **{'solved_potential': [solved_potential_ds, {'DEPEND_1':'simAlt','DEPEND_0':'simLShell', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}],}}




    #########################
    # --- OUTPUT THE DATA ---
    #########################

    outputPath = rf'{ElectroStaticToggles.outputFolder}\electrostaticPotential.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
