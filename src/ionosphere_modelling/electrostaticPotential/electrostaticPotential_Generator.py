
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

    # output all the variables into a data dict
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

    ###########################
    # --- RELAXATION METHOD ---
    ###########################

    # [1] get the indicies of all the initial conditions
    M = len(data_dict_output['simLShell'][0])
    N = len(data_dict_output['simAlt'][0])
    grid_potential_ds = deepcopy(data_dict_output['initial_potential'][0])
    IC_indicies = np.where(np.abs(grid_potential_ds) > 0)
    IC_idxs = np.array([IC_indicies[0][i] * N + IC_indicies[1][i] for i in range(len(IC_indicies[0]))  ]) # flatten the indicies

    # [2] loop over relaxation grid a number of times specified in the toggles
    grid_potential_relax = np.zeros(shape=np.shape(grid_potential_ds))
    grid_flattened = np.array([np.sum(arr) for arr in data_dict_output['initial_potential'][0]])
    grid_potential_relax[:, -1] = grid_flattened
    grid_potential_relax[0] = np.zeros(N)
    grid_potential_relax[-1] = np.zeros(N)
    data_dict_output['initial_potential'][0] = deepcopy(grid_potential_relax)

    for loop_idx in tqdm(range(ElectroStaticToggles.n_iter)):

        # copy the newest iteration
        grid_potential_temp = deepcopy(grid_potential_relax)

        for i in range(M):  # L-Shell variation
            for j in range(N):  # Altitude variation
                if checkBoundary(i=i, j=j, N=N, M=M) !=1: # if you're not on a boundary point
                    # if pos not in IC_idxs: # If you're not an IC value
                    first_term = (1/np.power(deltaZ,2) + p[i][j]/np.power(deltaX,2))**(-1)
                    second_term = (grid_potential_temp[i][j+1] + grid_potential_temp[i][j-1])/np.power(deltaZ,2)
                    third_term = p[i][j]*(grid_potential_temp[i+1][j] + grid_potential_temp[i-1][j])/np.power(deltaX,2)
                    fourth_term = g[i][j]*(grid_potential_temp[i][j+1] - grid_potential_temp[i][j-1])/(2*deltaZ)
                    fifth_term = q[i][j]*(grid_potential_temp[i+1][j] - grid_potential_temp[i-1][j])/(2*deltaX)
                    grid_potential_temp[i][j] = first_term*(second_term+third_term+fourth_term+fifth_term)

        # store the output of the iteration
        grid_potential_relax = deepcopy(grid_potential_temp)

    # store the output of the method
    data_dict_output = {**data_dict_output,
                        **{'potential':[grid_potential_relax,{'DEPEND_0':'simLShell','DEPEND_1':'simAlt', 'UNITS':'V', 'LABLAXIS': 'Electrostatic Potential', 'VAR_TYPE': 'data'}]},
                        }


    #########################
    # --- OUTPUT THE DATA ---
    #########################

    outputPath = rf'{ElectroStaticToggles.outputFolder}\electrostaticPotential_relaxation.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)


generateElectrostaticPotential()