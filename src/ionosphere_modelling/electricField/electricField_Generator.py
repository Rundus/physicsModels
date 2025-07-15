def generate_electricField():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.electricField.electricField_toggles import EFieldToggles
    from spacepy import pycdf
    import datetime as dt
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock  # used to determine the time I'm choosing the reference geomagentic field

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\spatial_environment\*.cdf*')[0])
    data_dict_potential = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\electrostaticPotential\electrostaticPotential.cdf')
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\geomagneticField\*.cdf*')[0])
    data_dict_auroral_coord = stl.loadDictFromFile('C:/Data/ACESII/coordinates/auroral_coordinates/low/ACESII_36364_auroral_coordinates_angle.cdf')

    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]

    ###########################################################
    # [1] Calculate the ECEF coordinates of the simulation Grid
    ###########################################################
    grid_ECEF_X = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_ECEF_Y = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_ECEF_Z = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_ECEF_radius = np.zeros(shape=(len(LShellRange), len(altRange)))

    for idx in range(len(LShellRange)):
        geodeticPos = np.array([data_dict_spatial['grid_alt'][0][idx] / 1000, data_dict_spatial['grid_lat'][0][idx], data_dict_spatial['grid_long'][0][idx]]).T
        ISOtime = [dt.datetime(2022, 11, 20, 17, 25).isoformat() for i in range(len(altRange))]
        cvals_GDZ = coord.Coords(geodeticPos, 'GDZ', 'sph')
        cvals_GDZ.ticks = Ticktock(ISOtime, 'ISO')
        cvals_GEO = cvals_GDZ.convert('GEO', 'car')

        grid_ECEF_X[idx] = cvals_GEO.x *stl.m_to_km*stl.Re
        grid_ECEF_Y[idx] = cvals_GEO.y*stl.m_to_km*stl.Re
        grid_ECEF_Z[idx] = cvals_GEO.z*stl.m_to_km*stl.Re

        # calculate the radius position vector BUT add in a small correction (for some reason the spacepy correction is off by ~20 km)
        grid_ECEF_radius[idx] = np.array([19.4 +np.linalg.norm([grid_ECEF_X[idx][i],grid_ECEF_Y[idx][i],grid_ECEF_Z[idx][i]]) for i in range(len(altRange))])


    ###########################################################
    # [2] Determine the simulation distance vectors in the ECEF
    ###########################################################

    # VERTICAL direction (p-hat direction)
    grid_rho_Z_in_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))

    for idx in range(len(LShellRange)):

        # Calculate the differences.
        # Note: multiply by -1 because I want the difference FROM the top of the simulation altitude
        # to the bottom. This is b/c the vertical direction is aligned to the B-Field and
        # I want to use this as the p-hat FAC direction.
        x_dir = -1*np.array([grid_ECEF_X[idx][i+1] - grid_ECEF_X[idx][i] for i in range(len(altRange)-1)])
        y_dir = -1*np.array([grid_ECEF_Y[idx][i+1] - grid_ECEF_Y[idx][i] for i in range(len(altRange)-1)])
        z_dir = -1*np.array([grid_ECEF_Z[idx][i+1] - grid_ECEF_Z[idx][i] for i in range(len(altRange)-1)])

        # form the vector
        vec = np.array([x_dir, y_dir, z_dir]).T
        vec = np.array(list(vec) + [vec[-1]]) # append the final value again

        # store output
        grid_rho_Z_in_ECEF[idx] = vec

    # HORIZONTAL direction (simulation Ex direction)
    grid_rho_X_in_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    for idx in range(len(altRange)):
        # Calculate the differences
        x_dir = np.array([grid_ECEF_X[i + 1][idx] - grid_ECEF_X[i][idx] for i in range(len(LShellRange) - 1)])
        y_dir = np.array([grid_ECEF_Y[i + 1][idx] - grid_ECEF_Y[i][idx] for i in range(len(LShellRange) - 1)])
        z_dir = np.array([grid_ECEF_Z[i + 1][idx] - grid_ECEF_Z[i][idx] for i in range(len(LShellRange) - 1)])

        # form the vector
        vec = np.array([x_dir, y_dir, z_dir]).T
        vec = np.array(list(vec) + [vec[-1]])  # append the final value again

        # store output
        grid_rho_X_in_ECEF[:, idx] = vec


    ###############################################
    # [3] Rotate the ECEF distance vectors into FAC
    ###############################################

    grid_ECEF_to_ENU_transform = np.zeros(shape=(len(LShellRange), len(altRange),3,3))
    grid_B_Field_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_Rsc_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_pHat_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_eHat_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_rHat_ECEF = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_rho_X_FAC = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_rho_Z_FAC = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_rho_X_auroral = np.zeros(shape=(len(LShellRange), len(altRange), 3))
    grid_rho_Z_auroral = np.zeros(shape=(len(LShellRange), len(altRange), 3))

    for i in tqdm(range(len(LShellRange))):
        for j in range(len(altRange)):

            # [3a] calculate the ENU to ECEF transformation matrix
            lat = deepcopy(data_dict_spatial['grid_lat'][0][i][j])
            long = deepcopy(data_dict_spatial['grid_long'][0][i][j])
            grid_ECEF_to_ENU_transform[i][j] = stl.ENUtoECEF(Lat=lat,Long=long)

            # [3b] Convert the B_ENU data to ECEF
            B = np.array([deepcopy(data_dict_Bgeo['B_E'][0][i][j]), deepcopy(data_dict_Bgeo['B_N'][0][i][j]), deepcopy(data_dict_Bgeo['B_U'][0][i][j])])
            grid_B_Field_ECEF[i][j] = np.matmul(grid_ECEF_to_ENU_transform[i][j], B)

            # [3c] Calculate Rocket Radius vector
            lat = deepcopy(data_dict_spatial['grid_lat'][0][i][j])
            long = deepcopy(data_dict_spatial['grid_long'][0][i][j])
            radius = grid_ECEF_radius[i][j]*stl.Re
            grid_Rsc_ECEF[i][j][0] = radius*np.sin(np.radians(90-lat))*np.cos(np.radians(long))
            grid_Rsc_ECEF[i][j][1] = radius*np.sin(np.radians(90-lat))*np.sin(np.radians(long))
            grid_Rsc_ECEF[i][j][2] = radius*np.cos(np.radians(90-lat))

            # [3d] calculate pHat from the ECEF B-Field
            grid_pHat_ECEF[i][j] = np.array(grid_B_Field_ECEF[i][j])/np.linalg.norm(grid_B_Field_ECEF[i][j])

            # [5e] calculate e-hat - the cross of pHat and the Rocket's radius vector
            cross = np.cross(grid_pHat_ECEF[i][j], grid_Rsc_ECEF[i][j])
            grid_eHat_ECEF[i][j] = cross / np.linalg.norm(cross)

            # [5f] calculate e-hat - the cross of pHat and the Rocket's radius vector
            grid_rHat_ECEF[i][j] = np.cross(grid_eHat_ECEF[i][j], grid_pHat_ECEF[i][j])

            # [5g] Transform the distance rho vectors from ECEF to FAC
            rHat = grid_rHat_ECEF[i][j]
            eHat = grid_eHat_ECEF[i][j]
            pHat = grid_pHat_ECEF[i][j]

            grid_rho_X_FAC[i][j] = np.matmul(np.array([rHat, eHat, pHat]), grid_rho_X_in_ECEF[i][j])
            grid_rho_Z_FAC[i][j] = np.matmul(np.array([rHat, eHat, pHat]), grid_rho_Z_in_ECEF[i][j])

            # [5h] Rotate the FAC distance vectors about the auroral angle to get the auroral coordinates
            grid_rho_X_auroral[i][j] = np.matmul(stl.Rz(deepcopy(data_dict_auroral_coord['rotation_Angle'][0])), grid_rho_X_FAC[i][j])
            grid_rho_Z_auroral[i][j] = np.matmul(stl.Rz(deepcopy(data_dict_auroral_coord['rotation_Angle'][0])), grid_rho_Z_FAC[i][j])


    ##########################################################
    # [4] Calculate the Electric Fields In Auroral Coordinates
    ##########################################################
    # Calculate the spatial gradients in the electrical field PERPENDICULAR to B-geo
    grid_E_T = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_E_N = np.zeros(shape=(len(LShellRange), len(altRange)))
    grid_E_p = np.zeros(shape=(len(LShellRange), len(altRange)))

    # Tangent
    for j in range(len(altRange)):

        # Calculate the horizontal distance
        dis = np.array([np.sum(grid_rho_X_auroral[:i+1, j, 0]) for i in range(len(LShellRange))])

        # Calculate E = -Grad(Phi)
        grid_E_N[:,j] = -1*np.gradient(data_dict_potential['potential'][0][:, j], dis)


    # Vertical
    for i in tqdm(range(len(LShellRange))):

        # Calculate the vertical distance
        dis = np.array([np.sum(grid_rho_Z_auroral[i, :j + 1, 2]) for j in range(len(altRange))])

        # Calculate E = -Grad(Phi)
        grid_E_p[i, :] = -1*np.gradient(data_dict_potential['potential'][0][i, :], dis)

    # --- Construct the Data Dict ---
    data_dict_output = { **data_dict_spatial,
                         **{
                             'ECEF_X': [grid_ECEF_X, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'Re', 'LABLAXIS': 'ECEF_X', 'VAR_TYPE': 'support_data'}],
                             'ECEF_Y': [grid_ECEF_Y, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'Re', 'LABLAXIS': 'ECEF_Y', 'VAR_TYPE': 'support_data'}],
                             'ECEF_Z': [grid_ECEF_Z, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'Re', 'LABLAXIS': 'ECEF_Z', 'VAR_TYPE': 'support_data'}],
                             'E_N': [grid_E_N, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_N', 'VAR_TYPE': 'data'}],
                             'E_T': [grid_E_T, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_T','VAR_TYPE': 'data'}],
                             'E_p': [grid_E_p, {'DEPEND_1': 'simAlt', 'DEPEND_0': 'simLShell', 'UNITS': 'V/m', 'LABLAXIS': 'E_p', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{EFieldToggles.outputFolder}\electric_Field.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
