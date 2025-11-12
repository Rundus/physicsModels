# TODO: Correct Ne when using the ACESII_ni spectrum using the EISCAT ne/ni ratio


def generatePlasmaEnvironment():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.plasma_environment.plasma_toggles import PlasmaToggles

    #######################
    # --- LOAD THE DATA ---
    #######################
    # get the geomagnetic field data dict
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\geomagneticField\*.cdf*')[0])
    data_dict_IRI = deepcopy(stl.loadDictFromFile(PlasmaToggles.IRI_filePath)) # collect the IRI data

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    # pre-define the IRI variables first
    dt_targetTime = SpatialToggles.target_time
    time_idx = np.abs(data_dict_IRI['time'][0] - int(dt_targetTime.hour * 60 + dt_targetTime.minute)).argmin()
    LShellRange = data_dict_Bgeo['simLShell'][0]
    altRange = data_dict_Bgeo['simAlt'][0]
    Imasses = np.array([stl.ion_dict[key] for key in PlasmaToggles.wIons])
    Ikeys = np.array([key for key in PlasmaToggles.wIons])

    data_dict_output = {
        'Te': [np.zeros(shape=(len(LShellRange),len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'K', 'LABLAXIS': 'Te'}],
        'Ti': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'K', 'LABLAXIS': 'Ti'}],
        'Ne': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['Ne'][1])],
        'Tn': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['Tn'][1])],
        'simLShell': deepcopy(data_dict_Bgeo['simLShell']),
        'simAlt': deepcopy(data_dict_Bgeo['simAlt']),
        'O+': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['O+'][1])],
        'H+': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['H+'][1])],
        'He+': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['He+'][1])],
        'O2+': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['O2+'][1])],
        'NO+': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['NO+'][1])],
        'N+': [np.zeros(shape=(len(LShellRange), len(altRange))), deepcopy(data_dict_IRI['N+'][1])],

        'C_O+': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'nj/ni_tot', 'LABLAXIS': 'C_O+'}],
        'C_H+': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'nj/ni_tot', 'LABLAXIS': 'C_H+'}],
        'C_He+': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'nj/ni_tot', 'LABLAXIS': 'C_He+'}],
        'C_O2+': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'nj/ni_tot', 'LABLAXIS': 'C_O2+'}],
        'C_NO+': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'nj/ni_tot', 'LABLAXIS': 'C_NO+'}],
        'C_N+': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'nj/ni_tot', 'LABLAXIS': 'C_N+'}],

        'ne_ni_ratio': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'ne/ni'}],
        'ni': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'ni'}],
        'm_eff_i': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i'}]
    }


    #################################################
    # --- DOWNSAMPLE THE IRI DATA TO L-SHELL GRID ---
    #################################################
    for idx1, Lval in tqdm(enumerate(LShellRange)):

        # for each L-Shell, choose the MIDDLE latitude/longitude value. Use that to get the closest IRI slice and interpolate the IRI over altitude there
        middle_idx = int(len(data_dict_Bgeo['grid_lat'][0][0])/2)
        target_lat_idxs = np.abs(data_dict_IRI['lat'][0] - data_dict_Bgeo['grid_lat'][0][idx1][middle_idx]).argmin()
        target_long_idxs = np.abs(data_dict_IRI['lon'][0] - data_dict_Bgeo['grid_long'][0][idx1][middle_idx]).argmin()

        alt_low_idx = np.abs(data_dict_IRI['ht'][0] - altRange[0] / stl.m_to_km).argmin()
        alt_high_idx = np.abs(data_dict_IRI['ht'][0] - altRange[-1] / stl.m_to_km).argmin()

        # Downsample all the IRI data
        for varname in data_dict_IRI.keys():
            if varname not in ['time', 'ht', 'lat', 'lon']:

                reducedData = deepcopy(np.array(data_dict_IRI[varname][0][time_idx, alt_low_idx:alt_high_idx,  target_lat_idxs, target_long_idxs]))

                # --- linear 1D interpolate data to assist the cubic interpolation ---
                interpolated_result = np.interp(altRange, data_dict_IRI['ht'][0][alt_low_idx:alt_high_idx]*stl.m_to_km, reducedData)
                data_dict_output[varname][0][idx1] = interpolated_result
                data_dict_output[varname][1]['DEPEND_0'] = 'simLShell'
                data_dict_output[varname][1]['DEPEND_1'] = 'simAlt'

                data_dict_output[varname][1].pop('DEPEND_2', None)
                data_dict_output[varname][1].pop('DEPEND_3', None)


    # --- rename the ions ---
    for idx, key in enumerate(Ikeys):
        data_dict_output[f'n_{key}'] = data_dict_output.pop(key)
        data_dict_output[f'n_{key}'][1]['UNITS'] = 'm!A-3!N'
        data_dict_output[f'n_{key}'][1]['LABLAXIS'] = f'n_{key}'

    #########################
    # --- IRI ne/ni ratio ---
    #########################
    ni_IRI = 1E6 * np.sum(np.array([deepcopy(data_dict_output[f"n_{key}"][0]) for key in PlasmaToggles.wIons]), axis=0) # convert data into m^-3
    ne_IRI = 1E6 * deepcopy(data_dict_output['Ne'][0])  # convert data into m^-3
    data_dict_output['ne_ni_ratio'][0] = np.divide(ne_IRI, ni_IRI)

    ############################
    # --- ION CONCENTRATIONS ---
    ############################
    for idx, key in enumerate(Ikeys):
        data_dict_output[f'C_{key}'][0] = deepcopy(np.divide(1E6*data_dict_output[f'n_{key}'][0], ni_IRI))
        data_dict_output[f'C_{key}'][1]['UNITS'] = '%'
        data_dict_output[f'C_{key}'][1]['LABLAXIS'] = f'C_{key}'

    ##################################
    # --- INDIVIDUAL ION DENSITIES ---
    ##################################
    if PlasmaToggles.useACESII_density_Profile:
        data_dict_ACESII_ni_spectrum = stl.loadDictFromFile(rf'{PlasmaToggles.outputFolder}\ACESII_ni_spectrum\ACESII_ni_spectrum.cdf')
        for idx, key in enumerate(Ikeys):
            data_dict_output[f'n_{key}'][0] = 1E6*np.multiply(data_dict_ACESII_ni_spectrum[f'ni_spectrum'][0], data_dict_output[f'C_{key}'][0])
    elif PlasmaToggles.useEISCAT_density_Profile:
        data_dict_EISCAT_ne_spectrum = stl.loadDictFromFile(rf'{PlasmaToggles.outputFolder}\EISCAT_ne_spectrum\EISCAT_ne_spectrum.cdf')
        for idx, key in enumerate(Ikeys):
            data_dict_output[f'n_{key}'][0] = (1/deepcopy(data_dict_output['ne_ni_ratio'][0]))*1E6*np.multiply(data_dict_EISCAT_ne_spectrum[f'ne_spectrum'][0], data_dict_output[f'C_{key}'][0])
    else:
        for idx, key in enumerate(Ikeys):
            data_dict_output[f'n_{key}'][0] = 1E6*deepcopy(data_dict_output[f'n_{key}'][0])


    ###########################
    # --- TOTAL ION DENSITY ---
    ###########################
    if PlasmaToggles.useACESII_density_Profile:
        data_dict_output['ni'][0] = 1E6 * deepcopy(data_dict_ACESII_ni_spectrum['ni_spectrum'][0])
    elif PlasmaToggles.useEISCAT_density_Profile:
        data_dict_output['ni'][0] = 1E6 * deepcopy(data_dict_EISCAT_ne_spectrum['ne_spectrum'][0]) * (1/deepcopy(data_dict_output['ne_ni_ratio'][0]))
    else:
        data_dict_output['ni'][0] = 1E6 * np.array([deepcopy(data_dict_output[f"n_{key}"][0]) for key in PlasmaToggles.wIons])

    ##################
    # --- ION MASS ---
    ##################
    # get the effective mass based on the IRI
    data_dict_output['m_eff_i'][0] = np.sum(np.array([deepcopy(data_dict_output[f"C_{key}"][0])*Imasses[idx] for idx,key in enumerate(PlasmaToggles.wIons)]), axis=0)

    #################################
    # --- ELECTRON PLASMA DENSITY ---
    #################################
    if PlasmaToggles.useACESII_density_Profile:
        ne_density = 1E6*deepcopy(data_dict_ACESII_ni_spectrum['ni_spectrum'][0])*deepcopy(data_dict_output['ne_ni_ratio'][0])
    elif PlasmaToggles.useEISCAT_density_Profile:
        ne_density = 1E6 * deepcopy(data_dict_EISCAT_ne_spectrum['ne_spectrum'][0])
    else:
        ne_density = 1E6 * deepcopy(data_dict_output['Ne'][0])  # convert data into m^-3

    data_dict_output['ne'] = data_dict_output.pop('Ne')
    data_dict_output['ne'] = [ne_density, {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'ne'}]


    #####################
    # --- PLASMA BETA ---
    #####################
    plasmaBeta = (2 * stl.u0 *stl.kB)*(data_dict_output['ne'][0] * data_dict_output['Te'][0] ) / np.power(data_dict_Bgeo['Bgeo'][0],2)
    data_dict_output = {**data_dict_output, **{'beta_e': [plasmaBeta, {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta_e'}]}}

    ##########################
    # --- PLASMA FREQUENCY ---
    ##########################
    plasmaDensity = data_dict_output['ne'][0]
    plasmaFreq = np.array([np.sqrt(plasmaDensity[i] * (stl.q0 * stl.q0) / (stl.ep0 * stl.m_e)) for i in range(len(plasmaDensity))])
    data_dict_output = {**data_dict_output, **{'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}]}}


    ############################
    # --- ION CYCLOTRON FREQ ---
    ############################
    n_ions = np.array([data_dict_output[f"n_{key}"][0] for key in Ikeys])
    ionCyclotron_ions = np.array([stl.q0 * data_dict_Bgeo['Bgeo'][0] / mass for mass in Imasses])
    ionCyclotron_eff = np.sum(ionCyclotron_ions * n_ions, axis=0) / data_dict_output['ni'][0]
    electronCyclotron = stl.q0 * data_dict_Bgeo['Bgeo'][0] / stl.m_e

    data_dict_output = {**data_dict_output,
                        **{'Omega_e': [electronCyclotron, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'electronCyclotron'}]},
                        **{'Omega_i_eff': [ionCyclotron_eff, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron_eff'}]},
                        **{f'Omega_{key}': [ionCyclotron_ions[idx], {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': f'ionCyclotron_{key}'}] for idx, key in enumerate(Ikeys)}
                 }



    ###########################
    # --- ION LARMOR RADIUS ---
    ###########################
    Ti = data_dict_output['Ti'][0]
    n_ions = np.array([data_dict_output[f"n_{key}"][0] for idx, key in enumerate(Ikeys)])
    vth_ions = np.array([np.sqrt(2) * np.sqrt(8 * stl.kB * Ti / mass) for mass in Imasses])  # the np.sqrt(2) comes from the vector sum of two dimensions
    ionLarmorRadius_ions = np.array([vth_ions[idx] / data_dict_output[f"Omega_{key}"][0] for idx, key in enumerate(Ikeys)])
    ionLarmorRadius_eff = np.sum(n_ions * ionLarmorRadius_ions, axis=0) / data_dict_output['ni'][0]
    data_dict_output = {**data_dict_output,
                 **{'ionLarmorRadius_eff': [ionLarmorRadius_eff, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'ionLarmorRadius_eff'}]},
                 **{f'ionLarmorRadius_{key}': [ionLarmorRadius_ions[idx], {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': f'ionLarmorRadius_{key}'}] for idx, key in enumerate(Ikeys)}
                 }


    #####################
    # --- OUTPUT DATA ---
    #####################

    # --- Construct the Data Dict ---
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': None, 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}


    # update the data dict attrs
    for key, val in data_dict_output.items():
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict_output[key][1].items():
            newAttrs[subKey] = subVal

        data_dict_output[key][1] = newAttrs

    outputPath = rf'{PlasmaToggles.outputFolder}\plasma_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
