# --- conductivity_Generator.py ---
# Description: Model the ionospheric conductivity

from src.physicsModels.ionosphere.conductivity.conductivity_classes import *

def generateIonosphericConductivity():

    # --- imports ---
    from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_toggles import ionizationRecombToggles
    from src.physicsModels.ionosphere.neutral_environment.neutral_toggles import neutralsToggles
    from src.physicsModels.ionosphere.plasma_environment.plasma_toggles import plasmaToggles
    from src.physicsModels.ionosphere.geomagneticField.geomagneticField_toggles import BgeoToggles
    from src.physicsModels.ionosphere.conductivity.conductivity_toggles import conductivityToggles
    import numpy as np
    import spaceToolsLib as stl
    from copy import deepcopy
    from scipy.integrate import trapz

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    # get the geomagnetic field data dict
    data_dict_Bgeo = stl.loadDictFromFile(rf'{BgeoToggles.outputFolder}\geomagneticField.cdf')

    # get the ionospheric neutral data dict
    data_dict_neutral = stl.loadDictFromFile(rf'{neutralsToggles.outputFolder}\neutral_environment.cdf')

    # get the ionospheric plasma data dict
    data_dict_plasma = stl.loadDictFromFile(rf'{plasmaToggles.outputFolder}\plasma_environment.cdf')

    # get the ionization-recombination data dict
    data_dict_ionRecomb = stl.loadDictFromFile(rf'{ionizationRecombToggles.outputFolder}\ionization_rcomb.cdf')

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    LShellRange = data_dict_plasma['simLShell'][0]
    altRange = data_dict_plasma['simAlt'][0]
    data_dict_output = {
        'simLShell': deepcopy(data_dict_plasma['simLShell']),
        'simAlt': deepcopy(data_dict_plasma['simAlt']),
        'ne_total': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm^-3', 'LABLAXIS': 'Electron Density'}],
        'nu_e': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'nu_e'}],
        'sigma_Pedersen': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Pedersen Conductivity'}],
        'sigma_Hall': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Hall Conductivity'}],
        'sigma_parallel': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Parallel Conductivity'}],
    }

    ###############################
    # --- CONSTRUCT n_e PROFILE ---
    ###############################

    # construct the density profile n(z, ILat) - Inverted-V density + Ionospheric base density
    data_dict_output['ne_total'][0] = data_dict_ionRecomb['ne_model'][0] + data_dict_plasma['ne'][0]


    #################################
    # --- Electron Collision Freq ---
    #################################
    # nu_en: electron-neutral collisions - Depends on Te and Nn
    model = Leda2019()
    nu_en_total = np.sum([model.electronNeutral_CollisionFreq(data_dict_neutral=data_dict_neutral,
                                                 data_dict_plasma=data_dict_plasma,
                                                 neutralKey=key) for key in neutralsToggles.wNeutrals], axis=0)

    # nu_ei: electron-ion collisions - Depends on Te and Ne
    model = Johnson1961()
    nu_ei_total = model.electronIon_CollisionFreq(data_dict_neutral, data_dict_plasma, ne_data=data_dict_output['ne_total'][0])

    # nu_e_total: total electron collision rate
    nu_e_total = nu_en_total + nu_ei_total

    # store the data
    data_dict_output['nu_e'][0] = nu_e_total

    ############################
    # --- Ion Collision Freq ---
    ############################
    # Determine the collision frequencies for the various different ions
    model = Leda2019()
    nu_in_profiles = [model.ionNeutral_CollisionsFreq(data_dict_neutral=data_dict_neutral,
                                             data_dict_plasma=data_dict_plasma,
                                             ionKey=key) for key in plasmaToggles.wIons]  # NOp, Op, O2p
    # store the individual collision freqs
    data_dict_output = {**data_dict_output, **{f'nu_i_{key}': [nu_in_profiles[idx], {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': f'ne_i_{key}'}] for idx, key in enumerate(plasmaToggles.wIons)}}

    ##################
    # --- Mobility ---
    ##################
    # electrons
    nu_e = data_dict_output['nu_e'][0]
    Omega_e = data_dict_plasma['Omega_e'][0]
    kappa_e = np.divide(Omega_e, nu_e)
    data_dict_output = {**data_dict_output, **{'kappa_e': [kappa_e, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'kappa_e'}]}}

    # ions
    for idx, key in enumerate(plasmaToggles.wIons):
        Omega_i_specific = deepcopy(data_dict_plasma[f'Omega_{key}'][0])
        kappa_i_specific = np.divide(Omega_i_specific, data_dict_output[f'nu_i_{key}'][0])
        data_dict_output = {**data_dict_output, **{f'kappa_i_{key}': [kappa_i_specific, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': f'kappa_i_{key}'}]}}


    # ######################
    # # --- Conductivity ---
    # ######################
    B_geo = data_dict_Bgeo['Bgeo'][0]

    # calculated electron sigmas
    sigma_par_e = q0 * np.multiply(np.divide(data_dict_output['ne_total'][0], B_geo), kappa_e)
    sigma_P_e = q0 * np.multiply(np.divide(data_dict_output['ne_total'][0], B_geo), kappa_e/(1 + np.power(kappa_e, 2)))
    sigma_H_e = q0 * np.multiply(np.divide(data_dict_output['ne_total'][0], B_geo), np.power(kappa_e, 2)/(1 + np.power(kappa_e, 2)))

    # calculated ion sigmas
    sigma_para_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['simLShell'][0]), len(data_dict_output['simAlt'][0])))
    sigma_P_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['simLShell'][0]), len(data_dict_output['simAlt'][0])))
    sigma_H_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['simLShell'][0]), len(data_dict_output['simAlt'][0])))

    for idx, key in enumerate(plasmaToggles.wIons):
        kappa_val = deepcopy(data_dict_output[f'kappa_i_{key}'][0])
        specific_ion_concentration = np.divide(data_dict_plasma[f'n_{key}'][0], data_dict_plasma['ni'][0])
        n_i = data_dict_output['ne_total'][0]*specific_ion_concentration
        sigma_para_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), kappa_val)
        sigma_P_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), kappa_val / (1 + np.power(kappa_val, 2)))
        sigma_H_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), np.power(kappa_val, 2) / (1 + np.power(kappa_val, 2)))

    data_dict_output['sigma_Pedersen'][0] = sigma_P_e + np.sum(sigma_P_ions, axis=0)
    data_dict_output['sigma_Hall'][0] = sigma_H_e - np.sum(sigma_H_ions, axis=0)
    data_dict_output['sigma_parallel'][0] = sigma_par_e + np.sum(sigma_para_ions, axis=0)


    ##########################################
    # --- Height-Integrated Conductivities ---
    ##########################################

    Sigma_Hall_HI = np.array([trapz(y=data_dict_output['sigma_Hall'][0][tme_idx], x=data_dict_output['simAlt'][0]) for tme_idx in range(len(data_dict_output['simLShell'][0]))])
    Sigma_Pedersen_HI = np.array([trapz(y=data_dict_output['sigma_Pedersen'][0][tme_idx], x=data_dict_output['simAlt'][0]) for tme_idx in range(len(data_dict_output['simLShell'][0]))])
    Sigma_Parallel_HI = np.array([trapz(y=data_dict_output['sigma_parallel'][0][tme_idx], x=data_dict_output['simAlt'][0]) for tme_idx in range(len(data_dict_output['simLShell'][0]))])

    data_dict_output = {**data_dict_output,
                 **{'Sigma_Hall': [Sigma_Hall_HI, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                 **{'Sigma_Pedersen': [Sigma_Pedersen_HI, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                 **{'Sigma_parallel': [Sigma_Parallel_HI, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                 }


    #####################
    # --- OUTPUT DATA ---
    #####################
    # --- Construct the Data Dict ---
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}


    # update the data dict attrs
    for key, val in data_dict_output.items():
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict_output[key][1].items():
            newAttrs[subKey] = subVal

        data_dict_output[key][1] = newAttrs

    outputPath = rf'{conductivityToggles.outputFolder}\conductivity.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
