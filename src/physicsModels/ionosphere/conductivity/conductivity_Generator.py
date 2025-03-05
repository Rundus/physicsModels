# --- conductivity_Generator.py ---
# Description: Model the ionospheric conductivity

# --- imports ---
from src.physicsModels.ionosphere.simToggles_Ionosphere import *
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.physicsModels.ionosphere.conductivity.conductivity_classes import *
import numpy as np
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata

def generateIonosphericConductivity(GenToggles, conductivityToggles, **kwargs):

    #######################
    # --- LOAD THE DATA ---
    #######################
    # get the geomagnetic field data dict
    data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simFolderPath}\geomagneticField\geomagneticField.cdf')

    # get the ionospheric neutral data dict
    data_dict_neutral = loadDictFromFile(rf'{GenToggles.simFolderPath}\neutralEnvironment\neutralEnvironment.cdf')

    # get the ionospheric plasma data dict
    data_dict_plasma = loadDictFromFile(rf'{GenToggles.simFolderPath}\plasmaEnvironment\plasmaEnvironment.cdf')

    ###################################
    # --- BEAM - n(z, ILat) profile ---
    ###################################
    if conductivityToggles.use_eepaa_beam:# get the model electron density from raw data
        data_dict_ni_beam = ''
        ni_beam = ''
    elif conductivityToggles.use_evans1974_beam:# get the model electron density from recombination/ionization
        data_dict_ni_beam = loadDictFromFile(rf'{GenToggles.simFolderPath}\ionizationRecomb\ionizationRecomb.cdf')
        ni_beam = deepcopy(data_dict_ni_beam['ne_IonRecomb'][0])
    else:
        raise Exception('Must select dataset for n(z,ILat) BEAM')

    #########################################
    # --- BACKGROUND - n(z, ILat) profile ---
    #########################################
    if conductivityToggles.use_eepaa_background:
        data_dict_ni_background = loadDictFromFile(rf'C:\Data\ACESII\science\Langmuir\ni_spectrum.cdf')
        ni_background = deepcopy(data_dict_ni_beam['ni_spectrum'][0])
    elif conductivityToggles.use_IRI_background:
        ni_background = data_dict_plasma['ne'][0]
    else:
        raise Exception('Must select dataset for n(z,ILat) BACKGROUND')

    # construct the density profile n(z, ILat) - Inverted-V density + Ionospheric base density
    n_e = ni_beam + ni_background

    # prepare the output
    data_dict_output = {'Epoch': deepcopy(data_dict_ni_beam['Epoch']),
                        'simAlt': deepcopy(data_dict_ni_beam['simAlt']),
                        'ILat': deepcopy(data_dict_ni_beam['ILat']),
                        'n_e': [n_e, {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt', 'UNITS': 'm^-3','LABLAXIS': 'Electron Density'}]
                        }

    #################################
    # --- Electron Collision Freq ---
    #################################
    # electron-neutral collisions - Depends on Te and Nn
    model = Leda2019()
    nu_en_profiles = [model.electronNeutral_CollisionFreq(data_dict_neutral=data_dict_neutral,
                                                 data_dict_plasma=data_dict_plasma,
                                                 neutralKey=key) for key in neutralsToggles.wNeutrals]

    nu_en_total = np.array([np.sum(nu_en_profiles, axis=0) for tme_idx in range(len(data_dict_output['Epoch'][0]))])


    # electron-ion collisions - Depends on Te and Ne
    model = Johnson1961()
    nu_ei_total = np.array([model.electronIon_CollisionFreq(data_dict_neutral, data_dict_plasma, ne_data=n_e[tme_idx]) for tme_idx in range(len(data_dict_output['Epoch'][0]))])

    # total electron collision rate
    nu_e_total = nu_en_total + nu_ei_total

    # store the data
    data_dict_output = {**data_dict_output, **{'nu_e': [nu_e_total, {'DEPEND_0': 'Epoch', 'DEPEND_1':'simAlt', 'UNITS': '1/s','LABLAXIS':'nu_e'}]}}

    ############################
    # --- Ion Collision Freq ---
    ############################
    # Determine the collision frequencies for the various different ions
    model = Leda2019()
    nu_in_profiles = [model.ionNeutral_CollisionsFreq(data_dict_neutral=data_dict_neutral,
                                             data_dict_plasma=data_dict_plasma,
                                             ionKey=key) for key in plasmaToggles.wIons]  # NOp, Op, O2p

    # individual collision freqs
    for idx, key in enumerate(plasmaToggles.wIons):
        nu_in_specific = np.array([nu_in_profiles[idx] for i in range(len(data_dict_output['Epoch'][0]))])
        data_dict_output = {**data_dict_output, **{f'nu_i_{key}': [nu_in_specific,
                                                                  {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt',
                                                                   'UNITS': '1/s', 'LABLAXIS': f'ne_i_{key}'}]}}
    ##################
    # --- Mobility ---
    ##################
    # electrons
    nu_e = data_dict_output['nu_e'][0]
    Omega_e = data_dict_plasma['Omega_e'][0]
    kappa_e = np.divide(Omega_e, nu_e)
    data_dict_output = {**data_dict_output, **{'kappa_e': [kappa_e, {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'kappa_e'}]}}

    # ions
    for idx, key in enumerate(plasmaToggles.wIons):
        Omega_i_specific = deepcopy(data_dict_plasma[f'Omega_{key}'][0])
        kappa_i_specific = np.divide(Omega_i_specific, data_dict_output[f'nu_i_{key}'][0])
        data_dict_output = {**data_dict_output, **{f'kappa_i_{key}': [kappa_i_specific, {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': f'kappa_i_{key}'}]}}

    ######################
    # --- Conductivity ---
    ######################
    B_geo = data_dict_Bgeo['Bgeo'][0]

    # calculated electron sigmas
    sigma_par_e = q0 * np.multiply(np.divide(n_e, B_geo), kappa_e)
    sigma_P_e = q0 * np.multiply(np.divide(n_e, B_geo), kappa_e/(1 + np.power(kappa_e, 2)))
    sigma_H_e = q0 * np.multiply(np.divide(n_e, B_geo), np.power(kappa_e, 2)/(1 + np.power(kappa_e, 2)))

    # calculated ion sigmas
    sigma_para_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['Epoch'][0]), len(data_dict_output['simAlt'][0])))
    sigma_P_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['Epoch'][0]), len(data_dict_output['simAlt'][0])))
    sigma_H_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['Epoch'][0]), len(data_dict_output['simAlt'][0])))

    for idx, key in enumerate(plasmaToggles.wIons):
        kappa_val = deepcopy(data_dict_output[f'kappa_i_{key}'][0])
        specific_ion_concentration = np.divide(np.array([deepcopy(data_dict_plasma[f'n_{key}'][0]) for val in range(len(data_dict_output['Epoch'][0]))]), data_dict_plasma['ni'][0])
        n_i = n_e*specific_ion_concentration
        sigma_para_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), kappa_val)
        sigma_P_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), kappa_val / (1 + np.power(kappa_val, 2)))
        sigma_H_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), np.power(kappa_val, 2) / (1 + np.power(kappa_val, 2)))

    sigma_Pedersen = sigma_P_e + np.sum(sigma_P_ions, axis=0)
    sigma_Hall = sigma_H_e - np.sum(sigma_H_ions, axis=0)
    sigma_Parallel = sigma_par_e + np.sum(sigma_para_ions, axis=0)

    data_dict_output = {**data_dict_output,
                 **{'sigma_P': [sigma_Pedersen, {'DEPEND_0': 'Epoch', 'DEPEND_1':'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Pedersen Conductivity'}]},
                 **{'sigma_H': [sigma_Hall, {'DEPEND_0': 'Epoch', 'DEPEND_1':'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Hall Conductivity'}]},
                 **{'sigma_Para': [sigma_Parallel, {'DEPEND_0': 'Epoch', 'DEPEND_1':'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Parallel Conductivity'}]},
                 }


    ##########################################
    # --- Height-Integrated Conductivities ---
    ##########################################
    from scipy.integrate import trapz
    Sigma_Hall_HI = np.array([trapz(y=sigma_Hall[tme_idx], x=data_dict_output['simAlt'][0]) for tme_idx in range(len(data_dict_output['Epoch'][0]))])
    Sigma_Pedersen_HI = np.array([trapz(y=sigma_Pedersen[tme_idx], x=data_dict_output['simAlt'][0]) for tme_idx in range(len(data_dict_output['Epoch'][0]))])
    Sigma_Parallel_HI = np.array([trapz(y=sigma_Parallel[tme_idx], x=data_dict_output['simAlt'][0]) for tme_idx in range(len(data_dict_output['Epoch'][0]))])

    data_dict_output = {**data_dict_output,
                 **{'Sigma_Hall': [Sigma_Hall_HI, {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                 **{'Sigma_Pedersen': [Sigma_Pedersen_HI, {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                 **{'Sigma_Parallel': [Sigma_Parallel_HI, {'DEPEND_0': 'Epoch', 'DEPEND_1': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
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

    outputPath = rf'{GenToggles.simFolderPath}\conductivity\conductivity.cdf'
    outputCDFdata(outputPath, data_dict_output)
