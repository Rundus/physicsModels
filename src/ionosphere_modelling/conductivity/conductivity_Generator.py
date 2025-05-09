# --- conductivity_Generator.py ---
# Description: Model the ionospheric conductivity
from src.ionosphere_modelling.conductivity.conductivity_classes import *
def generateIonosphericConductivity():

    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.ionosphere_modelling.sim_toggles import SimToggles
    from src.ionosphere_modelling.spatial_environment.spatial_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.ionosphere_modelling.ionization_recombination.ionizationRecomb_toggles import ionizationRecombToggles
    from src.ionosphere_modelling.neutral_environment.neutral_toggles import neutralsToggles
    from src.ionosphere_modelling.plasma_environment.plasma_toggles import plasmaToggles
    from src.ionosphere_modelling.conductivity.conductivity_toggles import conductivityToggles
    from scipy.integrate import simpson

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    # get the geomagnetic field data dict
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\geomagneticField\*.cdf*')[0])

    # get the ionospheric neutral data dict
    data_dict_neutral = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\neutral_environment\*.cdf*')[0])

    # get the ACES-II EEPAA Flux data
    data_dict_flux = stl.loadDictFromFile(rf'{ionizationRecombToggles.flux_path}\ACESII_36359_l3_eepaa_flux.cdf')

    # get the ACES-II L-Shell data
    data_dict_LShell = deepcopy(SpatialToggles.data_dict_HF_LShell)

    # get the ionospheric plasma data dict
    data_dict_plasma = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\plasma_environment\*.cdf*')[0])


    # get the ionization-recombination data dict
    data_dict_ionRecomb = stl.loadDictFromFile(glob(rf'{SimToggles.sim_root_path}\ionization_rcomb\*.cdf*')[0])

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

        'sigma_Pedersen_e': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Pedersen Conductivity'}],
        'sigma_Pedersen_i': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Pedersen Conductivity'}],
        'sigma_Hall_e': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Hall Conductivity'}],
        'sigma_Hall_i': [np.zeros(shape=(len(LShellRange), len(altRange))),{'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Hall Conductivity'}],

        'sigma_Hall': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Hall Conductivity'}],
        'sigma_parallel': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Specific Parallel Conductivity'}],
    }

    ###############################
    # --- CONSTRUCT n_e PROFILE ---
    ###############################

    # construct the background density profile n(z, ILat) - Inverted-V density + Ionospheric base density
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


    ######################
    # --- Conductivity ---
    ######################
    B_geo = data_dict_Bgeo['Bgeo'][0]

    # calculated electron sigmas
    sigma_parallel_e = q0 * np.multiply(np.divide(data_dict_output['ne_total'][0], B_geo), kappa_e)
    sigma_P_e = q0 * np.multiply(np.divide(data_dict_output['ne_total'][0], B_geo), kappa_e/(1 + np.power(kappa_e, 2)))
    sigma_H_e = q0 * np.multiply(np.divide(data_dict_output['ne_total'][0], B_geo), np.power(kappa_e, 2)/(1 + np.power(kappa_e, 2)))

    # calculated ion sigmas
    sigma_parallel_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['simLShell'][0]), len(data_dict_output['simAlt'][0])))
    sigma_P_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['simLShell'][0]), len(data_dict_output['simAlt'][0])))
    sigma_H_ions = np.zeros(shape=(len(plasmaToggles.wIons), len(data_dict_output['simLShell'][0]), len(data_dict_output['simAlt'][0])))

    for idx, key in enumerate(plasmaToggles.wIons):
        kappa_val = deepcopy(data_dict_output[f'kappa_i_{key}'][0])
        # specific_ion_concentration = np.divide(data_dict_plasma[f'n_{key}'][0], data_dict_plasma['ni'][0])
        n_i = data_dict_output['ne_total'][0]*deepcopy(data_dict_plasma[f'C_{key}'][0])
        sigma_parallel_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), kappa_val)
        sigma_P_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), kappa_val / (1 + np.power(kappa_val, 2)))
        sigma_H_ions[idx] = q0 * np.multiply(np.divide(n_i, B_geo), np.power(kappa_val, 2) / (1 + np.power(kappa_val, 2)))


    # clean the data
    sigma_Pedersen =sigma_P_e + np.sum(sigma_P_ions, axis=0)
    sigma_Pedersen[sigma_Pedersen<0] = 0
    sigma_Hall = sigma_H_e - np.sum(sigma_H_ions, axis=0)
    sigma_Hall[sigma_Hall<0] = 0
    sigma_Parallel = sigma_parallel_e + np.sum(sigma_parallel_ions, axis=0)
    sigma_Parallel[sigma_Parallel<0] = 0

    # Store the data
    data_dict_output['sigma_Pedersen'][0] = sigma_Pedersen
    data_dict_output['sigma_Hall'][0] = sigma_Hall
    data_dict_output['sigma_parallel'][0] = sigma_Parallel

    data_dict_output['sigma_Pedersen_e'][0] = sigma_P_e
    data_dict_output['sigma_Hall_e'][0] = sigma_H_e

    data_dict_output['sigma_Pedersen_i'][0] = np.sum(sigma_P_ions, axis=0)
    data_dict_output['sigma_Hall_i'][0] = np.sum(sigma_H_ions, axis=0)

    ##########################################
    # --- Height-Integrated Conductivities ---
    ##########################################
    Sigma_Hall_HI = np.array([simpson(y=data_dict_output['sigma_Hall'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
    Sigma_Pedersen_HI = np.array([simpson(y=data_dict_output['sigma_Pedersen'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])
    Sigma_Parallel_HI = np.array([simpson(y=data_dict_output['sigma_parallel'][0][L_idx], x=data_dict_output['simAlt'][0]) for L_idx in range(len(data_dict_output['simLShell'][0]))])

    data_dict_output = {**data_dict_output,
                        **{'Sigma_Hall': [Sigma_Hall_HI, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                        **{'Sigma_Pedersen': [Sigma_Pedersen_HI, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                        **{'Sigma_parallel': [Sigma_Parallel_HI, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                        }

    ##############################
    # --- OTHER CONDUCTIVITIES ---
    ##############################

    # --- Robinson Height-Integrated Conductivities ---
    Sigma_Hall_Robinson = np.zeros(shape=(len(data_dict_output['simLShell'][0])))
    Sigma_Pedersen_Robinson = np.zeros(shape=(len(data_dict_output['simLShell'][0])))

    # --- KAEPPLER Height-Integrated Conductivities ---
    Sigma_Hall_K10 = np.zeros(shape=(len(data_dict_output['simLShell'][0])))
    Sigma_Pedersen_K10 = np.zeros(shape=(len(data_dict_output['simLShell'][0])))
    Sigma_Hall_K50 = np.zeros(shape=(len(data_dict_output['simLShell'][0])))
    Sigma_Pedersen_K50 = np.zeros(shape=(len(data_dict_output['simLShell'][0])))
    Sigma_Hall_K90 = np.zeros(shape=(len(data_dict_output['simLShell'][0])))
    Sigma_Pedersen_K90 = np.zeros(shape=(len(data_dict_output['simLShell'][0])))

    # Reduce the EEPAA data to the relevant subset
    for idx1, LShell in enumerate(data_dict_output['simLShell'][0]):
        # get the flux data for the specific L-Shell
        dat_idx = np.abs(data_dict_LShell['L-Shell'][0] - LShell).argmin()
        energy_flux_ergs = (1/stl.erg_to_eV) * deepcopy(data_dict_flux['Phi_E_Parallel'][0][dat_idx])  # convert energy flux to ergs/cm^2 s^1
        energy_flux_watts = (1000*stl.q0*stl.cm_to_m*stl.cm_to_m)*deepcopy(data_dict_flux['Phi_E_Parallel'][0][dat_idx])  # convert energy flux to mW/m^2
        avgEnergy_keV = deepcopy(data_dict_flux['Energy_avg'][0][dat_idx]) / 1000  # convert to keV

        # robinson
        Sigma_Pedersen_Robinson[idx1] = ((40 * avgEnergy_keV) / (16 + np.power(avgEnergy_keV, 2))) * np.sqrt(energy_flux_ergs)
        Sigma_Hall_Robinson[idx1] = (0.45) * np.power(avgEnergy_keV, 0.85) * Sigma_Pedersen_Robinson[idx1]

        # kaeppler
        Sigma_Pedersen_K50[idx1] = 4.93 * np.power(energy_flux_watts, 0.48)
        Sigma_Hall_K50[idx1] = 8.11 * np.power(energy_flux_watts, 0.55)

        Sigma_Pedersen_K90[idx1] = 5.9*np.power(energy_flux_watts,0.48)
        Sigma_Hall_K90[idx1] = 10.77 * np.power(energy_flux_watts, 0.53)

        Sigma_Pedersen_K10[idx1] = 3.5 * np.power(energy_flux_watts, 0.49)
        Sigma_Hall_K10[idx1] = 5.19 * np.power(energy_flux_watts, 0.56)

    data_dict_output = {**data_dict_output,
                        **{'Sigma_Hall_Robinson': [Sigma_Hall_Robinson, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                        **{'Sigma_Pedersen_Robinson': [Sigma_Pedersen_Robinson, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                        **{'Sigma_Hall_K10': [Sigma_Hall_K10, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                        **{'Sigma_Pedersen_K10': [Sigma_Pedersen_K10, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                        **{'Sigma_Hall_K50': [Sigma_Hall_K50, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                        **{'Sigma_Pedersen_K50': [Sigma_Pedersen_K50, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                        **{'Sigma_Hall_K90': [Sigma_Hall_K90, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                        **{'Sigma_Pedersen_K50': [Sigma_Pedersen_K90, {'DEPEND_0': 'simLShell', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
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

    outputPath = rf'{conductivityToggles.outputFolder}\conductivity.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
