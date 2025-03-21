# --- ACESII_model_slice.py ---
# Description: compare the model results to the REAL ACES-II data
from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_classes import *
def generateIonizationRecomb():

    # --- imports ---
    from src.physicsModels.ionosphere.conductivity.conductivity_toggles import conductivityToggles
    import numpy as np
    from copy import deepcopy
    from spaceToolsLib.tools.CDF_output import outputCDFdata
    from tqdm import tqdm

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################

    # get the spatial data dict
    data_dict_conductivity = stl.loadDictFromFile(rf'{conductivityToggles.outputFolder}\conductivity.cdf')

    # get the ACES-II attitude data
    data_dict_attitude_high = stl.

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    LShellRange = data_dict_plasma['simLShell'][0]
    altRange = data_dict_plasma['simAlt'][0]

    data_dict_output = {
                        'ne_model': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm^-3', 'LABLAXIS': 'Electron Density'}],
                        'q_total': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1':'simAlt', 'UNITS': 'm^-3s^-1', 'LABLAXIS': 'qtot'}],
                         'alpha_recomb_total': [np.zeros(shape=(len(LShellRange), len(altRange))), {'DEPEND_0': 'simLShell', 'DEPEND_1':'simAlt', 'UNITS': 'm^3s^-1',  'LABLAXIS': 'Recombination Rate'}],
                        'simLShell': deepcopy(data_dict_plasma['simLShell']),
                        'simAlt': deepcopy(data_dict_plasma['simAlt'])
                        }


    ############################
    # --- RECOMBINATION RATE ---
    ############################
    model = schunkNagy2009()
    alpha_total, alpha_profiles = model.calcRecombinationRate(altRange=altRange, data_dict_plasma=data_dict_plasma)
    data_dict_output['alpha_recomb_total'][0] = alpha_total/(np.power(stl.cm_to_m,3)) # convert form cm^3/s to m^3/s
    for idx, ionNam in enumerate(plasmaToggles.wIons):
        data_dict_output = {**data_dict_output,
                            **{f'alpha_recomb_{ionNam}':[alpha_profiles[idx], {'DEPEND_0': 'simLShell', 'DEPEND_1':'simAlt', 'UNITS': 'm^3s^-1',  'LABLAXIS': f'Recombination Rate {ionNam}'}]}}


    # model = vickrey1982()
    # alpha_total, alpha_profiles = model.calcRecombinationRate(altRange=deepcopy(data_dict_spatial['grid_alt'][0]))
    # data_dict_output['alpha_recomb'][0] = alpha_total / (np.power(stl.cm_to_m, 3))  # convert form cm^3/s to m^3/s


    ################################
    # --- --- --- --- --- --- --- --
    # --- LOOP THROUGH BEAM DATA ---
    # --- --- --- --- --- --- --- --
    ################################
    for idx1, LShell in tqdm(enumerate(data_dict_output['simLShell'][0])):

        # get the flux data for the specific L-Shell
        dat_idx = np.abs(data_dict_LShell['L-Shell'][0] - LShell).argmin()
        varPhi_E_parallel_keV = deepcopy(data_dict_flux['varPhi_E_Parallel'][0][dat_idx])/1000
        Energy_keV = deepcopy(data_dict_flux['Energy'][0])/1000

        # CHOOSE THE MODEL
        model = fang2010(altRange=altRange,
                         Tn = deepcopy(data_dict_neutral['Tn'][0][idx1]),
                         m_eff_n = deepcopy(data_dict_neutral['m_eff_n'][0][idx1]),
                         rho_n = deepcopy(data_dict_neutral['rho_n'][0][idx1]),
                         inputEnergies=Energy_keV,
                         varPhi_E=varPhi_E_parallel_keV)

        q_profiles, q_total = model.ionizationRate()  # in cm^-3 s^-1
        data_dict_output['q_total'][0][idx1] = q_total*np.power(stl.cm_to_m,3) # convert to m^-3 s^-1

        ##################################
        # --- ELECTRON DENSITY (MODEL) ---
        ##################################
        data_dict_output['ne_model'][0][idx1] = np.sqrt(deepcopy(data_dict_output['q_total'][0][idx1]) / deepcopy(data_dict_output['alpha_recomb_total'][0][idx1]))  # in m^-3


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

    outputPath = rf'{ionizationRecombToggles.outputFolder}\ionization_rcomb.cdf'
    outputCDFdata(outputPath, data_dict_output)
