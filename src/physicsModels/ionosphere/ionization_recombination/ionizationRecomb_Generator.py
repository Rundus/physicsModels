# --- ionizationRecomb_Generator.py ---
# Description: For REAL data Use a ionization_recombination Methods to create electron density
# altitude profiles via Fang Parameterization.

# --- imports ---
from src.physicsModels.ionosphere.simToggles_Ionosphere import *
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import *
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_classes import *
from src.physicsModels.ionosphere.PlasmaEnvironment.plasmaEnvironment_classes import *
import numpy as np
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata
from tqdm import tqdm



def generateIonizationRecomb(GenToggles, ionizationRecombToggles):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################

    # get the ionospheric neutral data dict
    data_dict_neutral = loadDictFromFile(rf'{neutralsToggles.outputFolder}\neutralEnvironment.cdf')

    # get the ionospheric plasma data dict
    data_dict_plasma = loadDictFromFile(rf'{plasmaToggles.outputFolder}\plasmaEnvironment.cdf')

    # get the ionospheric backscatter data dict (Real Data)
    data_dict_backScatter = loadDictFromFile(rf'{backScatterToggles.outputFolder}\backScatter.cdf')

    # --- prepare the output data_dict ---
    data_dict_output = {
                 'ne_IonRecomb': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'm^-3', 'LABLAXIS': 'Electron Density'}],
                 'simAlt': [deepcopy(GenToggles.simAlt), {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}],
                  'q_total': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'm^-3s^-1', 'LABLAXIS': 'qtot'}],
                 'recombRate': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'm^3s^-1', 'LABLAXIS': 'Recombination Rate'}],
                 'Epoch': [deepcopy(data_dict_backScatter['Epoch'][0]), {'DEPEND_0': None, 'UNITS': 'ns', 'LABLAXIS': 'Epoch'}],
                 }

    Epoch = deepcopy(data_dict_output['Epoch'][0])
    alt_range = deepcopy(data_dict_output['simAlt'][0])

    ################################
    # --- --- --- --- --- --- --- --
    # --- LOOP THROUGH BEAM DATA ---
    # --- --- --- --- --- --- --- --
    ################################

    data_dict_output['recombRate'][0] = np.zeros(shape=(len(Epoch),len(alt_range)))
    data_dict_output['q_total'][0] = np.zeros(shape=(len(Epoch), len(alt_range)))
    data_dict_output['ne_IonRecomb'][0] = np.zeros(shape=(len(Epoch), len(alt_range)))

    for tmeIdx in tqdm(range(len(Epoch))):

        ############################
        # --- RECOMBINATION RATE ---
        ############################
        # get the recombination rate. It does NOT change over the epoch
        model = vickrey1982()
        alpha_total, alpha_profiles = model.calcRecombinationRate(alt_range, data_dict_plasma)
        data_dict_output['recombRate'][0][tmeIdx] = alpha_total

        num_flux_beam = deepcopy(data_dict_backScatter['num_flux_beam'][0][tmeIdx])
        num_flux_sec = deepcopy(data_dict_backScatter['num_flux_sec'][0][tmeIdx])
        num_flux_dgdPrim = deepcopy(data_dict_backScatter['num_flux_dgdPrim'][0][tmeIdx])

        # if the input data is good, only then fit it
        # print([ not np.any(num_flux_beam), not np.any(num_flux_sec), not np.any(num_flux_dgdPrim) ])
        if [np.any(num_flux_beam), np.any(num_flux_sec), np.any(num_flux_dgdPrim) ] == [True, True, True]:
            ##########################
            # --- IONIZATION RATE  ---
            ##########################

            # get the number flux data for the backscatter and number flux to units keV/cm^-2s^-1
            beam_energyGrid = deepcopy(data_dict_backScatter['beam_energy_Grid'][0][tmeIdx])
            response_energyGrid = deepcopy(data_dict_backScatter['energy_Grid'][0])

            engy_flux_beam = np.multiply(num_flux_beam, beam_energyGrid/1000)
            engy_flux_sec = np.multiply(num_flux_sec, response_energyGrid/1000)
            engy_flux_dgdPrim = np.multiply(num_flux_dgdPrim, response_energyGrid/1000)

            # --- Get the energy/energyFluxes of the incident beam + backscatter electrons ---
            monoEnergyProfile = np.append(response_energyGrid/1000, beam_energyGrid/1000)  # IN UNITS OF KEV
            energyFluxProfile = np.append(engy_flux_dgdPrim+engy_flux_sec, engy_flux_beam)

            # CHOOSE THE MODEL
            model = fang2010(alt_range, data_dict_neutral, data_dict_plasma, monoEnergyProfile, energyFluxProfile)
            q_profiles, q_total = model.ionizationRate()  # in cm^-3 s^-1
            data_dict_output['q_total'][0][tmeIdx] = q_total

            ##################################
            # --- ELECTRON DENSITY (MODEL) ---
            ##################################
            data_dict_output['ne_IonRecomb'][0][tmeIdx] = (stl.cm_to_m**3)*np.sqrt(deepcopy(data_dict_output['q_total'][0][tmeIdx]) / deepcopy(data_dict_output['recombRate'][0][tmeIdx]))  # in m^-3

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

    outputPath = rf'{ionizationRecombToggles.outputFolder}\ionizationRecomb.cdf'
    outputCDFdata(outputPath, data_dict_output)
