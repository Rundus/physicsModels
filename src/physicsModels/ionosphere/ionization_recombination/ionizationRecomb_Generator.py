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
                 'ne_IonRecomb': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'cm^-3', 'LABLAXIS': 'Electron Density'}],
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
    # for tmeIdx in tqdm([0,1]):

        ############################
        # --- RECOMBINATION RATE ---
        ############################
        # get the recombination rate. It does NOT change over the epoch
        model = schunkNagy2009()
        recombRate = model.calcRecombinationRate(alt_range, data_dict_plasma)
        data_dict_output['recombRate'][0][tmeIdx] = recombRate

        ##########################
        # --- IONIZATION RATE  ---
        ##########################

        # get the number flux data for the backscatter and number flux to units keV/cm^-2s^-1
        beam_energyGrid = deepcopy( data_dict_backScatter['beam_energy_Grid'][0][tmeIdx])
        response_energyGrid =deepcopy( data_dict_backScatter['energy_Grid'][0])
        engy_flux_beam = np.multiply(deepcopy(data_dict_backScatter['num_flux_beam'][0][tmeIdx]), beam_energyGrid/1000)
        engy_flux_sec = np.multiply(deepcopy(data_dict_backScatter['num_flux_sec'][0][tmeIdx]), response_energyGrid/1000)
        engy_flux_dgdPrim = np.multiply(deepcopy(data_dict_backScatter['num_flux_dgdPrim'][0][tmeIdx]), response_energyGrid/1000)

        # --- Get the energy/energyFluxes of the incident beam + backscatter electrons ---
        monoEnergyProfile = np.append(response_energyGrid/1000, beam_energyGrid/1000)  # IN UNITS OF KEV
        energyFluxProfile = np.append(engy_flux_dgdPrim+engy_flux_sec, engy_flux_beam)

        # CHOOSE THE MODEL
        model = fang2010(alt_range, data_dict_neutral, data_dict_plasma, monoEnergyProfile, energyFluxProfile)
        q_profiles = model.ionizationRate()  # in m^-3 s^-1
        data_dict_output['q_total'][0][tmeIdx] = np.sum(q_profiles)

        ##################################
        # --- ELECTRON DENSITY (MODEL) ---
        ##################################
        q = deepcopy(data_dict_output['q_total'][0][tmeIdx])  # sum up the recombination rates from all the incoming electrons
        alpha = deepcopy(data_dict_output['recombRate'][0][tmeIdx])
        n_e = np.sqrt(q / alpha)  # in m^-3
        data_dict_output['ne_IonRecomb'][0][tmeIdx] = n_e / (np.power(stl.cm_to_m, 3))

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
