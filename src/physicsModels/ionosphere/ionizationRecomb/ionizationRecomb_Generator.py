# --- ionizationRecomb_Generator.py ---
# Description: For REAL data Use a ionizationRecomb Methods to create electron density
# altitude profiles via Fang Parameterization.

# --- imports ---
from src.physicsModels.ionosphere.simToggles_Ionosphere import *
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import *
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.physicsModels.ionosphere.ionizationRecomb.ionizationRecomb_classes import *
from src.physicsModels.ionosphere.PlasmaEnvironment.plasmaEnvironment_classes import *
import numpy as np
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata
from tqdm import tqdm



def generateIonizationRecomb(ionizationRecombToggles):

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
    data_dict = {
                 'ne_IonRcmb': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'cm^-3', 'LABLAXIS': 'Electron Density'}],
                 'simAlt': [deepcopy(GenToggles.simAlt), {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}],
                  'qtot': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'm^-3s^-1', 'LABLAXIS': 'qtot'}],
                 'recombRate': [[], {'DEPEND_0': 'Epoch','DEPEND_1':'simAlt', 'UNITS': 'm^3s^-1', 'LABLAXIS': 'Recombination Rate'}],
                 'Epoch': [deepcopy(data_dict_backScatter['Epoch'][0]), {'DEPEND_0': None, 'UNITS': 'ns', 'LABLAXIS': 'Epoch'}],
                 }

    ###########################
    # --- GET THE BEAM DATA ---
    ###########################
    Epoch = deepcopy(data_dict['Epoch'][0])
    alt_range = deepcopy(data_dict['simAlt'][0])


    ################################
    # --- --- --- --- --- --- --- --
    # --- LOOP THROUGH BEAM DATA ---
    # --- --- --- --- --- --- --- --
    ################################

    data_dict['recombRate'][0] = np.zeros(shape=(len(Epoch),len(alt_range)))
    data_dict['qtot'][0] = np.zeros(shape=(len(Epoch), len(alt_range)))
    data_dict['ne_IonRcmb'][0] = np.zeros(shape=(len(Epoch), len(alt_range)))

    for tmeIdx in tqdm(range(len(Epoch))):

        ############################
        # --- RECOMBINATION RATE ---
        ############################
        # get the recombination rate. It does NOT change over the epoch
        model = schunkNagy2009()
        recombRate = model.calcRecombinationRate(alt_range, data_dict_plasma)
        data_dict['recombRate'][0][tmeIdx] = recombRate

        ##########################
        # --- IONIZATION RATE  ---
        ##########################

        # get the beam data for ALL pitch angles between 0 to 90deg
        jN_beam = data_dict_backScatter['jN_beam'][0][tmeIdx][0:10+1]
        jN_sec = data_dict_backScatter['jN_dgdPrim'][0][tmeIdx][0:10+1]
        jN_dgdPrim = data_dict_backScatter['jN_sec'][0][tmeIdx][0:10+1]
        beam_energyGrid = data_dict_backScatter['beam_energy_Grid'][0]
        response_energyGrid = data_dict_backScatter['energy_Grid'][0]

        # integrate beam data to get parallel energy flux (Phi_E) in keV/cm^-2s^-1



        # --- Get the energy/energyFluxes of the incident beam ---
        monoEnergyProfile = np.array([0.01, 0.1, 1, 10, 100, 1000])  # 100eV and 100keV, IN UNITS OF KEV
        energyFluxProfile = (6.242E8) * np.array([1 for i in range( len(monoEnergyProfile))])  # provide in ergs but convert from ergs/cm^-2s^-1 to keV/cm^-2s^-1

        # CHOOSE THE MODEL
        model = fang2010(alt_range, data_dict_neutral, data_dict_plasma, monoEnergyProfile, energyFluxProfile)
        H = model.scaleHeight()
        y = model.atmColumnMass(monoEnergyProfile)
        f = model.f(y, model.calcCoefficents(monoEnergyProfile))
        qtot = model.ionizationRate()  # in m^-3 s^-1

        ##################################
        # --- ELECTRON DENSITY (MODEL) ---
        ##################################
        q = np.sum(data_dict['qtot'][0], axis=0)  # sum up the recombination rates from all the incoming electrons
        alpha = data_dict['recombRate'][0]
        n_e = np.sqrt(q / alpha)  # in m^-3
        n_e_cm3 = n_e / (np.power(stl.cm_to_m, 3))






    #####################
    # --- OUTPUT DATA ---
    #####################

    # --- Construct the Data Dict ---
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}


    # update the data dict attrs
    for key, val in data_dict.items():
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict[key][1].items():
            newAttrs[subKey] = subVal

        data_dict[key][1] = newAttrs

    outputPath = rf'{ionizationRecombToggles.outputFolder}\ionizationRecomb.cdf'
    outputCDFdata(outputPath, data_dict)
