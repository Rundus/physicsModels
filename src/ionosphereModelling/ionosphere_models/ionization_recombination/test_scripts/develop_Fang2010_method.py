import matplotlib.pyplot as plt
import spaceToolsLib as stl
import numpy as np
from copy import deepcopy
from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_classes import *
from src.physicsModels.ionosphere.plasma_environment.plasma_toggles import plasmaToggles
from tqdm import tqdm

data_dict_flux_high = stl.loadDictFromFile('C:\Data\ACESII\L3\Energy_Flux\high\ACESII_36359_l3_eepaa_flux.cdf')
data_dict_flux_low = stl.loadDictFromFile('C:\Data\ACESII\L3\Energy_Flux\low\ACESII_36364_l3_eepaa_flux.cdf')
data_dict_spatial = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\spatial_environment\spatial_environment.cdf')
data_dict_plasma = stl.loadDictFromFile(rf'{plasmaToggles.outputFolder}\plasma_environment.cdf')
data_dict_neutral = stl.loadDictFromFile(r'C:\Data\physicsModels\ionosphere\neutral_environment\neutral_environment.cdf')
data_dict_diffNFlux = stl.loadDictFromFile(r'C:\Data\ACESII\L2\high\ACESII_36359_l2_eepaa_fullCal.cdf')

# prepare the data
altRange = data_dict_spatial['simAlt'][0]
altLShell = data_dict_spatial['simLShell'][0]

# get the EEPAA energy flux in  keV/cm^-2s^-1
wIdx = 0
q_output = np.zeros(shape=(len(data_dict_flux_high['Epoch'][0]), len(altRange)))

for tme, val in tqdm(enumerate(data_dict_flux_high['Epoch'][0])):

    # get the input data for the payload
    altitude = data_dict_diffNFlux['Alt'][0]
    simAlt_idx = np.abs(altRange-altitude[tme]).argmin()
    energies = data_dict_flux_high['Energy'][0]/1000 # convert to keV


    # --- get the PARALLEL response ---
    varPhi_E_para = data_dict_flux_high['varPhi_E_Parallel'][0][tme] / 1000  # convert to keV
    model = fang2010(altRange=altRange,
                     Tn=data_dict_neutral['Tn'][0][wIdx],
                     m_eff_n=data_dict_neutral['m_eff_n'][0][wIdx],
                     rho_n=data_dict_neutral['rho_n'][0][wIdx],
                     inputEnergies=energies,
                     varPhi_E=varPhi_E_para)


    # limit the model results to below the parallel response
    q_profiles, q_total = model.ionizationRate()  # in cm^-3 s^-1
    q_total[:simAlt_idx] = 0
    q_output[tme] += q_total


    # --- get the ANTI-PARALLEL response ---
    varPhi_E_antiPara = data_dict_flux_high['varPhi_E_antiParallel'][0][tme] / 1000  # convert to keV
    model = fang2010(altRange=altRange,
                     Tn=data_dict_neutral['Tn'][0][wIdx],
                     m_eff_n=data_dict_neutral['m_eff_n'][0][wIdx],
                     rho_n=data_dict_neutral['rho_n'][0][wIdx],
                     inputEnergies=energies,
                     varPhi_E=varPhi_E_antiPara)

    # limit the model results to below the parallel response
    q_profiles, q_total = model.ionizationRate()  # in cm^-3 s^-1
    q_total[simAlt_idx:] = 0
    q_output[tme] += q_total


data_dict_output = {'qtot':[q_output, {}],
                    'simLShell':deepcopy(data_dict_spatial['simLShell']),
                    'simAlt':deepcopy(data_dict_spatial['simAlt']),
                    'Epoch': deepcopy(data_dict_flux_high['Epoch'])
                    }

stl.outputCDFdata(outputPath=r'C:\Data\physicsModels\ionosphere\ionizationRecomb\testScripts\testData.cdf',
                data_dict=data_dict_output)

