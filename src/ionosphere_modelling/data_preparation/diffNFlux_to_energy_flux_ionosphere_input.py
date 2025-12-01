# --- diffFlux_to_Energy_Flux_ionosphere_input.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: using the specs of the ACESII ESAs, convert from differential Energy Flux
# to just energy flux as described in EEPAA_Flux_Conversion.pdf document in Overleaf
# BUT ONLY INCLUDE the flux which reaches a certain altitude in the ionosphere

# --- bookkeeping ---
# !/usr/bin/env python
__author__ = "Connor Feltman"
__date__ = "2022-08-22"
__version__ = "1.0.0"
import numpy as np
# --- --- --- --- ---


# --- --- --- ---
# --- TOGGLES ---
# --- --- --- ---
justPrintFileNames = False
wRocket = 4

# Don't consider pitch angles 80 or 90 degrees
# since at 350km they only make it to ~280km, well above our simulation
# whereas 70deg at 400km can reach 120km.
cutoff_pitch = 80


# ---------------------------
outputData = True
# ---------------------------

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
# from scipy.integrate import trapz
from scipy.integrate import simpson
import spaceToolsLib as stl
from tqdm import tqdm
from glob import glob
from copy import deepcopy


def diffFlux_to_Energy_Flux_ionosphere_input():

    # --- --- --- --- --- -
    # --- LOAD THE DATA ---
    # --- --- --- --- --- -
    # --- get the data from the file ---
    data_dict_eepaa = stl.loadDictFromFile(inputFilePath=glob('C:/Data/ACESII/L2/high/*l2_eepaa_fullCal.cdf*')[0])

    # --- --- --- --- -
    # --- INTEGRATE ---
    # --- --- --- --- -
    stl.prgMsg('Calculating Fluxes')
    Epoch = data_dict_eepaa['Epoch'][0]
    Pitch = data_dict_eepaa['Pitch_Angle'][0][1:20] # only get pitch angles 0deg to 180deg
    Energy = data_dict_eepaa['Energy'][0]
    diffNFlux = data_dict_eepaa['Differential_Number_Flux'][0][:, 1:20, :] # ONLY get 0deg to 180deg

    # Number Fluxes
    Phi_N = np.zeros(shape=(len(Epoch)))
    Phi_N_antiParallel = np.zeros(shape=(len(Epoch)))
    Phi_N_Parallel = np.zeros(shape=(len(Epoch)))
    varPhi_N = np.zeros(shape=(len(Epoch), len(Energy)))
    varPhi_N_antiParallel = np.zeros(shape=(len(Epoch), len(Energy)))
    varPhi_N_Parallel = np.zeros(shape=(len(Epoch), len(Energy)))

    # Energy Fluxes
    Phi_E = np.zeros(shape=(len(Epoch)))
    Phi_E_antiParallel = np.zeros(shape=(len(Epoch)))
    Phi_E_Parallel = np.zeros(shape=(len(Epoch)))
    varPhi_E = np.zeros(shape=(len(Epoch), len(Energy)))
    varPhi_E_antiParallel = np.zeros(shape=(len(Epoch), len(Energy)))
    varPhi_E_Parallel = np.zeros(shape=(len(Epoch), len(Energy)))

    # Average Energy
    Energy_avg_Robinson = np.zeros(shape=(len(Epoch)))
    cutoff_idx = np.abs(Energy - 100).argmin()

    # input flux Pitch cutoff
    pitch_cutoff_idx = np.abs(Pitch-cutoff_pitch).argmin()
    print(Pitch)
    print(pitch_cutoff_idx)


    # determine the DeltaE to use for the varphi integrations - DeltaE = the half distance to the next energy value
    deltaEs = []
    for idx, engy in enumerate(Energy):
        if idx == len(Energy)-1:
            deltaEs.append(Energy[-2] - Energy[-1])
        elif idx == 0:
            deltaEs.append(Energy[0] - Energy[1])
        else:
            lowerE =(Energy[idx] - Energy[idx+1])/2
            highE = (Energy[idx-1] - Energy[idx])/2
            deltaEs.append(lowerE+highE)


    # --- perform the integration_files ---
    alpha_grid, engy_grid = np.meshgrid(Pitch, data_dict_eepaa['Energy'][0])

    for tme in tqdm(range(len(Epoch))):

        #######################
        # --- NUMBER FLUXES ---
        #######################
        # --- Integrate over pitch angle ---

        # omni directional
        prepared = np.transpose(diffNFlux[tme]*2*np.pi*np.sin(np.radians(alpha_grid.T)))
        JE_N = np.array([simpson(y=prepared[idx], x=np.radians(Pitch)) for idx, engy in enumerate(Energy)]) # J_N(E) diffNFlux without pitch

        # parallel
        prepared = np.transpose(diffNFlux[tme]*2*np.pi*np.sin(np.radians(alpha_grid.T))*np.cos(np.radians(alpha_grid.T)))
        JE_N_Parallel = np.array([simpson(y=prepared[idx][0:pitch_cutoff_idx], x=np.radians(Pitch[0:pitch_cutoff_idx])) for idx, engy in enumerate(Energy)])

        # anti-parallel
        prepared = np.transpose(diffNFlux[tme]*2*np.pi* np.sin(np.radians(alpha_grid.T)) * np.cos(np.radians(alpha_grid.T)))
        JE_N_antiParallel = np.array([simpson(y=prepared[idx][10:], x=np.radians(Pitch[10:])) for idx, engy in enumerate(Energy)])

        # --- partially integrate over energy. ---
        # Description: To get in units of [cm^-2 s^-1], we assume j(E) doesn't change over the DeltaE interval between samples.
        # The integral between E-DeltaE and E+DeltaE around a central energy E is just: varphi(E) = DeltaE(E) * J(E) where DeltaE(E) depends
        # on the central energy. In our detector, DeltaE(E) is designed to be ~18% always --> DeltaE(E) = (1+gamma)E -(1-gamma)E = 2*gamma*E
        varPhi_N[tme] = np.array([deltaEs[idx]*JE_N[idx] for idx, engy in enumerate(Energy)])
        varPhi_N_Parallel[tme] = np.array([deltaEs[idx]*JE_N_Parallel[idx] for idx, engy in enumerate(Energy)])
        varPhi_N_antiParallel[tme] = np.array([deltaEs[idx]*JE_N_antiParallel[idx] for idx, engy in enumerate(Energy)])

        # Integrate over energy
        Phi_N[tme] = np.array(-1*simpson(y=JE_N, x=Energy)).clip(min=0)
        Phi_N_antiParallel[tme] = np.array(-1*simpson(y=JE_N_antiParallel, x=Energy)).clip(min=0)
        Phi_N_Parallel[tme] = np.array(-1*simpson(y=JE_N_Parallel, x=Energy)).clip(min=0)

        # ---------------------
        # --- ENERGY FLUXES ---
        # ---------------------
        varPhi_E[tme] = Energy*deepcopy(varPhi_N[tme])
        varPhi_E_antiParallel[tme] = Energy * deepcopy(varPhi_N_antiParallel[tme])
        varPhi_E_Parallel[tme] = Energy * deepcopy(varPhi_N_Parallel[tme])

        # Integrate over energy
        Phi_E[tme] = np.array(-1*simpson(y=JE_N*Energy, x=Energy)).clip(min=0)
        Phi_E_antiParallel[tme] = np.array(-1*simpson(y=JE_N_antiParallel*Energy, x=Energy)).clip(min=0)
        Phi_E_Parallel[tme] = np.array(-1*simpson(y=JE_N_Parallel*Energy, x=Energy)).clip(min=0)

        # -------------------------
        # --- ROBINSON FORMULAE ---
        # -------------------------

        # calculate the average energy
        # NOTE: the low-energy electrons DON'T contribute to the height_integrated conductivity very much,
        # thus ONLY use the PRIMARY beam to calculate the average energy, which is ~ 116 eV
        Energy_avg_Robinson[tme] = np.array(-1*simpson(y=JE_N_Parallel[:cutoff_idx]*Energy[:cutoff_idx], x=Energy[:cutoff_idx]))/np.array(-1*simpson(y=JE_N_Parallel[:cutoff_idx], x=Energy[:cutoff_idx]))

        # -------------------------
        # --- KAEPPLER FORMULAE ---
        # -------------------------

    # --- --- --- --- --- --- ---
    # --- WRITE OUT THE DATA ---
    # --- --- --- --- --- --- ---

    if outputData:
        stl.prgMsg('Creating output file')

        data_dict_output = {'Phi_N': [Phi_N, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'Phi_N_antiParallel': [Phi_N_antiParallel, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'Phi_N_Parallel': [Phi_N_Parallel, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'varPhi_N': [varPhi_N, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'varPhi_N_antiParallel': [varPhi_N_antiParallel, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'varPhi_N_Parallel': [varPhi_N_Parallel, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],

                            'Phi_E': [Phi_E, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'Phi_E_antiParallel': [Phi_E_antiParallel, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'Phi_E_Parallel': [Phi_E_Parallel, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'varPhi_E': [varPhi_E, deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'varPhi_E_antiParallel': [varPhi_E_antiParallel,deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'varPhi_E_Parallel': [varPhi_E_Parallel,deepcopy(data_dict_eepaa['Differential_Energy_Flux'][1])],
                            'Pitch_Angle': data_dict_eepaa['Pitch_Angle'],
                            'Energy': data_dict_eepaa['Energy'],
                            'Epoch': data_dict_eepaa['Epoch'],
                            'Alt': data_dict_eepaa['Alt'],
                            'L-Shell':data_dict_eepaa['L-Shell'],
                            'Energy_avg': [Energy_avg_Robinson, data_dict_eepaa['Energy'][1]],
                            }

        data_dict_output['Phi_N'][1]['LABLAXIS'] = 'Number_Flux'
        data_dict_output['Phi_N'][1]['UNITS'] = 'cm!A-2!N s!A-1!N'

        data_dict_output['Phi_E'][1]['LABLAXIS'] = 'Energy_Flux'
        data_dict_output['Phi_E'][1]['UNITS'] = 'eV cm!A-2!N s!A-1!N'

        data_dict_output['varPhi_N'][1]['UNITS'] = 'cm!A-2!N s!A-1!N'
        data_dict_output['varPhi_N'][1]['DEPEND_1'] = 'Energy'
        data_dict_output['varPhi_N'][1]['DEPEND_2'] = None

        data_dict_output['varPhi_E'][1]['UNITS'] = 'eV cm!A-2!N s!A-1!N'
        data_dict_output['varPhi_E'][1]['DEPEND_1'] = 'Energy'
        data_dict_output['varPhi_E'][1]['DEPEND_2'] = None


        data_dict_output['Phi_N_antiParallel'][1]['LABLAXIS'] = 'Anti_Parallel_Number_Flux'
        data_dict_output['Phi_N_antiParallel'][1]['UNITS'] = 'cm!A-2!N s!A-1!N'

        data_dict_output['Phi_E_antiParallel'][1]['LABLAXIS'] = 'Anti_Parallel_Energy_Flux'
        data_dict_output['Phi_E_antiParallel'][1]['UNITS'] = 'eV cm!A-2!N s!A-1!N'

        data_dict_output['Phi_N_Parallel'][1]['LABLAXIS'] = 'Parallel_Number_Flux'
        data_dict_output['Phi_N_Parallel'][1]['UNITS'] = 'cm!A-2!N s!A-1!N'

        data_dict_output['Phi_E_Parallel'][1]['LABLAXIS'] = 'Parallel_Energy_Flux'
        data_dict_output['Phi_E_Parallel'][1]['UNITS'] = 'eV cm!A-2!N s!A-1!N'


        data_dict_output['varPhi_N_antiParallel'][1]['UNITS'] = 'cm!A-2!N s!A-1!N'
        data_dict_output['varPhi_N_antiParallel'][1]['DEPEND_1'] = 'Energy'
        data_dict_output['varPhi_N_antiParallel'][1]['DEPEND_2'] = None

        data_dict_output['varPhi_E_antiParallel'][1]['UNITS'] = 'eV cm!A-2!N s!A-1!N'
        data_dict_output['varPhi_E_antiParallel'][1]['DEPEND_1'] = 'Energy'
        data_dict_output['varPhi_E_antiParallel'][1]['DEPEND_2'] = None

        data_dict_output['varPhi_N_Parallel'][1]['UNITS'] = 'cm!A-2!N s!A-1!N'
        data_dict_output['varPhi_N_Parallel'][1]['DEPEND_1'] = 'Energy'
        data_dict_output['varPhi_N_Parallel'][1]['DEPEND_2'] = None

        data_dict_output['varPhi_E_Parallel'][1]['UNITS'] = 'eV cm!A-2!N s!A-1!N'
        data_dict_output['varPhi_E_Parallel'][1]['DEPEND_1'] = 'Energy'
        data_dict_output['varPhi_E_Parallel'][1]['DEPEND_2'] = None

        # write out the data
        fileoutName = f'ACESII_36359_l3_eepaa_flux_input_ionosphere.cdf'
        outputPath = f'C:\Data\physicsModels\ionosphere\data_inputs\energy_flux\high\\{fileoutName}'
        stl.outputDataDict(outputPath, data_dict_output)


# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---
diffFlux_to_Energy_Flux_ionosphere_input()