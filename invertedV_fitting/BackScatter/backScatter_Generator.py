# --- backScatter_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take in the primary data fits and produce secondary/backscatter curves for each. Use the
# parameterized Evans 1964 curves to determine the curves.

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from invertedV_fitting.BackScatter.backScatter_classes import *
from invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
import spaceToolsLib as stl
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
start_time = time()
# --- --- --- --- ---


def generateSecondaryBackScatter(GenToggles, primaryBeamToggles, secondaryBackScatterToggles, **kwargs):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=GenToggles.input_diffNFiles[GenToggles.wFlyerFit],
                                              wKeys_Reduce=['Differential_Energy_Flux',
                                                            'Differential_Number_Flux',
                                                            'Epoch',
                                                            'Differential_Number_Flux_stdDev'])

    data_dict_beamFits = stl.loadDictFromFile(inputFilePath=r"C:\Data\physicsModels\invertedV\primaryBeam_Fitting\primaryBeam_fitting_parameters.cdf")

    # --- prepare the output data_dict ---
    data_dict = {'Phi_dPrim': [[], {'DEPEND_0': 'Epoch','DEPEND_1': 'energy_Grid', 'UNITS': 'cm^-2 s^-1 eV^-1', 'LABLAXIS': 'degradedPrimaryFlux'}],
                 'Phi_sec': [[], {'DEPEND_0': 'Epoch','DEPEND_1': 'energy_Grid', 'UNITS': 'cm^-2 s^-1 eV^-1', 'LABLAXIS': 'secondaryFlux'}],
                 'Phi_Beam':[[], {'DEPEND_0': 'Epoch','DEPEND_1': 'energy_Grid', 'UNITS': 'cm^-2 s^-1 eV^-1', 'LABLAXIS': 'Beam'}],
                 'energy_Grid': [[], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                 'Epoch':[[], {'DEPEND_0': None, 'UNITS': 'ns', 'LABLAXIS': 'Epoch'}],
                 }

    # Re-construct the 5-Averaged data. Only the Epoch dimension is reduced from original data
    relevantPitchs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90deg
    EpochFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                                GenToggles=GenToggles,
                                                                                N_avg=primaryBeamToggles.numToAverageOver)

    #####################
    # --- ENERGY GRID ---
    #####################
    model_energyGrid = np.linspace(10, 1E4, secondaryBackScatterToggles.N_energyGrid)

    ######################################
    # ------------------------------------
    # --- CALCULATE IONOSPHERIC RESPONSE ---
    # ------------------------------------
    ######################################
    beam_Flux_output = np.zeros(shape=(len(EpochFitData),len(model_energyGrid)))
    secondaries_Flux_output = np.zeros(shape=(len(EpochFitData),len(model_energyGrid)))
    dPrim_Flux_output = np.zeros(shape=(len(EpochFitData), len(model_energyGrid)))

    for tmeIdx in range(len(EpochFitData)):

        # --- get the RAW data ---
        detector_Energies = data_dict_diffFlux['Energy'][0]
        detector_diffNFlux = diffNFlux_avg[tmeIdx, relevantPitchs, :]

        # calculate the OmniFlux - varPhi(E)
        detector_omniDiffFlux = backScatter_class().calcOmni_diffNFlux(diffNFlux=detector_diffNFlux,
                                                              pitchValues=data_dict_diffFlux['Pitch_Angle'][0][relevantPitchs],
                                                              energyValues=detector_Energies)

        # get the beam-part of the omniDiffFlux (assumes the relevant beam pitch angles are isotropic)
        dataIdx_set = data_dict_beamFits['dataIdxs'][0][0][tmeIdx] # for alpha = 10deg
        detector_beam_Energies = detector_Energies[np.where(dataIdx_set > 0)[0]]
        detector_beam_omniDiffFlux = detector_omniDiffFlux[np.where(dataIdx_set > 0)[0]]

        # --- Get the Model Beam Parameters ---
        # Description: Use the model to generate a "Beam". Limit the model beam by the energies of the actual raw data.
        V0_fitParam = data_dict_beamFits['V0'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]
        n0_fitParam = data_dict_beamFits['n'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]
        Te_fitParam = data_dict_beamFits['Te'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]

        # --- Form the model beam diffNFlux ---
        if primaryBeamToggles.wDistributionToFit == 'Kappa':
            kappa_fitParam = data_dict_beamFits['kappa'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]
            params = [n0_fitParam, Te_fitParam, V0_fitParam, kappa_fitParam]
            model_diffNFlux = primaryBeam_class().diffNFlux_fitFunc_Kappa(model_energyGrid, *params) # Assume the beam is isotropic
        elif primaryBeamToggles.wDistributionToFit == 'Maxwellian':
            params = [n0_fitParam, Te_fitParam, V0_fitParam]
            model_diffNFlux = primaryBeam_class().diffNFlux_fitFunc_Maxwellian(model_energyGrid, *params)

        model_omniDiffFlux = 1 * model_diffNFlux  # multiply by 1 since integral sin(x)dx from 0 to 90deg = 1 for an isotropic distribution

        # limit the model beam by the observed energy ranges
        detector_to_model_beamIndicies = np.array([i for i in range(len(model_energyGrid)) if detector_beam_Energies.min() <= model_energyGrid[ i] <= detector_beam_Energies.max()])
        model_beam_omniDiffFlux = model_omniDiffFlux[detector_to_model_beamIndicies]  # limit model beam to only the values in the raw data
        model_beam_energies = model_energyGrid[detector_to_model_beamIndicies]

        # KEY STEP: Integrate the model_beam_omniDiffFlux over E+dE to determine the "number of incoming electrons for that energy".
        # des
        model_varPhi_E = np.array([ simpson(x=model_beam_energies[i:i+1 + 1], y=model_beam_omniDiffFlux[i:i+1 + 1]) for i in range(len(model_beam_energies))])

        # calculate the secondary response
        degradedPrim_OmniDiff, secondaries_OmniDiff = backScatter_class().calcBackscatter(
                                                               energy_Grid=model_energyGrid,
                                                               beam_Energies=model_beam_energies,
                                                               beam_OmniDiffFlux=model_varPhi_E)

        # output all the data and
        # re-form the beam data onto the model_energy grid variable
        beam_Flux_output[tmeIdx] = np.array([model_omniDiffFlux[i] if detector_beam_Energies.min() <= model_energyGrid[ i] <= detector_beam_Energies.max() else 0 for i in range(len(model_energyGrid)) ])
        dPrim_Flux_output[tmeIdx] = degradedPrim_OmniDiff
        secondaries_Flux_output[tmeIdx] = secondaries_OmniDiff



    # output the data
    data_dict['Phi_dPrim'][0] =  dPrim_Flux_output
    data_dict['Phi_sec'][0] = secondaries_Flux_output
    data_dict['Phi_Beam'][0] = beam_Flux_output
    data_dict['Epoch'][0] = EpochFitData
    data_dict['energy_Grid'][0] = model_energyGrid


    # --- --- --- --- --- ---
    # --- OUTPUT THE DATA ---
    # --- --- --- --- --- ---
    # Construct the Data Dict
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': None}

    # update the data dict attrs
    for key, val in data_dict.items():

        # convert the data to numpy arrays
        data_dict[key][0] = np.array(data_dict[key][0])

        # update the attributes
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict[key][1].items():
            newAttrs[subKey] = subVal

        data_dict[key][1] = newAttrs

    outputPath = rf'C:\Data\physicsModels\invertedV\backScatter\ionosphericResponse.cdf'
    stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict)