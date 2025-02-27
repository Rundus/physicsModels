# --- backScatter_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take in the primary data fits and produce secondary/backscatter curves for each. Use the
# parameterized Evans 1964 curves to determine the curves.
import matplotlib.pyplot as plt


# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from src.physicsModels.invertedV_fitting.backScatter.backScatter_classes import *
from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
import spaceToolsLib as stl
from time import time
from copy import deepcopy
from tqdm import tqdm
start_time = time()
# --- --- --- --- ---

def generateSecondaryBackScatter(GenToggles, primaryBeamToggles, backScatterToggles, **kwargs):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=GenToggles.input_diffNFiles[GenToggles.wFlyerFit],
                                              wKeys_Reduce=['Differential_Number_Flux',
                                                            'Epoch',
                                                            'Differential_Number_Flux_stdDev',
                                                            'ILat'])

    data_dict_beamFits = stl.loadDictFromFile(inputFilePath=rf"{primaryBeamToggles.outputFolder}\primaryBeam_fitting_parameters.cdf")

    # Re-construct the 5-Averaged data. Only the Epoch dimension is reduced from original data
    EpochFitData, ILatFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                                targetTimes=[data_dict_diffFlux['Epoch'][0][0], data_dict_diffFlux['Epoch'][0][-1]],
                                                                                N_avg=primaryBeamToggles.numToAverageOver)

    # --- prepare the output data_dict ---
    data_dict_output = {'jN_beam': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'beam_energy_Grid', 'UNITS': 'cm^-2 s^-1 sr^-1 eV^-1', 'LABLAXIS': 'beamFlux'}],
                        'num_flux_beam': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1': 'beam_energy_Grid', 'UNITS': 'cm^-2 s^-1', 'LABLAXIS': 'beam number flux'}],
                        'jN_dgdPrim': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'energy_Grid', 'UNITS': 'cm^-2 s^-1sr^-1 eV^-1', 'LABLAXIS': 'degradedPriamryFlux'}],
                        'num_flux_dgdPrim': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1': 'energy_Grid', 'UNITS': 'cm^-2 s^-1', 'LABLAXIS': 'degraded primaries number flux'}],
                        'jN_sec': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'energy_Grid', 'UNITS': 'cm^-2 s^-1 sr^-1eV^-1', 'LABLAXIS': 'secondaryFlux'}],
                        'num_flux_sec': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1': 'energy_Grid', 'UNITS': 'cm^-2 s^-1', 'LABLAXIS': 'secondaries number flux'}],
                        'energy_Grid': [[], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                        'beam_energy_Grid': [[], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                        'Epoch': [EpochFitData, deepcopy(data_dict_diffFlux['Epoch'][1])],
                        'ILat': [ILatFitData, deepcopy(data_dict_diffFlux['ILat'][1])],
                        'Pitch_Angle': deepcopy(data_dict_diffFlux['Pitch_Angle']),
                        }

    #####################
    # --- ENERGY GRID ---
    #####################
    model_energyGrid = backScatterToggles.model_energyGrid

    ########################################
    # --------------------------------------
    # --- CALCULATE IONOSPHERIC RESPONSE ---
    # --------------------------------------
    ########################################

    # prepare the output variables
    beam_jN_output = np.zeros(shape=(len(EpochFitData),  len(data_dict_diffFlux['Pitch_Angle'][0]), len(model_energyGrid)))
    beam_para_num_flux_output = np.zeros( shape=(len(EpochFitData), len(model_energyGrid)))

    sec_jN_output = np.zeros(shape=(len(EpochFitData),  len(data_dict_diffFlux['Pitch_Angle'][0]), len(model_energyGrid)))
    sec_para_num_flux_output = np.zeros(shape=(len(EpochFitData), len(model_energyGrid)))

    dgdPrim_jN_output = np.zeros(shape=(len(EpochFitData),  len(data_dict_diffFlux['Pitch_Angle'][0]), len(model_energyGrid)))
    dgdPrim_para_num_flux_output = np.zeros(shape=(len(EpochFitData), len(model_energyGrid)))

    beam_energy_grid_output = np.zeros(shape=(len(EpochFitData), len(model_energyGrid)))

    response_energy_grid_output = np.zeros(shape=(len(EpochFitData), len(model_energyGrid)))

    for tmeIdx in tqdm(range(len(EpochFitData))):
        # --- Get the Model Beam Parameters ---
        # Description: Use the model to generate a "Beam". Limit the model beam by the energies of the actual raw data.
        modelParamsIdx = np.abs(data_dict_diffFlux['Pitch_Angle'][0] - backScatterToggles.modelParametersPitchAngle).argmin()
        V0_fitParam = data_dict_beamFits['V0'][0][tmeIdx][modelParamsIdx]
        n0_fitParam = data_dict_beamFits['n'][0][tmeIdx][modelParamsIdx]
        Te_fitParam = data_dict_beamFits['Te'][0][tmeIdx][modelParamsIdx]

        beam_EnergyGrid = deepcopy(model_energyGrid) + V0_fitParam
        beam_energy_grid_output[tmeIdx] = deepcopy(beam_EnergyGrid)

        if [int(V0_fitParam), int(n0_fitParam), int(Te_fitParam)] != [0, 0, 0]:

            try: # if the data looks good to fit
                # get the energy grid of the beam

                # construct jN for the beam (Assume the beam is isotropic)
                if primaryBeamToggles.wDistributionToFit == 'Kappa':
                    kappa_fitParam = data_dict_beamFits['kappa'][0][tmeIdx][modelParamsIdx]
                    params = [n0_fitParam, Te_fitParam, V0_fitParam, kappa_fitParam]
                    model_beam_jN = primaryBeam_class().diffNFlux_fitFunc_Kappa(beam_EnergyGrid, *params)
                elif primaryBeamToggles.wDistributionToFit == 'Maxwellian':
                    params = [n0_fitParam, Te_fitParam, V0_fitParam]
                    model_beam_jN = primaryBeam_class().diffNFlux_fitFunc_Maxwellian(beam_EnergyGrid, *params)

                ############################
                # --- CALC IONO RESPONSE ---
                ############################
                para_num_flux_beam, para_num_flux_dgdPrim, para_num_flux_sec = backScatter_class().calcIonosphericResponse(
                            beta=backScatterToggles.betaChoice,
                            V0=V0_fitParam,
                            response_energy_grid=model_energyGrid,
                            beam_energy_grid=beam_EnergyGrid,
                            beam_jN=deepcopy(model_beam_jN)
                        )

                # store the parallel number flux data
                beam_para_num_flux_output[tmeIdx] = para_num_flux_beam
                sec_para_num_flux_output[tmeIdx] = para_num_flux_sec
                dgdPrim_para_num_flux_output[tmeIdx] = para_num_flux_dgdPrim

                # get the jN value for each pitch angle - used to compare the model to the real data
                for loopIdx, PitchValue in enumerate(data_dict_diffFlux['Pitch_Angle'][0]):
                    if PitchValue in np.setdiff1d(data_dict_diffFlux['Pitch_Angle'][0], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]): # calculates BackScatter only on wPitchToFit angles
                        dgdPrim_target_pitch, secondaries_target_pitch, beam_jN_target_pitch = np.zeros(shape=(len(model_energyGrid))), np.zeros(shape=(len(model_energyGrid))), np.zeros(shape=(len(model_energyGrid)))
                    else:
                        dgdPrim_target_pitch, secondaries_target_pitch, beam_jN_target_pitch = backScatter_class().calc_response_at_target_pitch(
                            beta = backScatterToggles.betaChoice,
                            V0 = V0_fitParam,
                            target_pitch = PitchValue,
                            energy_grid = model_energyGrid,
                            dgdPrim_num_flux = para_num_flux_dgdPrim,
                            sec_num_flux = para_num_flux_sec,
                            beam_energy_grid = beam_EnergyGrid,
                            beam_jN = deepcopy(model_beam_jN)
                        )

                    # --- output all the data ---
                    beam_jN_output[tmeIdx][loopIdx] = beam_jN_target_pitch
                    dgdPrim_jN_output[tmeIdx][loopIdx] = dgdPrim_target_pitch
                    sec_jN_output[tmeIdx][loopIdx] = secondaries_target_pitch # if
            except:

                # --- output the para_num_flux data ---
                beam_para_num_flux_output[tmeIdx] = np.zeros(shape=(len(model_energyGrid)))
                sec_para_num_flux_output[tmeIdx] = np.zeros(shape=(len(model_energyGrid)))
                dgdPrim_para_num_flux_output[tmeIdx] = np.zeros(shape=(len(model_energyGrid)))

                # --- output all the jN data ---
                beam_jN_output[tmeIdx][loopIdx] = np.zeros(shape=(len(model_energyGrid)))
                dgdPrim_jN_output[tmeIdx][loopIdx] = np.zeros(shape=(len(model_energyGrid)))
                sec_jN_output[tmeIdx][loopIdx] = np.zeros(shape=(len(model_energyGrid)))

    # output the all data
    data_dict_output['jN_beam'][0] = beam_jN_output
    data_dict_output['num_flux_beam'][0] = beam_para_num_flux_output

    data_dict_output['jN_dgdPrim'][0] = dgdPrim_jN_output
    data_dict_output['num_flux_dgdPrim'][0] = dgdPrim_para_num_flux_output

    data_dict_output['jN_sec'][0] = sec_jN_output
    data_dict_output['num_flux_sec'][0] = sec_para_num_flux_output

    data_dict_output['energy_Grid'][0] = model_energyGrid
    data_dict_output['beam_energy_Grid'][0] = beam_energy_grid_output

    # --- --- --- --- --- ---
    # --- OUTPUT THE DATA ---
    # --- --- --- --- --- ---
    # Construct the Data Dict
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': None}

    # update the data dict attrs
    for key, val in data_dict_output.items():

        # convert the data to numpy arrays
        data_dict_output[key][0] = np.array(data_dict_output[key][0])

        # update the attributes
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict_output[key][1].items():
            newAttrs[subKey] = subVal

        data_dict_output[key][1] = newAttrs

    outputPath = rf'{backScatterToggles.outputFolder}\backScatter.cdf'
    stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict_output)