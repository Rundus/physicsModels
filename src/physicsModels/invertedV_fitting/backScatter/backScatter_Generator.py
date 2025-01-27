# --- backScatter_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take in the primary data fits and produce secondary/backscatter curves for each. Use the
# parameterized Evans 1964 curves to determine the curves.

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
                                                            'Differential_Number_Flux_stdDev'])

    data_dict_beamFits = stl.loadDictFromFile(inputFilePath=rf"{primaryBeamToggles.outputFolder}\primaryBeam_fitting_parameters.cdf")

    # --- prepare the output data_dict ---
    data_dict = {'jN_beam': [[], {'DEPEND_0': 'Epoch','DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'beam_energy_Grid', 'UNITS': 'cm^-2 s^-1 sr^-1 eV^-1', 'LABLAXIS': 'beamFlux'}],
                 'jN_dgdPrim': [[], {'DEPEND_0': 'Epoch','DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'energy_Grid', 'UNITS': 'cm^-2 s^-1sr^-1 eV^-1', 'LABLAXIS': 'degradedPriamryFlux'}],
                 'jN_sec':[[], {'DEPEND_0': 'Epoch','DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'energy_Grid', 'UNITS': 'cm^-2 s^-1 sr^-1eV^-1', 'LABLAXIS': 'secondaryFlux'}],
                 'energy_Grid': [[], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                 'beam_energy_Grid': [[], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                 'Epoch':[[], {'DEPEND_0': None, 'UNITS': 'ns', 'LABLAXIS': 'Epoch'}],
                 'Pitch_Angle': [deepcopy(data_dict_diffFlux['Pitch_Angle'][0]), {'DEPEND_0': None, 'UNITS': 'Deg', 'LABLAXIS': 'Pitch_Angle'}],
                 }

    # Re-construct the 5-Averaged data. Only the Epoch dimension is reduced from original data
    EpochFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                                targetTimes=primaryBeamToggles.targetTimes,
                                                                                N_avg=primaryBeamToggles.numToAverageOver)

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
    beamFlux_output = np.zeros(shape=(len(EpochFitData),  len(data_dict_diffFlux['Pitch_Angle'][0]), len(model_energyGrid)))
    secondariesFlux_output = np.zeros(shape=(len(EpochFitData),  len(data_dict_diffFlux['Pitch_Angle'][0]), len(model_energyGrid)))
    dgdPrimFlux_output = np.zeros(shape=(len(EpochFitData),  len(data_dict_diffFlux['Pitch_Angle'][0]), len(model_energyGrid)))


    for tmeIdx in tqdm(range(len(EpochFitData))):

        # --- Get the Model Beam Parameters ---
        # Description: Use the model to generate a "Beam". Limit the model beam by the energies of the actual raw data.
        modelParamsIdx = np.abs(data_dict_diffFlux['Pitch_Angle'][0] - backScatterToggles.modelParametersPitchAngle).argmin()
        V0_fitParam = data_dict_beamFits['V0'][0][tmeIdx][modelParamsIdx]
        n0_fitParam = data_dict_beamFits['n'][0][tmeIdx][modelParamsIdx]
        Te_fitParam = data_dict_beamFits['Te'][0][tmeIdx][modelParamsIdx]

        # get the energy grid of the beam
        beam_EnergyGrid = model_energyGrid+V0_fitParam

        # construct jN for the beam
        if primaryBeamToggles.wDistributionToFit == 'Kappa':
            kappa_fitParam = data_dict_beamFits['kappa'][0][tmeIdx][modelParamsIdx]
            params = [n0_fitParam, Te_fitParam, V0_fitParam, kappa_fitParam]
            model_beam_diffNFlux = primaryBeam_class().diffNFlux_fitFunc_Kappa(beam_EnergyGrid, *params) # Assume the beam is isotropic
        elif primaryBeamToggles.wDistributionToFit == 'Maxwellian':
            params = [n0_fitParam, Te_fitParam, V0_fitParam]
            model_beam_diffNFlux = primaryBeam_class().diffNFlux_fitFunc_Maxwellian(beam_EnergyGrid, *params)

        ############################
        # --- CALC IONO RESPONSE ---
        ############################

        for loopIdx, PitchValue in enumerate(data_dict_diffFlux['Pitch_Angle'][0]):
            if PitchValue in [-10,100,110,120,130,140,150,160,170,180,190]:
                dgdPrimaries, secondaries, beam_jN_targetPitch = np.zeros(shape=(len(model_energyGrid))),np.zeros(shape=(len(model_energyGrid))),np.zeros(shape=(len(model_energyGrid)))
            else:
                dgdPrimaries, secondaries, beam_jN_targetPitch = backScatter_class().calcIonosphericResponse(
                    beta=backScatterToggles.betaChoice,
                    V0=V0_fitParam,
                    targetPitch=PitchValue,
                    response_energy_Grid=model_energyGrid,
                    beam_EnergyGrid=beam_EnergyGrid,
                    beam_diffNFlux=model_beam_diffNFlux,
                )

            # --- output all the data ---
            beamFlux_output[tmeIdx][loopIdx] = beam_jN_targetPitch
            dgdPrimFlux_output[tmeIdx][loopIdx] = dgdPrimaries
            secondariesFlux_output[tmeIdx][loopIdx] = secondaries

    # output the data
    data_dict['jN_beam'][0] = beamFlux_output
    data_dict['jN_dgdPrim'][0] = dgdPrimFlux_output
    data_dict['jN_sec'][0] = secondariesFlux_output
    data_dict['energy_Grid'][0] = model_energyGrid
    data_dict['beam_energy_Grid'][0] = beam_EnergyGrid
    data_dict['Epoch'][0] = EpochFitData

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

    outputPath = rf'{backScatterToggles.outputFolder}\backScatter.cdf'
    stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict)