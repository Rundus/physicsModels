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

    # Re-construct the 5-Averaged data. Only the Epoch dimension is reduced from original data
    relevantPitchs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90deg
    EpochFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                                GenToggles=GenToggles,
                                                                                primaryBeamToggles=primaryBeamToggles)


    #####################
    # --- ENERGY GRID ---
    #####################
    modelEnergyGrid = np.linspace(10, 1E4, secondaryBackScatterToggles.N_energyGrid)

    ######################################
    # ------------------------------------
    # --- CALCULATE IONOSPHERIC RESPONSE ---
    # ------------------------------------
    ######################################
    beam_Flux = np.zeros(shape=(len(EpochFitData),len(modelEnergyGrid)))
    secondaries_Flux = np.zeros(shape=(len(EpochFitData),len(modelEnergyGrid)))
    backScatter_Flux = np.zeros(shape=(len(EpochFitData), len(modelEnergyGrid)))

    for tmeIdx in range(len(EpochFitData)):

        # --- Get the Model Beam Parameters ---
        # Description: Use the model to generate a "Beam". THIS IS NOT USED TO CALCULATE ACTUAL BACKSCATTER, but is
        # applied to give a complete J_N Flux for the energyGrid i.e. the output
        V0_fitParam = data_dict_beamFits['V0'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]
        n0_fitParam = data_dict_beamFits['n'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]
        Te_fitParam = data_dict_beamFits['Te'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]

        # Form the beam diffNFlux
        if primaryBeamToggles.wDistributionToFit == 'Kappa':
            kappa_fitParam = data_dict_beamFits['kappa'][0][secondaryBackScatterToggles.wPtchIdx][tmeIdx]
            params = [n0_fitParam, Te_fitParam, V0_fitParam, kappa_fitParam]
            diffNFlux_beam = primaryBeam_class().diffNFlux_fitFunc_Kappa(modelEnergyGrid, *params)
            diffNFlux_beam[np.where(modelEnergyGrid<V0_fitParam)[0]] = 0 # limit beam to only above V0
            beam_Flux[tmeIdx] += diffNFlux_beam

        elif primaryBeamToggles.wDistributionToFit == 'Maxwellian':
            params = [n0_fitParam, Te_fitParam, V0_fitParam]
            diffNFlux_beam = primaryBeam_class().diffNFlux_fitFunc_Maxwellian(modelEnergyGrid, *params)
            diffNFlux_beam[np.where(modelEnergyGrid < V0_fitParam)[0]] = 0  # limit beam to only above V0
            beam_Flux[tmeIdx] += diffNFlux_beam

        # --- Get the REAL Beam ---
        detectorEnergies = data_dict_diffFlux['Energy'][0]
        beamEngyIdx = np.abs(detectorEnergies-V0_fitParam).argmin()
        beam_Energies = detectorEnergies[:beamEngyIdx+1]
        diffNFlux_beam = diffNFlux_avg[tmeIdx, :, :beamEngyIdx+1]

        # calculate the OmniFlux - varPhi(E)
        omniDiffFlux = backScatter_class().calcOmni_diffNFlux(diffNFlux=diffNFlux_beam[:][relevantPitchs][:],
                                                    pitchValues=data_dict_diffFlux['Pitch_Angle'][0][relevantPitchs],
                                                    energyValues=beam_Energies) # -1 is added since I do High-to-low energy

        degradedPrim_OmniDiff, secondaries_OmniDiff = backScatter_class().calcBackscatter(energy_Grid=modelEnergyGrid,
                                                               beam_Energies=beam_Energies,
                                                               beam_OmniDiffFlux=omniDiffFlux,
                                                               V0=V0_fitParam)









    #
    # # --- --- --- --- --- ---
    # # --- OUTPUT THE DATA ---
    # # --- --- --- --- --- ---
    # # Construct the Data Dict
    # exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
    #               'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
    #               'SCALETYP': 'linear', 'LABLAXIS': None}
    #
    # # update the data dict attrs
    # for key, val in data_dict.items():
    #
    #     # convert the data to numpy arrays
    #     data_dict[key][0] = np.array(data_dict[key][0])
    #
    #     # update the attributes
    #     newAttrs = deepcopy(exampleVar)
    #
    #     for subKey, subVal in data_dict[key][1].items():
    #         newAttrs[subKey] = subVal
    #
    #     data_dict[key][1] = newAttrs
    #
    # outputPath = rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\primaryBeam_fitting_parameters.cdf'
    # stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict)