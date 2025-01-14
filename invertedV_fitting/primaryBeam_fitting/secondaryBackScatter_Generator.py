# --- secondaryBackScatter_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take in the primary data fits and produce secondary/backscatter curves for each. Use the
# parameterized Evans 1964 curves to determine the curves.
import numpy as np

# TODO: note: the code ONLY uses the V0 value from the FIRST wFitPitchAngle value
# TODO: REALLY check the calcOmni_diffNFlux calculation to see if it's right. So too with calcOmni_diffNFlux
# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from invertedV_fitting.primaryBeam_fitting.model_primaryBeam_classes import *
from invertedV_fitting.primaryBeam_fitting.Evans_Model.parameterizationCurves_Evans1974_classes import *
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

    # Re-construct the 5-Averaged data over the pitch Angle Range relevant to Ionospheric backscatter
    relevantPitchs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 0, 10, 20, 30, 40, 50, 60, 70, 80deg
    # relevantPitchs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])  # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180deg
    EpochFitData, diffNFlux_avg, stdDev_avg = helperFitFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                          pitchIdxs=relevantPitchs,
                                                                          GenToggles=GenToggles,
                                                                          primaryBeamToggles=primaryBeamToggles)

    ######################################
    # ------------------------------------
    # --- CALCULATE SECONDARY RESPONSE ---
    # ------------------------------------
    ######################################
    secondaryFlux = np.zeros(shape=(len(EpochFitData),len(data_dict_diffFlux['Energy'][0])))

    for tmeIdx in range(len(EpochFitData)):

        # get the V0 value this time stamp
        V0_value = data_dict_beamFits['V0'][0][0][tmeIdx]
        engyIdx = np.abs(data_dict_diffFlux['Energy'][0] - V0_value).argmin()

        # get the reduced averaged data over the energy range. Pitch angle range is already accounted for.
        diffNFlux = diffNFlux_avg[tmeIdx, :, :engyIdx+1]

        # calculate the OmniFlux
        omniFlux = -1*helperFitFuncs().calcTotal_NFlux(diffNFlux=diffNFlux,
                                                    pitchValues=data_dict_diffFlux['Pitch_Angle'][0][relevantPitchs],
                                                    energyValues=data_dict_diffFlux['Energy'][0][:engyIdx+1]) # -1 is added since I do High-to-low energy

        # use OmniFlux to determine secondary response. Out is shape=(len(Energy))
        secondaryFlux[tmeIdx] = Evans1974().calcSecondaries(detectorEnergies=data_dict_diffFlux['Energy'][0],
                                                    InputOmniFlux=omniFlux,
                                                    Niterations=secondaryBackScatterToggles.Niterations_secondaries,
                                                    V0=V0_value)

        # # make a plot of the output
        # fig, ax = plt.subplots()
        # ax.set_title(f'{EpochFitData[tmeIdx]} UTC')
        # ax.plot(data_dict_diffFlux['Energy'][0], secondaryFlux)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_ylabel('# cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$')
        # ax.set_ylim(1E4, 5E7)
        # ax.set_xlim(1E1,1E4)
        # ax.set_xlabel('Energy [eV]')
        # plt.show()

    ########################################
    # --------------------------------------
    # --- CALCULATE BACKSCATTER RESPONSE ---
    # --------------------------------------
    ########################################
    backscatterFlux = np.zeros(shape=(len(EpochFitData),len(data_dict_diffFlux['Energy'][0])))
    secondaryFlux_backscatter = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Energy'][0])))

    for tmeIdx in range(len(EpochFitData)):

        # get the V0 value this time stamp
        V0_value = data_dict_beamFits['V0'][0][0][tmeIdx]
        engyIdx = np.abs(data_dict_diffFlux['Energy'][0] - V0_value).argmin()

        # get the reduced averaged data over the energy range. Pitch angle range is already accounted for.
        diffNFlux = diffNFlux_avg[tmeIdx, :, :engyIdx + 1]
        primaryBeam_energies = data_dict_diffFlux['Energy'][0][:engyIdx+1]

        # calculate the OmniFlux - Integrate over pitch angle
        omniDiffFlux = helperFitFuncs().calcOmni_diffNFlux(diffNFlux=diffNFlux,
                                                           pitchValues=data_dict_diffFlux['Pitch_Angle'][0][relevantPitchs],
                                                           energyValues=primaryBeam_energies)

        backscatterFlux[tmeIdx], secondaryFlux_backscatter[tmeIdx] = Evans1974().calcBackScatter(
                                                        IncidentBeamEnergies=primaryBeam_energies,
                                                        Incident_OmniDiffFlux=omniDiffFlux,
                                                        Niterations=secondaryBackScatterToggles.Niterations_backscatter,
                                                        V0=V0_value,
                                                        detectorEnergies = data_dict_diffFlux['Energy'][0])


    #######################################
    # --- Sum the Ionospheric Responses ---
    #######################################
    secondaries_total = secondaryFlux + secondaryFlux_backscatter
    ionoResponse_total = backscatterFlux + secondaries_total






    #########################
    # --- PLOT THE OUTPUT ---
    #########################

    # make a plot of the output
    for tmeIdx in range(len(EpochFitData)):
        fig, ax = plt.subplots()
        ax.set_title(f'{EpochFitData[tmeIdx]} UTC')
        ax.plot(data_dict_diffFlux['Energy'][0], backscatterFlux[tmeIdx], label='backscatter')
        ax.plot(data_dict_diffFlux['Energy'][0], secondaryFlux_backscatter[tmeIdx], label='secondaries (backscatter)')
        ax.plot(data_dict_diffFlux['Energy'][0], secondaries_total[tmeIdx], label='secondaries (total)')
        ax.plot(data_dict_diffFlux['Energy'][0], diffNFlux_avg[tmeIdx][0],'-o', color='black',label=r'$\alpha=10^{\circ}$')
        ax.plot(data_dict_diffFlux['Energy'][0], ionoResponse_total[tmeIdx], '-o', color='tab:red', label='Total Response')
        ax.plot(data_dict_diffFlux['Energy'][0], secondaryFlux[tmeIdx], '-o',color='magenta', label='Secondaries (Primary Beam)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('# cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$')
        ax.set_ylim(1E4, 1E9)
        ax.set_xlim(1E1,1E4)
        ax.set_xlabel('Energy [eV]')
        ax.legend()
        plt.show()



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