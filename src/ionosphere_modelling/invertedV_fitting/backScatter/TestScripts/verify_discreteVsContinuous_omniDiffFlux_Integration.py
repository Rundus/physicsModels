# --- verify_discreteVsContinuous_omniDiffFlux_Integration.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Verify that I get the ~same number of electrons from the integration of the
# beam regardless of whether I use real data or model data. Won't be exact but should be close.


#################
# --- IMPORTS ---
#################
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import GenToggles
from src.physicsModels.invertedV_fitting.backScatter.backScatter_classes import *
from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
import matplotlib.pyplot as plt
import numpy as np

#################
# --- TOGGLES ---
#################

energy_Grid = np.linspace(1E1, 1E4, 2000)

plotBeams = False
plotIntegrationDifferneces = False
Figure_width = 8 # in inches
Figure_height = 6  # in inches
Plot_LineWidth = 2.5
Text_FontSize = 20
Label_FontSize = 14.5
Label_Padding = 5
Tick_FontSize = 11.5
Tick_Length = 5
Tick_Width = 2
Tick_Padding = 10
Legend_FontSize = 13

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

relevantPitchs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 0, 10, 20, 30, 40, 50, 60, 70, 80deg
EpochFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                            GenToggles=GenToggles,
                                                                            N_avg=5)


Phi_model_output = np.zeros(len(EpochFitData))
Phi_raw_output = np.zeros(len(EpochFitData))


for timeIdx in range(len(EpochFitData)):
    # Form the beam on the data grid - 10deg Pitch Angle at specified time index
    V0_fitParam = data_dict_beamFits['V0'][0][0][timeIdx]
    n0_fitParam = data_dict_beamFits['n'][0][0][timeIdx]
    Te_fitParam = data_dict_beamFits['Te'][0][0][timeIdx]
    kappa_fitParam = data_dict_beamFits['kappa'][0][2][timeIdx]

    # --- RAW BEAM DATA ---
    # get the indicies and data of the datapoints that was fitted
    Energies_detector = data_dict_diffFlux['Energy'][0]
    diffNFlux_detector = diffNFlux_avg[timeIdx, 2, :]
    dataIdx_set = data_dict_beamFits['dataIdxs'][0][0][timeIdx]
    beam_data_Energies = Energies_detector[np.where(dataIdx_set > 0)[0]]
    beam_data_diffNFlux = diffNFlux_detector[np.where(dataIdx_set > 0)[0]]

    # --- MODEL FIT DATA ---
    params = [n0_fitParam, Te_fitParam, V0_fitParam, kappa_fitParam]
    diffNFlux_model = primaryBeam_class().diffNFlux_fitFunc_Kappa(energy_Grid, *params)
    dataIndicies = [i for i in range(len(energy_Grid)) if beam_data_Energies.min() <= energy_Grid[i] <= beam_data_Energies.max()]
    beam_model_diffNFlux = diffNFlux_model[dataIndicies]  # limit beam to below the maximum real datapoint and above the lowest real datapoint
    beam_model_Energies = energy_Grid[dataIndicies]


    #################################
    # --- PERFORM THE INTEGRATION ---
    #################################
    Phi_raw = -1*simpson(x=beam_data_Energies, y=beam_data_diffNFlux)
    Phi_model = simpson(x=beam_model_Energies, y=beam_model_diffNFlux)
    Phi_model_output[timeIdx] = Phi_model
    Phi_raw_output[timeIdx] = Phi_raw
    print('{:.3e}'.format(Phi_raw))
    print('{:.3e}'.format(Phi_model))
    print('Diff', '{:.3e}'.format(Phi_raw-Phi_model))
    print('%Diff', 100*(Phi_raw-Phi_model)/Phi_model)
    print('\n')

    if timeIdx in [103]:
        if plotBeams:
            # plot the two beams to ensure they're about equal
            fig, ax = plt.subplots()
            ax.set_title(f'{EpochFitData[timeIdx]} UTC')
            ax.plot(beam_data_Energies, beam_data_diffNFlux, '-o', color='tab:red',label='RawData', linewidth=Plot_LineWidth)
            ax.plot(beam_model_Energies,beam_model_diffNFlux,color='tab:blue',label='FitData', linewidth=Plot_LineWidth)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('[cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$]', fontsize=Label_FontSize)
            ax.set_ylim(1E4, 5E7)
            ax.set_xlim(20, 3E3)
            ax.set_xlabel('Energy [eV]', fontsize=Label_FontSize)
            ax.legend(fontsize=Legend_FontSize)
            ax.grid(alpha=0.5)
            ax.tick_params(axis='y', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
            ax.tick_params(axis='x', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
            plt.show()



if plotIntegrationDifferneces:
    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(range(len(EpochFitData)), Phi_raw_output)
    ax[0].set_ylabel('Raw Data Phi')
    ax[1].plot(range(len(EpochFitData)), Phi_model_output)
    ax[1].set_ylabel('Model Phi')
    ax[2].plot(range(len(EpochFitData)), 100*(Phi_raw_output-Phi_model_output)/Phi_model_output)
    ax[2].set_ylabel('% Difference')
    plt.show()