# --- backScatter_Plotting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: get the data from the backscatter Curves and plot the data WITHOUT regenerating all the curves again

from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import *
import spaceToolsLib as stl
import matplotlib.pyplot as plt
import os, shutil

##################
# --- PLOTTING ---
##################
Figure_width = 8 # in inches
Figure_height = 6  # in inches
Title_FontSize = 20
Label_FontSize = 20
Label_Padding = 8
Text_FontSize = 20

Tick_FontSize = 22
Tick_Length = 3
Tick_Width = 1.5
Tick_FontSize_minor = 15
Tick_Length_minor = 1
Tick_Width_minor = 1
Plot_LineWidth = 1.5
plot_MarkerSize = 12
Legend_FontSize = 14
dpi = 200

def generateBackScatterPlots(GenToggles,backScatterToggles,primaryBeamToggles, **kwargs):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=GenToggles.input_diffNFiles[GenToggles.wFlyerFit])
    data_dict = stl.loadDictFromFile(inputFilePath=rf'{backScatterToggles.outputFolder}\backScatter.cdf')
    data_dict_fitParams = stl.loadDictFromFile(inputFilePath=rf'{primaryBeamToggles.outputFolder}\primaryBeam_fitting_parameters.cdf')
    EpochFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                             targetTimes=primaryBeamToggles.targetTimes,
                                                                             N_avg=primaryBeamToggles.numToAverageOver)
    def plotIndividualBackScatters(data_dict):

        for loopIdx, PitchVal in enumerate(primaryBeamToggles.wPitchsToFit):

            ptchIdx = np.abs(data_dict_diffFlux['Pitch_Angle'][0] - PitchVal).argmin()

            # delete all images in the directory
            imagefolder = rf'{backScatterToggles.outputFolder}\fitPhotos\{data_dict_diffFlux["Pitch_Angle"][0][ptchIdx]}deg'
            for filename in os.listdir(imagefolder):
                file_path = os.path.join(imagefolder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            for tmeIdx in range(len(EpochFitData)):

                dgdPrim_Flux = data_dict['jN_dgdPrim'][0][tmeIdx][ptchIdx]
                secondaries_Flux = data_dict['jN_sec'][0][tmeIdx][ptchIdx]
                beam_Flux = data_dict['jN_beam'][0][tmeIdx][ptchIdx]

                if [dgdPrim_Flux, secondaries_Flux, beam_Flux] != [np.zeros(shape=(len(dgdPrim_Flux))),np.zeros(shape=(len(secondaries_Flux))),np.zeros(shape=(len(beam_Flux)))]:
                    fig, ax = plt.subplots()
                    fig.set_figwidth(Figure_width)
                    fig.set_figheight(Figure_height)
                    ax.set_title(rf'$\alpha = {data_dict_diffFlux["Pitch_Angle"][0][ptchIdx]}' + '^{\circ}$'+f'\n{EpochFitData[tmeIdx]} UTC', fontsize=Title_FontSize)

                    # plot the raw data
                    ax.plot(data_dict_diffFlux['Energy'][0], diffNFlux_avg[tmeIdx][ptchIdx], 'o-', color='black')

                    # plot the data which was fitted
                    dataIdxs = data_dict_fitParams['dataIdxs'][0][tmeIdx][ptchIdx]
                    fittedIndicies = np.where(dataIdxs > 0)[0]
                    ax.plot(data_dict_diffFlux['Energy'][0][fittedIndicies], diffNFlux_avg[tmeIdx, ptchIdx, fittedIndicies], 'o-', color='tab:red')

                    # plot the noise level
                    ax.plot(data_dict['energy_Grid'][0], helperFuncs().generateNoiseLevel(energyData=data_dict['energy_Grid'][0],countNoiseLevel=primaryBeamToggles.countNoiseLevel),'--',color='black',label=f'{primaryBeamToggles.countNoiseLevel}-Count Noise')

                    # plot the model beam reconstructed
                    ax.plot(data_dict['beam_energy_Grid'][0][tmeIdx], beam_Flux , color='tab:red', label='Beam', linewidth=Plot_LineWidth)

                    # plot the degraded primaries
                    ax.plot(data_dict['energy_Grid'][0], dgdPrim_Flux, color='tab:red',label='Upward Degraded Prim.', linewidth=Plot_LineWidth)

                    # plot the secondary flux
                    ax.plot(data_dict['energy_Grid'][0], secondaries_Flux, color='tab:green', label='Upward Secondaries', linewidth=Plot_LineWidth)

                    # plot the total response
                    ax.plot(data_dict['energy_Grid'][0], dgdPrim_Flux + secondaries_Flux, color='tab:blue', label='Total Response', linewidth=Plot_LineWidth)

                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_ylabel('[cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$]', fontsize=Label_FontSize)
                    ax.set_xlim(20, 1E4)
                    ax.set_ylim(5E3, 1E7)
                    ax.set_xlabel('Energy [eV]', fontsize=Label_FontSize)
                    ax.legend(fontsize=Legend_FontSize)
                    ax.grid(alpha=0.5)
                    ax.tick_params(axis='y', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
                    ax.tick_params(axis='x', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
                    plt.savefig(rf'{backScatterToggles.outputFolder}\fitPhotos\{PitchVal}deg\backScatter_{PitchVal}deg_{tmeIdx}.png')


    ######################
    # --- OUTPUT PLOTS ---
    ######################
    if kwargs.get('individualPlots', False):
        plotIndividualBackScatters(data_dict)