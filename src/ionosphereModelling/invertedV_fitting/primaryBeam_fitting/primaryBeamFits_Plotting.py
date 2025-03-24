# --- primaryBeamFits_Plotting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: get the data from the primary beam fits and plot the data WITHOUT regenerating all the fits again
# TODO: Add restrictions on the ChiSquare and exlude bad fits e.g. too few datapoints, poor ChiSquare values

from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
import spaceToolsLib as stl
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, shutil
import datetime as dt
from glob import glob

##################
# --- PLOTTING ---
##################

Title_FontSize = 20
Label_FontSize = 25
Label_Padding = 8
Text_FontSize = 20

Tick_FontSize = 22
Tick_Length = 3
Tick_Width = 1.5
Tick_FontSize_minor = 15
Tick_Length_minor = 1
Tick_Width_minor = 1
Plot_LineWidth = 0.5
plot_MarkerSize = 12
Legend_fontSize = 14
dpi = 200

# --- Cbar ---
my_cmap = stl.apl_rainbow_black0_cmap()
my_cmap.set_bad(color=(0, 0, 0))
cbarMin, cbarMax = 1E5, 1E7
cbarTickLabelSize = 14
cbar_Fontsize = 20


def generatePrimaryBeamFitPlots(GenToggles, primaryBeamToggles, **kwargs):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    paramFiles = glob(f'{primaryBeamToggles.outputFolder}\*primaryBeam_fitting_parameters.cdf*')
    data_dict_params = stl.loadDictFromFile(inputFilePath=paramFiles[0])
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=primaryBeamToggles.inputDataPath)

    def plotIndividualFits(data_dict, data_dict_diffFlux):

        # Individualized Toggles
        figure_width = 10  # in inches
        figure_height = 10  # in inches

        # get the raw data, fitted data and fit parameters, then evaluate the model and plot it on the data with the fit parameters
        for ploopIdx, pitchAngle in enumerate(primaryBeamToggles.wPitchsToFit):

            ptchIdx = np.abs(data_dict_diffFlux['Pitch_Angle'][0] - pitchAngle).argmin()

            # delete all images in the directory
            imagefolder = rf'{primaryBeamToggles.outputFolder}\fitPhotos\{pitchAngle}deg'
            for filename in os.listdir(imagefolder):
                file_path = os.path.join(imagefolder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            # get the groupAveraged Data
            Epoch_groupAverage, fitData_groupAverage, stdDev_groupAverage = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                                                            targetTimes= primaryBeamToggles.targetTimes,
                                                                                                        N_avg=primaryBeamToggles.numToAverageOver)
            # create new images
            for tmeIdx, tmeStamp in tqdm(enumerate(Epoch_groupAverage)):

                # get the full data at the time/pitch slice
                xData_raw = data_dict_diffFlux['Energy'][0]
                yData_raw = fitData_groupAverage[tmeIdx][ptchIdx] # this does NOT depend upon pitch angle

                # get the indicies and data of the datapoints that was fitted
                dataIdx_set = data_dict['dataIdxs'][0][tmeIdx][ptchIdx]
                fittedX = deepcopy(xData_raw[np.where(dataIdx_set > 0)[0]])
                fittedY = deepcopy(yData_raw[np.where(dataIdx_set > 0)[0]])

                # get the model parameters for this time/pitch slice
                n_model  = data_dict['n'][0][tmeIdx][ptchIdx]
                T_model  = data_dict['Te'][0][tmeIdx][ptchIdx]
                V0_model = data_dict['V0'][0][tmeIdx][ptchIdx]
                kappa_model = data_dict['kappa'][0][tmeIdx][ptchIdx]
                ChiSquare_model = data_dict['ChiSquare'][0][tmeIdx][ptchIdx]

                # generate the evaluated model data for display
                xData_model = np.linspace(fittedX.min(), fittedX.max(), 100)
                if primaryBeamToggles.wDistributionToFit == 'Maxwellian':
                    yData_model = primaryBeam_class().diffNFlux_fitFunc_Maxwellian(xData_model, n_model,T_model,V0_model)
                elif primaryBeamToggles.wDistributionToFit == 'Kappa':
                    yData_model = primaryBeam_class().diffNFlux_fitFunc_Kappa(xData_model, n_model, T_model, V0_model, kappa_model)


                # Generate the noise level
                yData_noise = helperFuncs().generateNoiseLevel(xData_raw, countNoiseLevel=primaryBeamToggles.countNoiseLevel)

                # --- MAKE THE PLOTS ---
                fig, ax = plt.subplots()
                fig.set_size_inches(figure_width, figure_height)
                fig.suptitle(f'Pitch Angle = {pitchAngle}$^\circ$ \n {tmeStamp} UTC' + f'\n {primaryBeamToggles.numToAverageOver}-Averaged Points', fontsize=Title_FontSize)


                # Raw Data
                ax.plot(xData_raw, yData_raw, '-o',color='tab:blue')

                # fitted Data
                ax.plot(fittedX, fittedY, '-o', color='tab:red')


                # plot the noise
                ax.plot(xData_raw,yData_noise, color='black',label=f'{primaryBeamToggles.countNoiseLevel}-Count Noise')

                # Model Fit
                ax.plot(xData_model, yData_model, color='red',
                        label=f'n = {round(n_model, 1)}' + ' cm$^{-3}$' +
                              f'\n T = {round(T_model, 1)} eV\n' +
                              f'V = {round(V0_model, 1)} eV\n' +
                              f'$\kappa$ = {round(kappa_model, 1)}\n' +
                              r'$\chi^{2}_{\nu}= $' +
                              f'{round(ChiSquare_model, 3)}')

                # vline of the maximum  point
                ax.axvline(fittedX[np.argmax(fittedY)], color='red')

                # beautify th eplot
                ax.tick_params(axis='y', which='major', colors='black', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
                ax.tick_params(axis='y', which='minor', colors='black', labelsize=Tick_FontSize_minor, length=Tick_Length_minor, width=Tick_Width_minor)
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_xlabel('Energy [eV]', fontsize=Label_FontSize)
                ax.set_ylabel('[cm$^{-2}$s$^{-1}$str$^{-1}$ eV/eV]', fontsize=Label_FontSize)
                ax.set_xlim(20, 1E4)
                ax.set_ylim(5E3, 1E7)
                ax.grid(True, alpha=0.4, which='both')
                # ax.text(250, 5E4, s='Primaries', fontsize=Text_FontSize, weight='bold', va='center', ha='center')
                # ax.text(70, 5E4, s='Secondaries/Backscatter', fontsize=Text_FontSize, weight='bold', va='center', ha='center')
                ax.legend(fontsize=Legend_fontSize)

                plt.savefig(rf'{primaryBeamToggles.outputFolder}\fitPhotos\{pitchAngle}deg\FitData_{pitchAngle}deg_{tmeIdx}.png')
                plt.close()

    def plotFitParameters(data_dict, data_dict_diffFlux):

        # Individualized Toggles
        figure_width = 14  # in inches
        figure_height = 18  # in inches

        # find the indicies of the plotted region
        lowIdx,highIdx = np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][0] ).argmin(),np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][1] ).argmin()


        for pLoopIdx, ptchVal in enumerate(primaryBeamToggles.wPitchsToFit):

            ptchIdx = np.abs(data_dict_diffFlux['Pitch_Angle'][0] - ptchVal).argmin()

            # get the fit parameters
            ne = data_dict['n'][0][:,ptchIdx]
            Te = data_dict['Te'][0][:,ptchIdx]
            V0 = data_dict['V0'][0][:,ptchIdx]
            kappa = data_dict['kappa'][0][:,ptchIdx]
            ChiSquare = data_dict['ChiSquare'][0][:,ptchIdx]
            timestamp = data_dict['timestamp_fitData'][0]

            # Apply restrictions/Filters to the data i.e. don't include bad fits

            # -- make the plot ---
            fig, ax = plt.subplots(nrows=6,sharex=True)
            fig.set_size_inches(figure_width, figure_height)

            # Make the title
            fig.suptitle(rf'ACES-II ($\alpha = {data_dict_diffFlux["Pitch_Angle"][0][ptchIdx]}$)' + '$^{\circ}$\n' +
                         f'{timestamp[0].strftime("%H:%M:%S")} to {timestamp[-1].strftime("%H:%M:%S")} UTC', fontsize=Title_FontSize, weight='bold')

            # raw data spectrogram
            cmap = ax[0].pcolormesh(data_dict_diffFlux['Epoch'][0][lowIdx:highIdx+1], data_dict_diffFlux['Energy'][0], data_dict_diffFlux['Differential_Number_Flux'][0][lowIdx:highIdx+1,ptchIdx,:].T, cmap=my_cmap, vmin=cbarMin, vmax=cbarMax, norm='log') # plot spectrogram

            ax[0].set_ylabel(rf'Energy [eV]', fontsize=Label_FontSize, labelpad=Label_Padding)
            ax[0].set_yscale('log')
            ax[0].set_ylim(28.22, 2500)

            # ChiSquare Values
            ax[1].scatter(timestamp,ChiSquare,color='black')
            ax[1].set_ylabel(r'$\chi^{2}_{\nu}$', fontsize=Label_FontSize)
            ax[1].set_ylim(1E-1,1E2)
            ax[1].set_yscale('log')
            ax[1].axhline(y=1, color='red')

            # Density Values
            ax[2].scatter(timestamp, ne,color='black')
            ax[2].set_ylim(0, 4)
            ax[2].set_ylabel(r'n$_{0}$ [cm$^{-3}$]', fontsize=Label_FontSize)

            # V0 Values
            ax[3].scatter(timestamp, V0, color='black')
            ax[3].set_ylim(50, 800)
            ax[3].set_ylabel(r'V$_{0}$ [eV]', fontsize=Label_FontSize)

            # Te Values
            ax[4].scatter(timestamp, Te,color='black')
            ax[4].set_ylim(20, 300)
            ax[4].set_ylabel(r'T$_{e}$ [eV]', fontsize=Label_FontSize)

            # Kappa Values
            ax[5].scatter(timestamp, kappa, color='black')
            ax[5].set_ylim(1E-1, 1E2)
            ax[5].set_yscale('log')
            ax[5].axhline(y=10)
            ax[5].set_ylabel(r'$\kappa$', fontsize=Label_FontSize)

            # Beautify the Plot
            fig.align_ylabels(ax[:])
            for i in range(0,6):
                ax[i].minorticks_on()
                ax[i].tick_params(axis='y', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
                ax[i].tick_params(axis='x', labelsize=Tick_FontSize - 5, length=Tick_Length, width=Tick_Width)
                if i != 0:
                    ax[i].grid(True, which='Major', alpha=0.5)

            # colorbar
            cax = fig.add_axes([0.915, 0.79, 0.022, 0.14])
            cbar = plt.colorbar(cmap, cax=cax)
            cbar.ax.minorticks_on()
            cbar.ax.tick_params(labelsize=cbar_Fontsize)
            cbar.set_label('[cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$]', fontsize=cbar_Fontsize, weight='bold')

            # set the x-ticks
            xtickTimes = [
                dt.datetime(2022, 11, 20, 17, 25, 25, 000),
                dt.datetime(2022, 11, 20, 17, 25, 39, 000),
                dt.datetime(2022, 11, 20, 17, 25, 53, 000),
                dt.datetime(2022, 11, 20, 17, 26, 8, 000)
            ]
            xtick_indicies = np.array([np.abs(data_dict_diffFlux['Epoch'][0] - tick).argmin() for tick in xtickTimes])
            ILat_ticks = [str(round(tick, 2)) for tick in data_dict_diffFlux['ILat'][0][xtick_indicies]]
            Alt_ticks = [str(round(tick / 1000, 1)) for tick in data_dict_diffFlux['Alt'][0][xtick_indicies]]
            time_ticks = [tick.strftime("%M:%S.") + str(round(tick.microsecond / 1000)) for tick in xtickTimes]
            tickLabels = [f'{time_ticks[k]}\n{Alt_ticks[k]}\n{ILat_ticks[k]}' for k in range(len(xtick_indicies))]
            ax[5].set_xticks(xtickTimes)
            ax[5].set_xticklabels(tickLabels)
            ax[5].set_xlabel('time [UTC]\nAlt [km]\nILat [deg]', fontsize=14, weight='bold')
            ax[5].xaxis.set_label_coords(-0.07, -0.09)

            fig.subplots_adjust(left=0.12, bottom=0.07, right=0.91, top=0.93, wspace=0.04,hspace=0.1)  # remove the space between plots

            plt.savefig(rf'{primaryBeamToggles.outputFolder}\FitParameters_{data_dict_diffFlux["Pitch_Angle"][0][ptchIdx]}deg.png')
            plt.close()

    ######################
    # --- OUTPUT PLOTS ---
    ######################
    if kwargs.get('individualPlots', False):
        plotIndividualFits(data_dict_params, data_dict_diffFlux)

    if kwargs.get('parameterPlots', False):
        plotFitParameters(data_dict_params, data_dict_diffFlux)