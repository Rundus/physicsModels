# --- primaryBeamFits_Plotting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: get the data from the primary beam fits and plot the data WITHOUT regenerating all the fits again


# TODO: Add a spectrogram of the real data above each fit?

from invertedV_fitting.primaryBeam_fitting.model_primaryBeam_classes import *
import spaceToolsLib as stl
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, shutil

##################
# --- PLOTTING ---
##################
figure_width = 10 # in inches
figure_height =8 # in inches
Title_FontSize = 20
Label_FontSize = 20
Label_Padding = 8
Text_FontSize = 18
Tick_FontSize = 16
Tick_Length = 1
Tick_Width = 1
Tick_FontSize_minor = 10
Tick_Length_minor = 1
Tick_Width_minor = 1
Plot_LineWidth = 0.5
plot_MarkerSize = 12
Legend_fontSize = 14
dpi = 200

# --- Cbar ---
mycmap = stl.apl_rainbow_black0_cmap()
cbarMin, cbarMax = 1E-18, 1E-14
cbarTickLabelSize = 14

def generatePrimaryBeamFitPlots(GenToggles, primaryBeamToggles, **kwargs):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=GenToggles.input_diffNFiles[GenToggles.wFlyerFit])

    data_dict = stl.loadDictFromFile(inputFilePath=rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\primaryBeam_fitting_parameters.cdf')

    # get the raw data, fitted data and fit parameters, then evaluate the model and plot it on the data with the fit parameters
    for ptchIdx, ptchVal in enumerate(primaryBeamToggles.wPitchsToFit):

        # delete all images in the directory
        imagefolder = rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\fitPhotos\{data_dict_diffFlux["Pitch_Angle"][0][ptchVal]}deg'
        for filename in os.listdir(imagefolder):
            file_path = os.path.join(imagefolder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


        # create new images
        for idx,tmeStamp in tqdm(enumerate(data_dict['timestamp_fitData'][0][ptchIdx])):

            # get the full data at the time/pitch slice
            tmeIdx = np.abs(data_dict_diffFlux['Epoch'][0] - tmeStamp).argmin() # get the index of where the data was taken from
            xData_raw = data_dict_diffFlux['Energy'][0]
            yData_raw = data_dict_diffFlux['Differential_Number_Flux'][0][tmeIdx][ptchVal]

            # get the indicies and data of the datapoints that was fitted
            dataIdx_set = data_dict['dataIdxs'][0][ptchIdx][idx]
            fittedX = deepcopy(xData_raw[np.where(dataIdx_set>0)[0]])
            fittedY = deepcopy(yData_raw[np.where(dataIdx_set > 0)[0]])

            # get the model parameters for this time/pitch slice
            n_model  = data_dict['n'][0][ptchIdx][idx]
            T_model  = data_dict['Te'][0][ptchIdx][idx]
            V0_model = data_dict['V0'][0][ptchIdx][idx]
            kappa_model = data_dict['kappa'][0][ptchIdx][idx]
            ChiSquare_model = data_dict['ChiSquare'][0][ptchIdx][idx]

            # generate the evaluated model data for display
            xData_model = np.linspace(fittedX.min(), fittedX.max(), 100)
            if primaryBeamToggles.wDistributionToFit == 'Maxwellian':
                yData_model = fittingDistributions().diffNFlux_fitFunc_Maxwellian(xData_model, n_model,T_model,V0_model, stl.m_e, stl.q0 )
            elif primaryBeamToggles.wDistributionToFit == 'Kappa':
                yData_model = fittingDistributions().diffNFlux_fitFunc_Kappa(xData_model, n_model, T_model, V0_model, kappa_model, stl.m_e, stl.q0)


            # Generate the noise level
            yData_noise = helperFitFuncs().generateNoiseLevel(xData_raw, primaryBeamToggles)

            # --- MAKE THE PLOTS ---
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            fig.suptitle(f'Pitch Angle = {data_dict_diffFlux["Pitch_Angle"][0][ptchVal]}$^\circ$ \n {tmeStamp} UTC', fontsize=Title_FontSize)


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
            ax.set_ylim(1E4, 5E7)
            ax.grid(True, alpha=0.4, which='both')
            # ax.text(250, 5E4, s='Primaries', fontsize=Text_FontSize, weight='bold', va='center', ha='center')
            # ax.text(70, 5E4, s='Secondaries/Backscatter', fontsize=Text_FontSize, weight='bold', va='center', ha='center')
            ax.legend(fontsize=Legend_fontSize)

            plt.savefig(rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\fitPhotos\{data_dict_diffFlux["Pitch_Angle"][0][ptchVal]}deg\FitData_{data_dict_diffFlux["Pitch_Angle"][0][ptchVal]}deg_{idx}.png')
            plt.close()