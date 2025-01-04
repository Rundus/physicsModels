# --- primaryBeamFits_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: using the method outline in Kaeppler's thesis, we can fit inverted-V distributions
# to get estimate the magnetospheric temperature, density and electrostatic potential that accelerated
# our particles


# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from invertedV_fitting.primaryBeam_fitting.model_primaryBeam_classes import *
import spaceToolsLib as stl
from functools import partial
from time import time
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
start_time = time()
# --- --- --- --- ---

##################
# --- PLOTTING ---
##################
figure_width = 10 # in inches
figure_height =8 # in inches
Title_FontSize = 20
Label_FontSize = 20
Label_Padding = 8
Tick_FontSize = 12
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

def generatePrimaryBeamFit(GenToggles, primaryBeamToggles, **kwargs):


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

    # --- prepare the output data_dict ---
    showPlot = kwargs.get('showPlot', False)
    data_dict = {'Te': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Te [eV]'}],
                 'n': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm!A-3!N', 'LABLAXIS': 'ne'}],
                 'V0': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'V0'}],
                 'kappa': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'kappa'}],
                 'ChiSquare': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm^-2 str^-1 s^-1 ev^-1', 'LABLAXIS': 'V0'}],
                 'energy_FitData': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'V0'}],
                 'diffNflux_fitData': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm^-2 str^-1 s^-1 ev^-1', 'LABLAXIS': 'V0'}],
                 'timestamp_fitData': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm^-2 str^-1 s^-1 ev^-1', 'LABLAXIS': 'V0'}],
                 'Fitted_Pitch_Angles': [data_dict_diffFlux['Pitch_Angle'][0][np.array(primaryBeamToggles.wPitchsToFit)], {'DEPEND_0': None, 'UNITS': 'deg', 'LABLAXIS': 'Fitted Pitch Angle'}],
                 }

    # --- calculate noise level ---
    def generateNoiseLevel(energyData, fitToggles):
        count_interval = 0.8992E-3
        geo_factor = 8.63E-5
        deadtime = 324E-9

        # --- DEFINE THE NOISE LEVEL ---
        diffNFlux_NoiseCount = np.zeros(shape=(len(energyData)))

        for engy in energyData:
            deltaT = (count_interval) - (fitToggles.countNoiseLevel * deadtime)
            diffNFlux_NoiseCount[engy] = (fitToggles.countNoiseLevel) / (geo_factor * deltaT * engy)

        return diffNFlux_NoiseCount


    # --- NUMBER FLUX DISTRIBUTION FIT FUNCTION ---
    def fitPrimaryBeam(xData, yData, stdDevs, V0_guess,wDistFunc):
        # take in the data to be fit, and return the fit parameters and fitted data

        # define the fit function with specific charge/mass
        if wDistFunc == 'Maxwellian':
            fitFunc = partial(fittingDistributions().diffNFlux_fitFunc_Maxwellian, charge=stl.q0, mass=stl.m_e)
        elif wDistFunc == 'Kappa':
            fitFunc = partial(fittingDistributions().diffNFlux_fitFunc_Kappa, charge=stl.q0, mass=stl.m_e)

        # --- FIT THE DATA ---
        deviation = 0.18
        guess = [1, 120, 250]  # observed plasma at dispersive region is 0.5E5 cm^-3 BUT this doesn't make sense to use as the kappa fit since the kappa fit comes from MUCH less dense populations above
        boundVals = [[0.001, 8],  # n [cm^-3]
                     [10, 500],  # T [eV]
                     [(1 - deviation) * V0_guess, (1 + deviation) * V0_guess]]  # V [eV]

        bounds = tuple([[boundVals[i][0] for i in range(len(boundVals))], [boundVals[i][1] for i in range(len(boundVals))]])
        params, cov = curve_fit(fitFunc, xData, yData, maxfev=int(1E9), bounds=bounds)

        # --- calculate the Chi Square ---
        ChiSquare = (1 / (len(params) - 1)) * sum([(fitFunc(xData[i], *params) - yData[i]) ** 2 / (stdDevs[i] ** 2) for i in range(len(xData))])

        return params, ChiSquare


    def refineFitPrimaryBeam(xData,yData, stdDevs, wDistFunc):
        # --- --- --- --- --- --- ---
        # --- SECOND ITERATION FIT ---
        # --- --- --- --- --- --- ---
        # --- Determine the number flux at this time for pitch angles 0 to 90 deg and energies above the parallel potential ---
        # PitchIndicies = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        # allPitchFitData = diffNFlux[low_idx + tmeIdx, 1:PitchIndicies[-1]+1, :peakDiffNVal_index]
        # numberFlux_perPitch = []
        #
        # for idx, ptch in enumerate(PitchIndicies):  # integrate over energy
        #     dataValues = allPitchFitData[idx]
        #     dataValues_cleaned = np.array([val if val > 0 else 0 for val in dataValues])  # Clean up the data by setting fillvalues to == 0
        #     EnergyValues = Energy[:peakDiffNVal_index]
        #     numberFlux_perPitch.append(simpson(y=np.cos(np.radians(Pitch[ptch])) * np.sin(np.radians(Pitch[ptch])) * dataValues_cleaned, x=EnergyValues))
        #
        # # integrate over pitch
        # numberFlux = (2*2*np.pi)*simpson(y=-1*np.array(numberFlux_perPitch), x=Pitch[PitchIndicies])
        #
        # # define the new fit parameters
        # # n0Guess = (1E-2) * np.sqrt((2*np.pi*m_e*params[1])/q0) * numberFlux / params[2] # simple verison
        # # print(numberFlux, n0Guess)
        # betaChoice = 6
        # n0Guess = (-1E-6)*numberFlux*np.power(q0*params[1]/(2*np.pi*m_e),-0.5)*np.power(betaChoice*(1 - (1 - 1/betaChoice)*np.exp(-(0-parallelPotential)/(params[1]*(betaChoice-1)))),-1)
        #
        # # Second iterative fit
        # deviation = 0.18
        # guess = [n0Guess, 120, 250]  # observed plasma at dispersive region is 0.5E5 cm^-3 BUT this doesn't make sense to use as the kappa fit since the kappa fit comes from MUCH less dense populations above
        # boundVals = [[0.0001, 8],  # n [cm^-3]
        #              [10, 300],  # T [eV]
        #              [100, 400]]  # V [eV]
        #
        # bounds = tuple([[boundVals[i][0] for i in range(len(boundVals))],
        #                 [boundVals[i][1] for i in range(len(boundVals))]])
        # params, cov = curve_fit(fitFuncAtPitch, xData_fit, yData_fit, maxfev=int(1E9), bounds=bounds)
        #
        # # --- Fit the second iteration fit ---
        # fittedX = np.linspace(xData_fit.min(), xData_fit.max(), 100)
        # fittedY = fitFuncAtPitch(fittedX,*params)
        return


    ##################################
    # --------------------------------
    # --- LOOP THROUGH DATA TO FIT ---
    # --------------------------------
    ##################################
    for ptchIdx, pitchVal in enumerate(primaryBeamToggles.wPitchsToFit):

        ##############################
        # --- COLLECT THE FIT DATA ---
        ##############################
        # collect the data to fit one single pitch
        low_idx, high_idx = np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][0]).argmin(), np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][1]).argmin()
        EpochFitData = data_dict_diffFlux['Epoch'][0][low_idx:high_idx + 1]
        fitData = data_dict_diffFlux['Differential_Number_Flux'][0][low_idx:high_idx+1, pitchVal, :]
        fitData_stdDev = data_dict_diffFlux['Differential_Number_Flux_stdDev'][0][low_idx:high_idx+1, pitchVal, :]

        ##############################
        # --- FIT EVERY TIME SLICE ---
        ##############################
        # for each slice in time, loop over the data and identify the peak differentialNumberFlux (This corresponds to the
        # peak energy of the inverted-V since the location of the maximum number flux tells you what energy the low-energy BULk got accelerated to)
        # Note: The peak in the number flux is very likely the maximum value AFTER 100 eV, just find this point
        for tmeIdx in tqdm(range(len(EpochFitData))):

            # --- Determine the accelerated potential from the peak in diffNflux based on a threshold limit ---
            engythresh_Idx = np.abs(data_dict_diffFlux['Energy'][0] - primaryBeamToggles.engy_Thresh).argmin()
            peakDiffNIdx = np.argmax(fitData[tmeIdx][:engythresh_Idx])

            # ---  get the subset of data to fit ---
            xData_fit, yData_fit, yData_fit_stdDev = np.array(data_dict_diffFlux['Energy'][0][:peakDiffNIdx+1]), np.array(fitData[tmeIdx][:peakDiffNIdx+1]), np.array(fitData_stdDev[tmeIdx][:peakDiffNIdx+1])
            V0_guess = xData_fit[-1]

            # Only include data with non-zero points
            nonZeroIndicies = np.where(yData_fit!=0)[0]
            xData_fit, yData_fit, yData_fit_stdDev = xData_fit[nonZeroIndicies],  yData_fit[nonZeroIndicies], yData_fit_stdDev[nonZeroIndicies]


            ### Perform the fit ###
            params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev,V0_guess, primaryBeamToggles.wDistributionToFit)

            # --- update the data_dict ---
            data_dict['Te'][0][ptchIdx].append(params[1])
            data_dict['n'][0][ptchIdx].append(params[0])
            data_dict['V0'][0][ptchIdx].append(params[2])
            data_dict['kappa'][0][ptchIdx].append(0 if primaryBeamToggles.wDistributionToFit == 'Maxwellian' else params[3])
            data_dict['ChiSquare'][0][ptchIdx].append(ChiSquare)
            # data_dict['energy_FitData'][0][ptchIdx].append(xData)
            # data_dict['diffNflux_fitData'][0][ptchIdx].append(yData)
            data_dict['timestamp_fitData'][0][ptchIdx].append(data_dict_diffFlux['Epoch'][0][tmeIdx])

    # --- --- --- --- --- ---
    # --- PLOT THE DATA ---
    # --- --- --- --- --- ---
    def makePlotOfFits:
        if kwargs.get('showPlot', False):
            fittedX = np.linspace(xData_fit.min(), xData_fit.max(), 100)
            fittedY = fitFunc(fittedX, *params)
            tmeIdx = np.abs(data_dict_diffFlux['Epoch'][0] - timeStamp).argmin()

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            fig.suptitle(f'Pitch Angle = {data_dict_diffFlux["Pitch"][0][pitchIdx]} \n {timeStamp} UTC',
                         fontsize=Title_FontSize)

            ax.plot(xData, fitData[tmeIdx][:], '-o')
            ax.plot(fittedX, fittedY, color='red',
                    label=f'n = {round(params[0], 1)}' + ' cm$^{-3}$' + f'\n T = {round(params[1], 1)} eV\n' + f'V = {round(params[2], 1)} eV\n' + r'$\chi^{2}_{\nu}= $' + f'{round(ChiSquare, 3)}')

            ax.tick_params(axis='y', which='major', colors='black', labelsize=Tick_FontSize, length=Tick_Length,
                           width=Tick_Width)
            ax.tick_params(axis='y', which='minor', colors='black', labelsize=Tick_FontSize_minor,
                           length=Tick_Length_minor, width=Tick_Width_minor)

            ax.axvline(xData[np.argmax(yData)], color='red')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Energy [eV]', fontsize=Label_FontSize)
            ax.set_ylabel('diffNFlux [cm$^{-2}$s$^{-1}$str$^{-1}$ eV/eV]', fontsize=Label_FontSize - 4)
            ax.set_xlim(28, 1E4)
            ax.set_ylim(1E4, 5E7)

            # plot the noise
            ax.plot(xData, generateNoiseLevel(xData, primaryBeamToggles), color='black',
                    label=f'{primaryBeamToggles.countNoiseLevel}-count noise')
            ax.legend(fontsize=Legend_fontSize)
            plt.savefig(
                rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\fitPhotos\{data_dict_diffFlux["Pitch"][0][pitchIdx]}deg\FitData_Pitch{data_dict_diffFlux["Pitch"][0][pitchIdx]}_{tmeIdx}.png')
            plt.close()

        return data_dict
    def makePlotOfFitStatistics:
        # --- output Plot of the time series of modeled data ---
        fig, ax = plt.subplots(4, sharex=True)
        fig.set_size_inches(figure_width, figure_height)
        fig.suptitle(f'Pitch = {Pitch[pitchVal]}',fontsize=Label_FontSize)

        # Density
        ax[0].scatter(paramTime,modeled_n,marker='o')
        avg_n = round(sum(modeled_n)/len(modeled_n),1)
        ax[0].axhline(avg_n,color='red',label=rf'n (Avg) = {avg_n}')# plot the average value
        ax[0].legend(fontsize= Legend_fontSize)
        ax[0].set_ylim(0,5)
        ax[0].set_ylabel('n  [cm$^{-3}$]')

        # Temperature
        ax[1].scatter(paramTime,modeled_T,marker='o')
        avg_T = round(sum(modeled_T)/len(modeled_T),1)
        ax[1].axhline(avg_T,color='red',label=rf'T (Avg) = {avg_T} ')# plot the average value
        ax[1].legend(fontsize= Legend_fontSize)
        ax[1].set_ylim(10,1000)
        ax[1].set_ylabel('T [eV]')
        ax[1].set_yscale('log')

        # Potential
        ax[2].scatter(paramTime,modeled_V,marker='o')
        avg_V = round(sum(modeled_V)/len(modeled_V),1)
        ax[2].axhline(avg_V,color='red',label=rf'V (Avg) = {avg_V}')# plot the average value
        ax[2].legend(fontsize= Legend_fontSize)
        ax[2].set_ylim(10, 500)
        ax[2].set_ylabel('V [eV]')
        ax[2].set_yscale('log')

        #ChiSquare
        ax[3].scatter(paramTime, modeled_ChiVal, marker='o')
        avg_Chi = round(sum(modeled_ChiVal) / len(modeled_ChiVal), 1)
        ax[3].axhline(avg_Chi, color='red', label=rf'$\chi$ (Avg) = {avg_Chi}')  # plot the average value
        ax[3].legend(fontsize=Legend_fontSize)
        ax[3].set_ylim(0.1, 10)
        ax[3].set_ylabel(r'$\chi ^{2}_{\nu}$')
        ax[3].set_yscale('log')

        if wInvertedV == 0:
            outputPath = rf'C:\Data\ACESII\science\invertedV\TempDensityPotential_Fitting\DispersiveRegion\Parameters_Pitch_{pitchVal}.png'
        else:
            outputPath = rf'C:\Data\ACESII\science\invertedV\TempDensityPotential_Fitting\PrimaryInvertedV\Parameters_Pitch_{pitchVal}.png'
        plt.savefig(outputPath)

    # --- --- --- --- --- ---
    # --- OUTPUT THE DATA ---
    # --- --- --- --- --- ---
    # Construct the Data Dict
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': None}

    # update the data dict attrs
    for key, val in data_dict.items():
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict[key][1].items():
            newAttrs[subKey] = subVal

        data_dict[key][1] = newAttrs

    outputPath = rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\primaryBeam_fitting.cdf'
    stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict)