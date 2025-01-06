# --- primaryBeamFits_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: using the method outline in Kaeppler's thesis, we can fit inverted-V distributions
# to get estimate the magnetospheric temperature, density and electrostatic potential that accelerated
# our particles

# TODO: Re-work the fitting code to also work if the "Maxwellian" option is selected

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
import numpy as np
from invertedV_fitting.primaryBeam_fitting.model_primaryBeam_classes import *
import spaceToolsLib as stl
from functools import partial
from time import time
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.special import gamma
start_time = time()
# --- --- --- --- ---



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
    data_dict = {'Te': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Te'}],
                 'n': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm!A-3!N', 'LABLAXIS': 'ne'}],
                 'V0': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'V0'}],
                 'kappa': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'kappa'}],
                 'ChiSquare': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'ChiSquare'}],
                 'dataIdxs': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'Index'}], # indicies corresponding to the NON-ENERGY-REDUCED dataslice so as to re-construct the fit
                 'timestamp_fitData': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm^-2 str^-1 s^-1 ev^-1', 'LABLAXIS': 'V0'}],
                 'Fitted_Pitch_Angles': [data_dict_diffFlux['Pitch_Angle'][0][np.array(primaryBeamToggles.wPitchsToFit)], {'DEPEND_0': None, 'UNITS': 'deg', 'LABLAXIS': 'Fitted Pitch Angle'}],
                 'numFittedPoints': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'Number of Fitted Points'}],
                 }


    # --- NUMBER FLUX DISTRIBUTION FIT FUNCTION ---
    def fitPrimaryBeam(xData, yData, stdDevs, V0_guess, primaryBeamToggles, **kwargs):

        # define the bound values
        boundVals = [primaryBeamToggles.n_bounds, primaryBeamToggles.Te_bounds,
                     [(1 - primaryBeamToggles.V0_deviation) * V0_guess,
                      (1 + primaryBeamToggles.V0_deviation) * V0_guess]]

        # define the guess
        p0guess = [primaryBeamToggles.n_guess, primaryBeamToggles.T_guess, V0_guess]  # construct the guess

        # define the fit function with specific charge/mass
        if primaryBeamToggles.wDistributionToFit == 'Maxwellian':
            fitFunc = partial(fittingDistributions().diffNFlux_fitFunc_Maxwellian, charge=stl.q0, mass=stl.m_e)

        elif primaryBeamToggles.wDistributionToFit == 'Kappa':
            fitFunc = partial(fittingDistributions().diffNFlux_fitFunc_Kappa, charge=stl.q0, mass=stl.m_e)
            boundVals.append(primaryBeamToggles.kappa_bounds)
            p0guess.append(primaryBeamToggles.kappa_guess) # construct the guess

        # modify the boundVals if specified
        if kwargs.get('specifyBoundVals', []) != []:
            boundVals = kwargs.get('specifyBoundVals', [])

        # modify the guess if specified
        if kwargs.get('specifyGuess', []) != []:
            p0guess = kwargs.get('specifyGuess', [])

        # format the bounds
        bounds = tuple([[boundVals[i][0] for i in range(len(boundVals))], [boundVals[i][1] for i in range(len(boundVals))]])

        #########################
        # --- PERFORM THE FIT ---
        #########################
        if len(yData) > 0:
            # --- fit the data ---
            params, cov = curve_fit(fitFunc, xData, yData, maxfev=int(1E4), bounds=bounds, p0=p0guess)

            # --- calculate the Chi Square ---
            ChiSquare = (1 / (len(params) - 1)) * sum([(fitFunc(xData[i], *params) - yData[i]) ** 2 / (stdDevs[i] ** 2) for i in range(len(xData))])

            return params, ChiSquare
        else:
            return [np.NaN for i in range(len(boundVals))],np.NaN

    def n0GuessKaeppler2014(data_dict_diffNFlux, tmeStamp, Te_guess, V0_guess, Kappa_guess, primaryBeamToggles):

        # find when/where this data was
        tmeIdx = np.abs(data_dict_diffNFlux['Epoch'][0] - tmeStamp).argmin()
        EminIdx = np.abs(data_dict_diffNFlux['Energy'][0] - V0_guess).argmin()

        # Get the relevant energy/pitch range
        EnergyRange = data_dict_diffNFlux['Energy'][0][:EminIdx + 1]
        PitchRange = data_dict_diffNFlux['Pitch_Angle'][0][2:10 + 1]  # 10 to 90deg. The 0deg bin DOESN'T matter since this is 0 anyway

        # Collect the numToAverageOver Dataset across pitch angles 0 to 90 for energies >= Emin
        diffNFluxRange =data_dict_diffNFlux['Differential_Number_Flux'][0][tmeIdx - round(primaryBeamToggles.numToAverageOver/2) :tmeIdx + round(primaryBeamToggles.numToAverageOver/2),2:10+1,:EminIdx+1]
        diffNFlux_averaged = np.zeros(shape=(len(PitchRange), len(EnergyRange)))

        for ptchIdx in range(len(PitchRange)):
            # get the energy values for all times but at one specific pitch angle
            engyData = np.array([diffNFluxRange[i][ptchIdx] for i in range(len(diffNFluxRange))]).T
            print(engyData)
            diffNFlux_averaged[ptchIdx] = np.array([np.mean(arr[np.where(arr>=0)[0]]) for arr in engyData])

        # integrate over energy and pitch angle
        numberFlux_perPitch = (np.cos(np.radians(PitchRange))*
                      np.sin(np.radians(PitchRange))*
                      np.array([simpson(diffNFlux_averaged[i], EnergyRange) for i in range(len(diffNFlux_averaged))]))

        print(diffNFlux_averaged[0])
        print(EnergyRange)
        print(PitchRange)
        numberFlux = 2*np.pi*simpson(numberFlux_perPitch, PitchRange)
        print(numberFlux)

        # calculate the n0Guess
        betaChoice = primaryBeamToggles.beta_guess
        Akappa = (betaChoice/2) *np.sqrt((Te_guess*(2*Kappa_guess-3))/(np.pi*stl.m_e)) * (gamma(Kappa_guess + 1) / ( Kappa_guess*(Kappa_guess-1)*gamma(Kappa_guess-0.5)))
        print(Akappa)
        n0Guess = numberFlux/(Akappa - Akappa*(1 - 1/betaChoice)*np.power(1 + (V0_guess)/(Te_guess*(Kappa_guess-3/2)*(betaChoice-1)),-(Kappa_guess-1))    )
        print(n0Guess)
        return n0Guess


    ##################################
    # --------------------------------
    # --- LOOP THROUGH DATA TO FIT ---
    # --------------------------------
    ##################################
    noiseData = helperFitFuncs().generateNoiseLevel(data_dict_diffFlux['Energy'][0],primaryBeamToggles)

    for ptchIdx, pitchVal in enumerate(primaryBeamToggles.wPitchsToFit):

        ##############################
        # --- COLLECT THE FIT DATA ---
        ##############################
        # ensure the data is divided into chunks that can be sub-divided. If not, keep drop points from the end until it can be
        low_idx, high_idx = np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][0]).argmin(), np.abs( data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][1]).argmin()

        if (high_idx -low_idx)%primaryBeamToggles.numToAverageOver != 0:
            high_idx -= (high_idx -low_idx)%primaryBeamToggles.numToAverageOver

        chunkedEpoch = np.split(data_dict_diffFlux['Epoch'][0][low_idx:high_idx], round(len(data_dict_diffFlux['Epoch'][0][low_idx:high_idx + 1])/primaryBeamToggles.numToAverageOver) )
        chunkedyData = np.split(data_dict_diffFlux['Differential_Number_Flux'][0][low_idx:high_idx, pitchVal, :], round(len(data_dict_diffFlux['Differential_Number_Flux'][0][low_idx:high_idx, pitchVal, :])/primaryBeamToggles.numToAverageOver))
        chunkedStdDevs = np.split(data_dict_diffFlux['Differential_Number_Flux_stdDev'][0][low_idx:high_idx, pitchVal, :],round(len(data_dict_diffFlux['Differential_Number_Flux_stdDev'][0][low_idx:high_idx, pitchVal, :])/primaryBeamToggles.numToAverageOver))

        # --- Average the chunked data ---
        EpochFitData = []
        fitData = np.zeros(shape=(len(chunkedyData), len(data_dict_diffFlux['Energy'][0]) ))
        fitData_stdDev = np.zeros(shape=(len(chunkedStdDevs), len(data_dict_diffFlux['Energy'][0])))

        for i in range(len(chunkedEpoch)):
            EpochFitData.append(chunkedEpoch[i][ int((primaryBeamToggles.numToAverageOver-1)/2)]) # take the middle timestamp value

            # average the diffFlux data by only choosing data which is valid
            fitData[i]=np.array([np.mean(arr[np.where(arr>0)[0]]) for arr in chunkedyData[i].T])

            # average the diffFlux data by only choosing data which is valid
            fitData_stdDev[i] = np.array([np.mean(arr[np.where(arr > 0)[0]]) for arr in chunkedStdDevs[i].T])

        ####################################
        # --- FIT AVERAGED TIME SECTIONS ---
        ####################################
        # for each slice in time, loop over the data and identify the peak differentialNumberFlux (This corresponds to the
        # peak energy of the inverted-V since the location of the maximum number flux tells you what energy the low-energy BULk got accelerated to)
        # Note: The peak in the number flux is very likely the maximum value AFTER 100 eV, just find this point

        for tmeIdx in tqdm(range(len(EpochFitData))):

            # --- Determine the accelerated potential from the peak in diffNflux based on a threshold limit ---
            engythresh_Idx = np.abs(data_dict_diffFlux['Energy'][0] - primaryBeamToggles.engy_Thresh).argmin() # only consider data above a certain index
            peakDiffNIdx = np.nanargmax(fitData[tmeIdx][:engythresh_Idx]) # in that data subset, find where the maximum diffNFlux occurs AND ADD 1 MORE ENERGY TO IT, similar to KAEPPLER
            dataIdxs = np.array([1 if i <= peakDiffNIdx and fitData[tmeIdx][i] > noiseData[i] else 0 for i in range(len(data_dict_diffFlux['Energy'][0]))]) # put a 0 or 1 in the original data length indicating if that datapoint was used for fitting

            # ---  get the subset of data to fit ---
            fitTheseIndicies = np.where(dataIdxs == 1)[0]
            xData_fit, yData_fit, yData_fit_stdDev = np.array(data_dict_diffFlux['Energy'][0][fitTheseIndicies]), np.array(fitData[tmeIdx][fitTheseIndicies]), np.array(fitData_stdDev[tmeIdx][fitTheseIndicies])
            V0_guess = xData_fit[-1]

            # Only include data with non-zero points
            nonZeroIndicies = np.where(yData_fit!=0)[0]
            xData_fit, yData_fit, yData_fit_stdDev = xData_fit[nonZeroIndicies],  yData_fit[nonZeroIndicies], yData_fit_stdDev[nonZeroIndicies]

            ### Perform the fit ###
            params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles)

            ### Use the First fit to motivate the n0 guess ###
            n0guess = n0GuessKaeppler2014(data_dict_diffFlux, EpochFitData[tmeIdx], params[1], params[2], params[3], primaryBeamToggles)

            ### Perform the informed fit again ###
            params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles,
                                               specifyGuess=[n0guess, params[1],params[2],params[3]],
                                               specifyBoundVals= [
                                                   [n0guess*(1-primaryBeamToggles.n0guess_deviation), n0guess*(1+primaryBeamToggles.n0guess_deviation)],
                                                   primaryBeamToggles.Te_bounds,
                                                   [(1 - primaryBeamToggles.V0_deviation) * V0_guess, (1 + primaryBeamToggles.V0_deviation) * V0_guess], #V0
                                                   [1.5, 30] # kappa
                                               ]
                                               )


            # --- update the data_dict ---
            data_dict['Te'][0][ptchIdx].append(params[1])
            data_dict['n'][0][ptchIdx].append(params[0])
            data_dict['V0'][0][ptchIdx].append(params[2])
            data_dict['kappa'][0][ptchIdx].append(0 if primaryBeamToggles.wDistributionToFit == 'Maxwellian' else params[3])
            data_dict['ChiSquare'][0][ptchIdx].append(ChiSquare)
            data_dict['dataIdxs'][0][ptchIdx].append(dataIdxs)
            data_dict['timestamp_fitData'][0][ptchIdx].append(EpochFitData[tmeIdx])
            data_dict['numFittedPoints'][0][ptchIdx].append(len(yData_fit))

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

    outputPath = rf'C:\Data\physicsModels\invertedV\primaryBeam_Fitting\primaryBeam_fitting_parameters.cdf'
    stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict)