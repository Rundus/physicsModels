# --- primaryBeamFits_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: using the method outline in Kaeppler's thesis, we can fit inverted-V distributions
# to get estimate the magnetospheric temperature, density and electrostatic potential that accelerated
# our particles
import numpy as np

# TODO: Re-work the fitting code to also work if the "Maxwellian" option is selected
# TODO: Perform fits on ACES-I Kaeppler data to compare results of methods!

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import GenToggles
import spaceToolsLib as stl
from time import time
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.special import gamma
import warnings
warnings.filterwarnings("ignore")
start_time = time()
# --- --- --- --- ---

def generatePrimaryBeamFit(primaryBeamToggles, outputFolder):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=primaryBeamToggles.inputDataPath,
                                              wKeys_Reduce=['Differential_Energy_Flux',
                                                            'Differential_Number_Flux',
                                                            'Epoch',
                                                            'Differential_Number_Flux_stdDev'])

    # --- prepare the output data_dict_output ---
    data_dict_output = {'Te': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1':'Pitch_Angle',  'UNITS': 'eV', 'LABLAXIS': 'Te'}],
                 'n': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1':'Pitch_Angle','UNITS': 'cm!A-3!N', 'LABLAXIS': 'ne'}],
                 'V0': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1':'Pitch_Angle','UNITS': 'eV', 'LABLAXIS': 'V0'}],
                 'kappa': [[], {'DEPEND_0': 'Epoch', 'DEPEND_1':'Pitch_Angle','UNITS': None, 'LABLAXIS': 'kappa'}],
                 'ChiSquare': [[], {'DEPEND_0': 'Epoch', 'UNITS': None, 'LABLAXIS': 'ChiSquare'}],
                 'timestamp_fitData': [[], {'DEPEND_0': None, 'UNITS': 'cm^-2 str^-1 s^-1 ev^-1', 'LABLAXIS': 'V0'}],
                 'Fitted_Pitch_Angles': [primaryBeamToggles.wPitchsToFit, {'DEPEND_0': None, 'UNITS': 'deg', 'LABLAXIS': 'Fitted Pitch Angle'}],
                 'dataIdxs': [[], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'Index'}], # indicies corresponding to the NON-ENERGY-REDUCED dataslice so as to re-construct the fit
                 'numFittedPoints': [[], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'Number of Fitted Points'}],
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
            fitFunc = primaryBeam_class().diffNFlux_fitFunc_Maxwellian

        elif primaryBeamToggles.wDistributionToFit == 'Kappa':
            fitFunc = primaryBeam_class().diffNFlux_fitFunc_Kappa
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

            if kwargs.get('useNoGuess',False):
                params, cov = curve_fit(fitFunc, xData, yData, maxfev=primaryBeamToggles.maxfev, bounds=bounds)
            else:
                # --- fit the data ---
                params, cov = curve_fit(fitFunc, xData, yData, maxfev=primaryBeamToggles.maxfev, bounds=bounds, p0=p0guess)

            # --- calculate the Chi Square ---
            ChiSquare = (1 / (len(params) - 1)) * sum([(fitFunc(xData[i], *params) - yData[i]) ** 2 / (stdDevs[i] ** 2) for i in range(len(xData))])

            return params, ChiSquare
        else:
            return [np.NaN for i in range(len(boundVals))],np.NaN

    def n0GuessKaeppler2014(jN, firstFitParams, peakDiffNIdx, beta, energyRange):
        '''
        :param jN: 2D array of averaged diffNFlux data for a single time slice from pitch angles -10deg to 190 deg
        :param firstFitParams: 1D array with format [n0_guess, T_guess, V0_guess, kappa_guess]
        :param beta: scalar ratio of Bmax/Bmin
        :return:
        n0Guess: scalar value
        '''

        # clean up data before integration
        jN[jN < 0] = np.NaN
        jN[np.where(np.isnan(jN) == True)] = 0

        # only get pitch angles 0 to 90deg and reformat the data into [energy, pitch angles]
        jN = jN[1:10+1, :].T

        # --- integrate jN over parallel pitch angles ---
        pitchRange = np.radians([0 + 10*i for i in range(10)])
        jN_pitch = np.array([np.cos(pitchRange)*np.sin(pitchRange)*jN[engyIdx] for engyIdx in range(len(jN))])
        varPhi_E_parallel = 2*np.pi*np.array([simpson(x=pitchRange, y=arr) for arr in jN_pitch])

        # --- integrate varPhi_E over the BEAM energy range ---
        beamEnergies = energyRange[:peakDiffNIdx+1]
        beamFlux = varPhi_E_parallel[:peakDiffNIdx+1]
        numberFlux = -1*simpson(x=beamEnergies, y=beamFlux) # multiply by -1 b/c I'm integrating over high-to-low energies

        # calculate the n0Guess
        Te = firstFitParams[1]
        kappa = firstFitParams[3]
        V0 = energyRange[peakDiffNIdx]
        Akappa = (stl.cm_to_m)*(beta/2) *np.sqrt((stl.q0*Te*(2*kappa-3))/(np.pi*stl.m_e)) * (gamma(kappa + 1) / ( kappa*(kappa-1)*gamma(kappa-0.5))) # multiplied by 100 to convert to cm/s for unit matching
        n0Guess = numberFlux/(Akappa - Akappa*(1 - 1/beta)*np.power(1 + V0/(Te*(kappa-3/2)*(beta-1)),-(kappa-1))    )

        return n0Guess

    ##################################
    # --------------------------------
    # --- LOOP THROUGH DATA TO FIT ---
    # --------------------------------
    ##################################
    noiseData = helperFuncs().generateNoiseLevel(data_dict_diffFlux['Energy'][0], countNoiseLevel=primaryBeamToggles.countNoiseLevel)

    EpochFitData, IlatFitData, fitData, fitData_stdDev = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                              targetTimes=[data_dict_diffFlux['Epoch'][0][0], data_dict_diffFlux['Epoch'][0][-1]],
                                                                              N_avg=primaryBeamToggles.numToAverageOver)

    data_dict_output['dataIdxs'][0] = np.zeros(shape = (len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0]), len(data_dict_diffFlux['Energy'][0])))
    data_dict_output['numFittedPoints'][0] = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0])))
    data_dict_output['Te'][0] = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0])))
    data_dict_output['n'][0] = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0])))
    data_dict_output['V0'][0] = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0])))
    data_dict_output['kappa'][0] = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0])))
    data_dict_output['ChiSquare'][0] = np.zeros(shape=(len(EpochFitData), len(data_dict_diffFlux['Pitch_Angle'][0])))
    data_dict_output['timestamp_fitData'][0] = deepcopy(EpochFitData)

    data_dict_output ={**data_dict_output,
                       **{'Pitch_Angle': deepcopy(data_dict_diffFlux['Pitch_Angle']),
                          'Epoch': [EpochFitData, deepcopy(data_dict_diffFlux['Epoch'][1])],
                          'ILat': [IlatFitData, deepcopy(data_dict_diffFlux['ILat'][1])]
                          }
                       }

    # --- get the indicies to fit --
    # description: get the indicies of the invertedV-times to fit from the EpochFitData
    fit_these_ranges = []
    for time_range in GenToggles.invertedV_times:
        low_idx, high_idx = np.abs(EpochFitData - time_range[0]).argmin(),np.abs(EpochFitData - time_range[1]).argmin()
        fit_these_ranges.append([i for i in range(low_idx,high_idx+1, 1)])
    fit_these_ranges = np.array([val for sublist in fit_these_ranges for val in sublist])

    ###################
    # --- MAIN LOOP ---
    ###################
    for loopIdx, pitchAngle in enumerate(primaryBeamToggles.wPitchsToFit):
        ####################################
        # --- FIT AVERAGED TIME SECTIONS ---
        ####################################
        # for each slice in time, loop over the data and identify the peak differentialNumberFlux (This corresponds to the
        # peak energy of the inverted-V since the location of the maximum number flux tells you what energy the low-energy BULk got accelerated to)
        # Note: The peak in the number flux is very likely the maximum value AFTER 100 eV, just find this point
        pitchIdx = np.abs(data_dict_diffFlux['Pitch_Angle'][0] - pitchAngle).argmin()

        for tmeIdx in tqdm(fit_these_ranges):
            try: # try fit the data
                # --- Determine the accelerated potential from the peak in diffNflux based on a threshold limit ---
                engythresh_Idx = np.abs(data_dict_diffFlux['Energy'][0] - primaryBeamToggles.engy_Thresh).argmin()  # only consider data above a certain index
                peakDiffNIdx = np.nanargmax(fitData[tmeIdx][pitchIdx][:engythresh_Idx + 1])  # only consider data above the Energy_Threshold, to avoid secondaries/backscatter

                # Determine which datapoints are good to fit
                # Description: After finding the peak in the jN spectrum, fit the data starting 1 BEFORE this peak value
                peakIdx = peakDiffNIdx  # Kaeppler added +1 here to make his ChiSquare's get better. Our are MUCH better without doing that
                dataIdxs = np.array([1 if fitData[tmeIdx][pitchIdx][j] > noiseData[j] and j <= peakIdx else 0 for j in range(len(data_dict_diffFlux['Energy'][0]))])

                ###################################
                # ---  get the Beam data subset ---
                ###################################
                fitTheseIndicies = np.where(dataIdxs == 1)[0]
                xData_fit, yData_fit, yData_fit_stdDev = np.array(data_dict_diffFlux['Energy'][0][fitTheseIndicies]), np.array(fitData[tmeIdx][pitchIdx][fitTheseIndicies]), np.array(fitData_stdDev[tmeIdx][pitchIdx][fitTheseIndicies])
                V0_guess = xData_fit[-1]

                # Only include data with non-zero points
                nonZeroIndicies = np.where(yData_fit!=0)[0]
                xData_fit, yData_fit, yData_fit_stdDev = xData_fit[nonZeroIndicies],  yData_fit[nonZeroIndicies], yData_fit_stdDev[nonZeroIndicies]

                ### Perform the fit ###
                params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles, useNoGuess=primaryBeamToggles.useNoGuess)

                ### Kaeppler fit refinement ###
                if primaryBeamToggles.useFitRefinement:

                    ### Use the First fit to motivate the n0 guess ###
                    firstFitParams = deepcopy(params)  # construct the guess

                    # define the fit function with specific charge/mass
                    if primaryBeamToggles.wDistributionToFit == 'Maxwellian':
                        firstFitParams[3] = 10

                    ### Use the First fit to motivate the n0 guess ###
                    n0guess = n0GuessKaeppler2014(fitData[tmeIdx],
                                                  firstFitParams,
                                                  peakDiffNIdx,
                                                  beta=primaryBeamToggles.beta_guess,
                                                  energyRange=data_dict_diffFlux['Energy'][0])

                    ### Perform the informed fit again ###
                    newBounds = [[n0guess*(1-primaryBeamToggles.n0guess_deviation), n0guess*(1+primaryBeamToggles.n0guess_deviation)],
                                 primaryBeamToggles.Te_bounds,
                                 [(1 - primaryBeamToggles.V0_deviation) * V0_guess, (1 + primaryBeamToggles.V0_deviation) * V0_guess], #V0
                                 primaryBeamToggles.kappa_bounds # kappa
                                ]
                    params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles,
                                                       specifyGuess = [n0guess, params[1], params[2], params[3]],
                                                       specifyBoundVals = newBounds,
                                                       useNoGuess = primaryBeamToggles.useNoGuess
                                                       )

                # --- update the data_dict_output ---
                data_dict_output['Te'][0][tmeIdx][pitchIdx] = params[1]
                data_dict_output['n'][0][tmeIdx][pitchIdx] = params[0]
                data_dict_output['V0'][0][tmeIdx][pitchIdx] = params[2]
                data_dict_output['kappa'][0][tmeIdx][pitchIdx] = 0 if primaryBeamToggles.wDistributionToFit == 'Maxwellian' else params[3]
                data_dict_output['ChiSquare'][0][tmeIdx][pitchIdx]= ChiSquare
                data_dict_output['dataIdxs'][0][tmeIdx][pitchIdx] = dataIdxs
                data_dict_output['numFittedPoints'][0][tmeIdx][pitchIdx] = len(yData_fit)
            except: # output all nans
                # --- update the data_dict_output ---
                data_dict_output['Te'][0][tmeIdx][pitchIdx] = np.nan
                data_dict_output['n'][0][tmeIdx][pitchIdx] = np.nan
                data_dict_output['V0'][0][tmeIdx][pitchIdx] = np.nan
                data_dict_output['kappa'][0][tmeIdx][pitchIdx] = np.nan
                data_dict_output['ChiSquare'][0][tmeIdx][pitchIdx] = np.nan
                data_dict_output['dataIdxs'][0][tmeIdx][pitchIdx] = np.nan
                data_dict_output['numFittedPoints'][0][tmeIdx][pitchIdx] = np.nan

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

    outputPath = rf'{outputFolder}\primaryBeam_fitting_parameters.cdf'
    stl.outputCDFdata(outputPath=outputPath,data_dict=data_dict_output)