# --- primaryBeamFits_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: using the method outline in Kaeppler's thesis, we can fit inverted-V distributions
# to get estimate the magnetospheric temperature, density and electrostatic potential that accelerated
# our particles

# TODO: Re-work the fitting code to also work if the "Maxwellian" option is selected
# TODO: Perform fits on ACES-I Kaeppler data to compare results of methods!

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
import spaceToolsLib as stl
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

    def n0GuessKaeppler2014(data_dict_diffNFlux, tmeStamp, Te_guess, V0_guess, Kappa_guess, primaryBeamToggles):

        # find when/where this data was
        tmeIdx = np.abs(data_dict_diffNFlux['Epoch'][0] - tmeStamp).argmin()
        EminIdx = np.abs(data_dict_diffNFlux['Energy'][0] - V0_guess).argmin()

        # Get the relevant energy/pitch range
        EnergyRange = data_dict_diffNFlux['Energy'][0][:EminIdx + 1]
        PitchRange = data_dict_diffNFlux['Pitch_Angle'][0][2:10 + 1]  # 10 to 90deg. The 0deg bin DOESN'T matter since this is 0 anyway

        # Collect the numToAverageOver Dataset across pitch angles 0 to 90 for energies >= Emin
        diffNFluxRange =data_dict_diffNFlux['Differential_Number_Flux'][0][tmeIdx - round(primaryBeamToggles.numToAverageOver/2) :tmeIdx + round(primaryBeamToggles.numToAverageOver/2),2:10+1,:EminIdx+1]


        # average the diffNFlux data while excluding fillvals from the mean
        diffNFluxRange[diffNFluxRange < 0] = np.NaN
        diffNFlux_averaged = np.nanmean(diffNFluxRange,axis=0)

        # integrate over energy and pitch angle
        numberFlux_perPitch = (np.cos(np.radians(PitchRange))*
                      np.sin(np.radians(PitchRange))*
                      np.array([simpson(diffNFlux_averaged[i][::-1], EnergyRange[::-1]) for i in range(len(diffNFlux_averaged))])) # Order is inverted to integrate from low Energy to High Energy

        numberFlux = 2*np.pi*simpson(numberFlux_perPitch, PitchRange)

        # calculate the n0Guess
        betaChoice = primaryBeamToggles.beta_guess
        Akappa = (stl.cm_to_m)*(betaChoice/2) *np.sqrt((stl.q0*Te_guess*(2*Kappa_guess-3))/(np.pi*stl.m_e)) * (gamma(Kappa_guess + 1) / ( Kappa_guess*(Kappa_guess-1)*gamma(Kappa_guess-0.5))) # multiplied by 100 to convert to cm/s for unit matching
        n0Guess = numberFlux/(Akappa - Akappa*(1 - 1/betaChoice)*np.power(1 + (V0_guess)/(Te_guess*(Kappa_guess-3/2)*(betaChoice-1)),-(Kappa_guess-1))    )
        return n0Guess


    ##################################
    # --------------------------------
    # --- LOOP THROUGH DATA TO FIT ---
    # --------------------------------
    ##################################
    noiseData = helperFuncs().generateNoiseLevel(data_dict_diffFlux['Energy'][0],primaryBeamToggles)

    EpochFitData, fitData, fitData_stdDev = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                              GenToggles=GenToggles,
                                                                              primaryBeamToggles=primaryBeamToggles)
    for loopIdx, pitchIdx in enumerate(primaryBeamToggles.wPitchsToFit):

            ####################################
            # --- FIT AVERAGED TIME SECTIONS ---
            ####################################
            # for each slice in time, loop over the data and identify the peak differentialNumberFlux (This corresponds to the
            # peak energy of the inverted-V since the location of the maximum number flux tells you what energy the low-energy BULk got accelerated to)
            # Note: The peak in the number flux is very likely the maximum value AFTER 100 eV, just find this point

        for tmeIdx in tqdm(range(len(EpochFitData))):

            # --- Determine the accelerated potential from the peak in diffNflux based on a threshold limit ---
            engythresh_Idx = np.abs(data_dict_diffFlux['Energy'][0] - primaryBeamToggles.engy_Thresh).argmin() # only consider data above a certain index
            peakDiffNIdx = np.nanargmax(fitData[tmeIdx][pitchIdx][:engythresh_Idx + 1]) # only consider data above the Energy_Threshold, to avoid secondaries/backscatter
            dataIdxs = np.array([1 if fitData[tmeIdx][pitchIdx][j] > noiseData[j] and j <= peakDiffNIdx else 0 for j in range(len(data_dict_diffFlux['Energy'][0]))])

            # ---  get the subset of data to fit ---
            fitTheseIndicies = np.where(dataIdxs == 1)[0]
            xData_fit, yData_fit, yData_fit_stdDev = np.array(data_dict_diffFlux['Energy'][0][fitTheseIndicies]), np.array(fitData[tmeIdx][pitchIdx][fitTheseIndicies]), np.array(fitData_stdDev[tmeIdx][pitchIdx][fitTheseIndicies])
            V0_guess = xData_fit[-1]

            # Only include data with non-zero points
            nonZeroIndicies = np.where(yData_fit!=0)[0]
            xData_fit, yData_fit, yData_fit_stdDev = xData_fit[nonZeroIndicies],  yData_fit[nonZeroIndicies], yData_fit_stdDev[nonZeroIndicies]

            ### Perform the fit ###
            params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles, useNoGuess=primaryBeamToggles.useNoGuess)

            if primaryBeamToggles.useFitRefinement:
                ### Use the First fit to motivate the n0 guess ###
                n0guess = n0GuessKaeppler2014(data_dict_diffFlux, EpochFitData[tmeIdx], params[1], params[2], params[3], primaryBeamToggles)

                ### Perform the informed fit again ###
                newBounds = [[n0guess*(1-primaryBeamToggles.n0guess_deviation), n0guess*(1+primaryBeamToggles.n0guess_deviation)],
                             primaryBeamToggles.Te_bounds,
                             [(1 - primaryBeamToggles.V0_deviation) * V0_guess, (1 + primaryBeamToggles.V0_deviation) * V0_guess], #V0
                             [1.5, 30] # kappa
                            ]

                params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles,
                                                   specifyGuess = [n0guess, params[1], params[2], params[3]],
                                                   specifyBoundVals = newBounds,
                                                   useNoGuess = primaryBeamToggles.useNoGuess
                                                   )

            # --- update the data_dict ---
            data_dict['Te'][0][loopIdx].append(params[1])
            data_dict['n'][0][loopIdx].append(params[0])
            data_dict['V0'][0][loopIdx].append(params[2])
            data_dict['kappa'][0][loopIdx].append(0 if primaryBeamToggles.wDistributionToFit == 'Maxwellian' else params[3])
            data_dict['ChiSquare'][0][loopIdx].append(ChiSquare)
            data_dict['dataIdxs'][0][loopIdx].append(dataIdxs)
            data_dict['timestamp_fitData'][0][loopIdx].append(EpochFitData[tmeIdx])
            data_dict['numFittedPoints'][0][loopIdx].append(len(yData_fit))

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