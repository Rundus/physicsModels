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
                 }




    # --- NUMBER FLUX DISTRIBUTION FIT FUNCTION ---
    def fitPrimaryBeam(xData, yData, stdDevs, V0_guess, primaryBeamToggles):

        # get the bounds
        boundVals = [primaryBeamToggles.n_bounds, primaryBeamToggles.Te_bounds, [(1 - primaryBeamToggles.V0_deviation) * V0_guess, (1 + primaryBeamToggles.V0_deviation) * V0_guess]]

        # define the fit function with specific charge/mass
        if primaryBeamToggles.wDistributionToFit == 'Maxwellian':
            fitFunc = partial(fittingDistributions().diffNFlux_fitFunc_Maxwellian, charge=stl.q0, mass=stl.m_e)

        elif primaryBeamToggles.wDistributionToFit == 'Kappa':
            fitFunc = partial(fittingDistributions().diffNFlux_fitFunc_Kappa, charge=stl.q0, mass=stl.m_e)
            boundVals.append(primaryBeamToggles.kappa_bounds)

        # format the bounds
        bounds = tuple([[boundVals[i][0] for i in range(len(boundVals))], [boundVals[i][1] for i in range(len(boundVals))]])

        # --- fit the data ---
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
            engythresh_Idx = np.abs(data_dict_diffFlux['Energy'][0] - primaryBeamToggles.engy_Thresh).argmin() # only consider data above a certain index
            peakDiffNIdx = np.argmax(fitData[tmeIdx][:engythresh_Idx]) # in that data subset, find where the maximum diffNFlux occurs
            dataIdxs = np.array([1 if i <= peakDiffNIdx and fitData[tmeIdx][i]>0 else 0 for i in range(len(data_dict_diffFlux['Energy'][0]))]) # put a 0 or 1 in the original data length indicating if that datapoint was used for fitting

            # ---  get the subset of data to fit ---
            xData_fit, yData_fit, yData_fit_stdDev = np.array(data_dict_diffFlux['Energy'][0][:peakDiffNIdx+1]), np.array(fitData[tmeIdx][:peakDiffNIdx+1]), np.array(fitData_stdDev[tmeIdx][:peakDiffNIdx+1])
            V0_guess = xData_fit[-1]

            # Only include data with non-zero points
            nonZeroIndicies = np.where(yData_fit!=0)[0]
            xData_fit, yData_fit, yData_fit_stdDev = xData_fit[nonZeroIndicies],  yData_fit[nonZeroIndicies], yData_fit_stdDev[nonZeroIndicies]

            ### Perform the fit ###
            params, ChiSquare = fitPrimaryBeam(xData_fit, yData_fit, yData_fit_stdDev, V0_guess, primaryBeamToggles)

            # --- update the data_dict ---
            data_dict['Te'][0][ptchIdx].append(params[1])
            data_dict['n'][0][ptchIdx].append(params[0])
            data_dict['V0'][0][ptchIdx].append(params[2])
            data_dict['kappa'][0][ptchIdx].append(0 if primaryBeamToggles.wDistributionToFit == 'Maxwellian' else params[3])
            data_dict['ChiSquare'][0][ptchIdx].append(ChiSquare)
            data_dict['dataIdxs'][0][ptchIdx].append(dataIdxs)
            data_dict['timestamp_fitData'][0][ptchIdx].append(data_dict_diffFlux['Epoch'][0][tmeIdx])

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