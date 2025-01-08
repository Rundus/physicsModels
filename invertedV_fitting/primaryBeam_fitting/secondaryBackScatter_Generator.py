# --- secondaryBackScatter_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take in the primary data fits and produce secondary/backscatter curves for each. Use the
# parameterized Evans 1964 curves to determine the curves.

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
from scipy.integrate import simpson
from scipy.special import gamma
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

    # Epoch_groupAverage, fitData_groupAverage, stdDev_groupAverage = helperFitFuncs().groupAverageData(
    #     data_dict_diffFlux=data_dict_diffFlux,
    #     pitchVal=ptchVal,
    #     GenToggles=GenToggles,
    #     primaryBeamToggles=primaryBeamToggles)

    # --- prepare the output data_dict ---
    # data_dict = {'Te': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Te'}],
    #              'n': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm!A-3!N', 'LABLAXIS': 'ne'}],
    #              'V0': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'V0'}],
    #              'kappa': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'kappa'}],
    #              'ChiSquare': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'ChiSquare'}],
    #              'dataIdxs': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'Index'}], # indicies corresponding to the NON-ENERGY-REDUCED dataslice so as to re-construct the fit
    #              'timestamp_fitData': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': 'cm^-2 str^-1 s^-1 ev^-1', 'LABLAXIS': 'V0'}],
    #              'Fitted_Pitch_Angles': [data_dict_diffFlux['Pitch_Angle'][0][np.array(primaryBeamToggles.wPitchsToFit)], {'DEPEND_0': None, 'UNITS': 'deg', 'LABLAXIS': 'Fitted Pitch Angle'}],
    #              'numFittedPoints': [[[] for i in range(len(primaryBeamToggles.wPitchsToFit))], {'DEPEND_0': None, 'UNITS': None, 'LABLAXIS': 'Number of Fitted Points'}],
    #              }



    ##################################
    # --------------------------------
    # --- LOOP THROUGH DATA TO FIT ---
    # --------------------------------
    ##################################
    noiseData = helperFitFuncs().generateNoiseLevel(data_dict_diffFlux['Energy'][0],primaryBeamToggles)

    for ptchIdx, pitchVal in enumerate(primaryBeamToggles.wPitchsToFit):

        EpochFitData, fitData,fitData_stdDev = helperFitFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                                                            pitchVal=pitchVal,
                                                                                                            GenToggles= GenToggles,
                                                                                                            primaryBeamToggles=primaryBeamToggles)
        ####################################
        # --- Calculate Secondary Curves ---
        ####################################


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