# --- ionoNeutralEnvironment_Generator ---
# get the NRLMSIS data and export the neutral temperature vs altitude at ACES-II times.
# Also, interpolate each variable in order to sample the data at my model cadence


# --- imports ---
import spaceToolsLib as stl
from numpy import datetime64,squeeze
import pymsis
import numpy as np
from copy import deepcopy
from src.physicsModels.ionosphere.neutral_environment.neutral_toggles import neutralsToggles
from src.physicsModels.ionosphere.spatial_environment.spatial_toggles import SpatialToggles


def generateNeutralEnvironment(**kwargs):

    #######################
    # --- LOAD THE DATA ---
    #######################
    # get the geomagnetic field data dict
    data_dict_spatial = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\spatial_environment\spatial_environment.cdf')

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################
    LShellRange = data_dict_spatial['simLShell'][0]
    altRange = data_dict_spatial['simAlt'][0]

    data_dict_output = {
        'rho_n': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'N2': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'O2': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'O': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'HE': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'H': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'AR': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'N': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'ANOMALOUS_O': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'NO': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'Tn': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'm_eff_n': [np.zeros(shape=(len(LShellRange), len(altRange))), {}],
        'simLShell': deepcopy(data_dict_spatial['simLShell']),
        'simAlt': deepcopy(data_dict_spatial['simAlt']),
    }

    ##############################
    # --- GET THE NRLMSIS DATA ---
    ##############################
    f107 = 150  # the F10.7 (DON'T CHANGE)
    f107a = 150  # ap data (DON't CHANGE)
    ap = 7
    aps = [[ap] * 7]
    dt_targetTime = SpatialToggles.target_time
    date = datetime64(f"{dt_targetTime.year}-{dt_targetTime.month}-{dt_targetTime.day}T{dt_targetTime.hour}:{dt_targetTime.minute}")

    for idx1, Lval in enumerate(LShellRange):
        for idx2, altVal in enumerate(altRange):

            lon = data_dict_spatial['grid_long'][0][idx1][idx2]
            lat = data_dict_spatial['grid_lat'][0][idx1][idx2]
            alts = data_dict_spatial['grid_alt'][0][idx1][idx2] / stl.m_to_km

            #  output is of the shape (1, 1, 1, 1000, 11), use squeeze to Get rid of the single dimensions
            NRLMSIS_data = squeeze(pymsis.calculate(date, lon, lat, alts, f107, f107a, aps))

            for var in pymsis.Variable:

                dat = NRLMSIS_data[var]
                if dat < 1E-25:
                    varData = 0
                else:
                    varData = dat


                if var.name =='MASS_DENSITY':
                    varunits = 'kg m!A-3!N'
                    varname = 'rho_n'

                elif var.name =='TEMPERATURE':
                    varunits = 'K'
                    varname = 'Tn'

                else:
                    varunits = 'm!A-3!N'
                    varname = var.name

                data_dict_output[f'{varname}'][0][idx1][idx2] = varData

                if idx1 == 0:
                    data_dict_output[f'{varname}'][1] = {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': varunits, 'LABLAXIS': f'{varname}', 'VAR_TYPE':'data'}


    # add the total neutral density
    n_n = np.array([data_dict_output[f"{key}"][0] for key in neutralsToggles.wNeutrals])
    data_dict_output = {**data_dict_output, **{'nn': [np.sum(n_n,axis=0), {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'nn', 'VAR_TYPE':'data'}]}}

    # add the effective neutral mass
    m_eff_n = np.sum(np.array( [stl.netural_dict[key]*data_dict_output[f"{key}"][0] for key in neutralsToggles.wNeutrals]), axis=0)/data_dict_output['nn'][0]
    data_dict_output = {**data_dict_output, **{'m_eff_n': [m_eff_n, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'nn', 'VAR_TYPE':'data'}]}}

    #####################
    # --- OUTPUT DATA ---
    #####################

    outputPath = rf'{neutralsToggles.outputFolder}\neutral_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
