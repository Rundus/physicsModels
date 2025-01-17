# --- IRI_ACESII_SliceSampling.py ---
# --- Author: C. Feltman ---
# DESCRIPTION:



# --- bookkeeping ---
# !/usr/bin/env python
__author__ = "Connor Feltman"
__date__ = "2022-08-22"
__version__ = "1.0.0"

import numpy as np

from ACESII_code.myImports import *
start_time = time.time()
# --- --- --- --- ---


# --- --- --- ---
# --- TOGGLES ---
# --- --- --- ---
justPrintFileNames = False
wRocket = 4
modelFilePath = r"C:\Users\cfelt\PycharmProjects\UIOWA_CDF_operator\ACESII_code\supportCode\IonosphereModels\IRI\CDF\IRI_3D_2022324.cdf"
# ---------------------------
smoothData = True
widthSize = 800

# ---------------------------
outputData = True
# ---------------------------

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from astropy.convolution import convolve, Box1DKernel


def IRI_ACESII_SliceSampling(wRocket, rocketFolderPath, justPrintFileNames):

    # --- ACES II Flight/Integration Data ---
    rocketAttrs, b, c = ACES_mission_dicts()
    rocketID = rocketAttrs.rocketID[wRocket-4]
    globalAttrsMod = rocketAttrs.globalAttributes[wRocket-4]
    globalAttrsMod['Logical_source'] = globalAttrsMod['Logical_source'] + 'L2'
    ModelData = L2_TRICE_Quick(wRocket-4)

    # Load the attitude Data
    inputFiles = glob(f'{rocketFolderPath}\\attitude\\{fliers[wRocket-4]}\\*.cdf')
    input_names = [ifile.replace(f'{rocketFolderPath}\\attitude\{fliers[wRocket-4]}\\', '') for ifile in inputFiles]

    if justPrintFileNames:
        for i, file in enumerate(inputFiles):
            print('[{:.0f}] {:80s}{:5.1f} MB '.format(i, input_names[i], round(getsize(file) / (10 ** 6), 1)))
        return

    print('\n')

    # --- --- --- --- --- -
    # --- LOAD THE DATA ---
    # --- --- --- --- --- -


    # --- get the data from the file ---
    prgMsg(f'Loading data from Attitude Files')
    data_dict_attitude = loadDictFromFile(inputFiles[0])
    Done(start_time)

    # Convert attitude time to minutes in the day
    Epoch_attitude_minutes = (np.array(dateTimetoTT2000(InputEpoch=data_dict_attitude['Epoch'][0],inverse=False)) - pycdf.lib.datetime_to_tt2000(dt.datetime(2022,11,20,00,00,000)))/(1E9*60)

    for i in range(len(Epoch_attitude_minutes)-1):

        val = Epoch_attitude_minutes[i+1] - Epoch_attitude_minutes[i]
        if val <= 0:
            print(i, Epoch_attitude_minutes[i+1], Epoch_attitude_minutes[i])


    prgMsg(f'Loading data from IRI File')
    data_dict_IRI = loadDictFromFile(modelFilePath)
    Done(start_time)

    # --- Loop through attitude data and find position indicies of time,lat,long and then sample/store them ---
    # Sample these Variables from the IRI data dict
    # NOTE: Dimensions of IRI data are (Time, Height, lattitude, longitutde)
    sampleTheseVars = ['Ne', 'O+','H+','He+','O2+','NO+','N+','Tn','Ti','Te']
    data_dict_output ={varNam:[[],{}] for varNam in sampleTheseVars}

    prgMsg('down-Sampling IRI data to attitude')
    for i in range(len(data_dict_attitude['Epoch'][0])):


        # get the attitude indicies for this specific time
        indexTime = np.abs(data_dict_IRI['time'][0] - Epoch_attitude_minutes[i]).argmin()
        indexAlt = np.abs(data_dict_IRI['ht'][0] - (np.array(data_dict_attitude['Alt'][0][i]) / 1000)).argmin()
        indexLat = np.abs(data_dict_IRI['lat'][0] - data_dict_attitude['Lat'][0][i]).argmin()
        indexLong =np.abs(data_dict_IRI['lon'][0]-data_dict_attitude['Long'][0][i]).argmin()

        # append all the data in IRI to respective data in data_dict_output
        for varNam in sampleTheseVars:
            data_dict_output[varNam][0].append(
                data_dict_IRI[varNam][0][indexTime][indexAlt][indexLat][indexLong]
            )

    # update data_dict_output's variable Attributes and smooth data
    for varNam in sampleTheseVars:
        newAttrs = deepcopy(data_dict_IRI[varNam][1])
        newAttrs['DEPEND_0'] = 'Epoch'
        newAttrs['DEPEND_1'] = None
        newAttrs['DEPEND_2'] = None
        newAttrs['DEPEND_3'] = None

        if 'cm' in newAttrs['UNITS']:
            newAttrs['UNITS'] = '!Ncm!A-3!N'
        data_dict_output[varNam][1] = newAttrs

        # convert data to numpy arrays and smooth data
        if smoothData:
            data_dict_output[varNam][0] = np.array(convolve(data_dict_output[varNam][0], Box1DKernel(widthSize)))
        else:
            data_dict_output[varNam][0] = np.array(data_dict_output[varNam][0])


    # include attitude solution's lat,long,alt, geomlat,geomlong,geomalt information into the output
    varNam_attitude = ['Epoch','Lat', 'Lat_geom', 'Alt', 'Alt_geom','Long','Long_geom','invarLat']
    for varNam in varNam_attitude:

        data_dict_output = {**data_dict_output,**{varNam:deepcopy(data_dict_attitude[varNam])}}

    Done(start_time)

    ExampleVarAttrs = {'FIELDNAM': None,
                       'LABLAXIS': None,
                         'DEPEND_0': None,
                         'DEPEND_1': None,
                         'DEPEND_2': None,
                         'FILLVAL': None,
                         'FORMAT': None,
                         'UNITS': None,
                         'VALIDMIN': None,
                         'VALIDMAX': None,
                         'VAR_TYPE': 'data',
                         'SCALETYP': 'linear'}

    # create the total ion number density variable
    ionNames = ['O+', 'H+', 'He+', 'O2+', 'NO+', 'N+']
    n_i = np.zeros(shape=len(data_dict_attitude['Epoch'][0]))
    for varNam in ionNames:
        n_i += np.array(data_dict_output[varNam][0])

    n_i_attrs = deepcopy(ExampleVarAttrs)
    n_i_attrs['FIELDNAM'] = 'n_i'
    n_i_attrs['LABLAXIS'] = 'n_i'
    n_i_attrs['UNITS'] = 'cm^-3'
    data_dict_output = {**data_dict_output,**{'n_i': [n_i,n_i_attrs]}}

    # create the total charged particle mass density variable
    # NOTE: Format (e-,O+, H+, He+, O2+, NO+, N+)
    chargedParticleNames =  ['Ne', 'O+', 'H+', 'He+', 'O2+', 'NO+', 'N+']
    particleMasses = [9.11E-31, 2.6567E-26, 1.6738E-27, 6.646477E-27, 5.3134E-26, 4.9826E-26, 2.3259E-26]
    rho = []
    for i in range(len(data_dict_attitude['Epoch'][0])):
        rho.append(sum([ particleMasses[j]*data_dict_output[varNam][0][i] for j,varNam in enumerate(chargedParticleNames)]))

    rho = np.array(rho)
    rho_attrs = deepcopy(ExampleVarAttrs)
    rho_attrs['FIELDNAM'] = 'rho'
    rho_attrs['LABLAXIS'] = 'rho'
    rho_attrs['UNITS'] = 'kg cm^-3'
    data_dict_output = {**data_dict_output, **{'rho': [rho, rho_attrs]}}

    # rename electron number density variable
    data_dict_output['n_e'] = data_dict_output.pop('Ne')
    data_dict_output['n_e'][1]['LABLAXIS'] = 'n_e'
    data_dict_output['n_e'][1]['FIELDNAM'] = 'n_e'

    # calculate the average ion mass over the flight
    m_i_avg = np.array([rho[f]/n_i[f] for f in range(len(rho)) ])

    m_i_avg_attrs = deepcopy(ExampleVarAttrs)
    m_i_avg_attrs['FIELDNAM'] = 'm_i_avg'
    m_i_avg_attrs['LABLAXIS'] = 'm_i_avg'
    m_i_avg_attrs['UNITS'] = 'kg'
    data_dict_output = {**data_dict_output, **{'m_i_avg': [m_i_avg, m_i_avg_attrs]}}



    # --- --- --- --- --- --- ---
    # --- WRITE OUT THE DATA ---
    # --- --- --- --- --- --- ---

    if outputData:
        prgMsg('Creating output file')

        fileoutName = f'ACESII_{rocketID}_IRI_smoothed.cdf' if smoothData else f'ACESII_{rocketID}_IRI_slice.cdf'
        outputPath = f'{rocketFolderPath}\\science\\Ineternational_Reference_Ionosphere_ACESII_Slice\\{fliers[wRocket-4]}\\{fileoutName}'
        outputCDFdata(outputPath, data_dict_output, ModelData, globalAttrsMod, 'attitude')
        Done(start_time)





# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---
if wRocket == 4:  # ACES II High
    rocketFolderPath = ACES_data_folder
elif wRocket == 5: # ACES II Low
    rocketFolderPath = ACES_data_folder

IRI_ACESII_SliceSampling(wRocket, rocketFolderPath, justPrintFileNames)
