# --- L2_to_L3_downsampled_data.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Take the ACESII EEPAA data and average it together by "N" datapoints

# --- bookkeeping ---
# !/usr/bin/env python
__author__ = "Connor Feltman"
__date__ = "2022-08-22"
__version__ = "1.0.0"
import time
start_time = time.time()
# --- --- --- --- ---

# --- --- --- ---
# --- TOGGLES ---
# --- --- --- ---

# --- Select the Rocket ---
# 4 -> ACES II High Flier
# 5 -> ACES II Low Flier
wRocket = [4, 5]

# --- OutputData ---
outputData = True

N_avg = 3

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
import spaceToolsLib as stl


#######################
# --- MAIN FUNCTION ---
#######################
def L2_to_L3_downsampled_data(wRocket):


    # --- Load the Data ---
    stl.prgMsg(f'Loading data')
    data_dict_eepaa = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\L2\{ACESII.fliers[wRocket-4]}\*eepaa_fullCal.cdf*')[0])
    data_dict_Lshell = stl.loadDictFromFile(glob(rf'C:\Data\ACESII\coordinates\Lshell\{ACESII.fliers[wRocket-4]}\*_Lshell.cdf*')[0])
    stl.Done(start_time)

    # --- prepare the output ---
    data_dict_output = {}

    # --- --- --- --- --- --- ---
    # --- DOWNSAMPLE THE DATA ---
    # --- --- --- --- --- --- ---
    stl.prgMsg('Calculating Fixed ni')

    # ensure the data is divided into chunks that can be sub-divided. If not, keep drop points from the end until it can be
    low_idx, high_idx = np.abs(data_dict_diffFlux['Epoch'][0] - targetTimes[0]).argmin(), np.abs(data_dict_diffFlux['Epoch'][0] - targetTimes[1]).argmin()

    if (high_idx - low_idx) % N_avg != 0:
        high_idx -= (high_idx - low_idx) % N_avg

    # Handle the Epoch
    chunkedEpoch = np.split(data_dict_diffFlux['Epoch'][0][low_idx:high_idx],
                            round(len(data_dict_diffFlux['Epoch'][0][low_idx:high_idx]) / N_avg))
    EpochFitData = np.array([chunkedEpoch[i][int((N_avg - 1) / 2)] for i in range(len(chunkedEpoch))])
    chunkedIlat = np.split(data_dict_diffFlux['ILat'][0][low_idx:high_idx],
                           round(len(data_dict_diffFlux['ILat'][0][low_idx:high_idx]) / N_avg))
    ILatFitData = np.array([chunkedIlat[i][int((N_avg - 1) / 2)] for i in range(len(chunkedIlat))])
    chunkedAlt = np.split(data_dict_diffFlux['Alt'][0][low_idx:high_idx],
                          round(len(data_dict_diffFlux['Alt'][0][low_idx:high_idx]) / N_avg))
    AltFitData = np.array([chunkedIlat[i][int((N_avg - 1) / 2)] for i in range(len(chunkedAlt))])

    # --- handle the multi-dimenional data ---
    # create the storage variable
    detectorPitchAngles = data_dict_diffFlux['Pitch_Angle'][0]
    diffFlux_avg = np.zeros(shape=(len(EpochFitData), len(detectorPitchAngles), len(data_dict_diffFlux['Energy'][0])))
    stdDevs_avg = np.zeros(shape=(len(EpochFitData), len(detectorPitchAngles), len(data_dict_diffFlux['Energy'][0])))

    for loopIdx, pitchValue in enumerate(detectorPitchAngles):

        chunkedyData = np.split(data[low_idx:high_idx, loopIdx, :],
                                round(len(data[low_idx:high_idx, loopIdx, :]) / N_avg))
        chunkedStdDevs = np.split(data_stdDev[low_idx:high_idx, loopIdx, :],
                                  round(len(data_stdDev[low_idx:high_idx, loopIdx, :]) / N_avg))

        # --- Average the chunked data ---
        fitData = np.zeros(shape=(len(chunkedyData), len(data_dict_diffFlux['Energy'][0])))
        fitData_stdDev = np.zeros(shape=(len(chunkedStdDevs), len(data_dict_diffFlux['Energy'][0])))

        for i in range(len(chunkedEpoch)):
            # average the diffFlux data by only choosing data which is valid
            chunkedyData[i][chunkedyData[i] < 0] = np.NaN
            fitData[i] = np.nanmean(chunkedyData[i], axis=0)

            # average the diffFlux data by only choosing data which is valid
            chunkedStdDevs[i][chunkedStdDevs[i] < 0] = np.NaN
            fitData_stdDev[i] = np.nanmean(chunkedStdDevs[i], axis=0)

        diffFlux_avg[:, loopIdx, :] = fitData
        stdDevs_avg[:, loopIdx, :] = fitData_stdDev





    varAttrs = {'LABLAXIS': 'plasma density', 'DEPEND_0': 'Epoch',
                                                                   'DEPEND_1': None,
                                                                   'DEPEND_2': None,
                                                                   'FILLVAL': ACESII.epoch_fillVal,
                                                                   'FORMAT': 'E12.2',
                                                                   'UNITS': '!Ncm!A-3!N',
                                                                   'VALIDMIN': 0,
                                                                   'VALIDMAX': 0,
                                                                   'VAR_TYPE': 'data', 'SCALETYP': 'linear'}




    # --- --- --- --- --- --- ---
    # --- WRITE OUT THE DATA ---
    # --- --- --- --- --- --- ---
    if outputData:

        stl.prgMsg('Creating output file')
        fileoutName_fixed = f'ACESII_{rocketID}_langmuir_fixed.cdf'
        outputPath = f'{rocket_folder_path}{fToggles.outputPath_modifier}\{ACESII.fliers[wflyer]}\\{fileoutName_fixed}'
        stl.outputCDFdata(outputPath, data_dict_fixed, instrNam= 'Langmuir Probe',globalAttrsMod=globalAttrsMod)
        stl.Done(start_time)





# --- --- --- ---
# --- EXECUTE ---
# --- --- --- ---

for idx in wRocket:
    L2_to_L3_downsampled_data(idx)

