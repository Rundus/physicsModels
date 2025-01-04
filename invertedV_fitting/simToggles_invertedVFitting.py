# --- imports ---
import numpy as np
from spaceToolsLib.variables import m_to_km
from datetime import datetime

######################
# ---GENERAL SETUP ---
######################
class GenToggles:
    wFlyerFit = 0
    input_diffNFiles = ['C:\Data\ACESII\L2\high\ACESII_36359_l2_eepaa_fullCal.cdf',
                        'C:\Data\ACESII\L2\high\ACESII_36364_l2_eepaa_fullCal.cdf']
    wRegion = 1  # pick the region below to use in the inverted-V times
    # should do this in terms of ILat!
    invertedV_times = [[datetime(2022, 11, 20, 17, 25, 1, 162210), datetime(2022, 11, 20, 17, 25, 2, 962215)],
                       # Dispersive Region
                       [datetime(2022, 11, 20, 17, 25, 23, 762000), datetime(2022, 11, 20, 17, 25, 46, 612000)]
                       # Primary inverted-V
                       ]

##########################
# --- PRIMARY BEAM FIT ---
##########################
class primaryBeamToggles:

    invertedV_fitDensityTempPotential = True
    PlotIndividualFits = False
    outputStatisticsPlot = False

    # --- controlling the noise floor ---
    countNoiseLevel = 4

    # --- accelerating potential toggles ---
    engy_Thresh = 140  # minimum allowable energy of the inverted-V potential

    # --- Levenberg-Marquart Fit toggles ---
    wPitchsToFit = [2]
    wDistributionToFit = 'Kappa' # 'Maxwellian' or 'Kappa'

    # -- Fit Statistics Toggles ---
    nPoints_Thresh = 3  # Number of y-points that are needed in order to fit the data
    chiSquare_ThreshRange = [0.1, 100]  # range that the ChiSquare must fall into in order to be counted













