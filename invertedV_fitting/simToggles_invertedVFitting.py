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
                        'C:\Data\ACESII\L2\low\ACESII_36364_l2_eepaa_fullCal.cdf']
    wRegion = 2  # pick the region below to use in the inverted-V times
    invertedV_times = [
                        [datetime(2022, 11, 20, 17, 24, 12, 162000), datetime(2022, 11, 20, 17, 24, 18, 812000)], # Very First ,Inverted-V, the high energy one
                        [datetime(2022, 11, 20, 17, 24, 45, 862000), datetime(2022, 11, 20, 17, 24, 49, 312000)], # small inverted-V, after the High energy One
                        [datetime(2022, 11, 20, 17, 25, 23, 762000), datetime(2022, 11, 20, 17, 26, 8, 212000)],  # Primary inverted-V
                        [datetime(2022, 11, 20, 17, 26, 11, 412000), datetime(2022, 11, 20, 17, 26, 19, 912000)],  # Inverted-V right after the Primary-V, has STEBs on either sides of it
                        [datetime(2022, 11, 20, 17, 26, 35, 112000), datetime(2022, 11, 20, 17, 26, 40, 712000)], # Inverted-V two after the Primary-V
                        [datetime(2022, 11, 20, 17, 28, 17, 112000), datetime(2022, 11, 20, 17, 28, 34, 612000)] # Faint inverted-V on the most northside of the flight
                       ]

##########################
# --- PRIMARY BEAM FIT ---
##########################
class primaryBeamToggles:

    invertedV_fitDensityTempPotential = True
    PlotIndividualFits = False
    outputStatisticsPlot = False

    # --- controlling the noise floor ---
    countNoiseLevel = 3

    # --- accelerating potential toggles ---
    engy_Thresh = 110  # minimum allowable energy of the inverted-V potential

    # --- Levenberg-Marquart Fit toggles ---
    wPitchsToFit = [2, 3]
    wDistributionToFit = 'Kappa' # 'Maxwellian' or 'Kappa'
    numToAverageOver = 5 # HOW many datapoints are averaged together when fitting

    # Determine guesses for the fitted data
    V0_deviation = 0.18
    n_bounds = [0.001,3]  # n [cm^-3]
    Te_bounds =  [10, 500]
    kappa_bounds = [1.5,30]

    if wDistributionToFit == 'Maxwellian':
        n_guess = 1
        T_guess = 300
        # can't do V0 guess, that's generated in the code itself
    elif wDistributionToFit == 'Kappa':
        n_guess = 1
        T_guess = 300
        kappa_guess = 20

    # --- fit refinement ---
    beta_guess = 6 # altitude of the inverted-V
    n0guess_deviation = 0.8

    # -- Fit Statistics Toggles ---
    chiSquare_ThreshRange = [0.1, 100]  # range that the ChiSquare must fall into in order to be counted













