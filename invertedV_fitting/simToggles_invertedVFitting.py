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
                        ]
    input_attitudeFiles = [
        r'C:\Data\ACESII\attitude\high\ACESII_36359_Attitude_Solution.cdf'
    ]


    if wFlyerFit == 0: # ACES-II Data
        wRegion = 2  # pick the region below to use in the inverted-V times
        invertedV_times = [
                            [datetime(2022, 11, 20, 17, 24, 12, 162000), datetime(2022, 11, 20, 17, 24, 18, 812000)], # Very First ,Inverted-V, the high energy one
                            [datetime(2022, 11, 20, 17, 24, 45, 862000), datetime(2022, 11, 20, 17, 24, 49, 312000)], # small inverted-V, after the High energy One
                            [datetime(2022, 11, 20, 17, 25, 23, 762000), datetime(2022, 11, 20, 17, 26, 8, 212000)],  # Primary inverted-V
                            [datetime(2022, 11, 20, 17, 26, 11, 412000), datetime(2022, 11, 20, 17, 26, 19, 912000)],  # Inverted-V right after the Primary-V, has STEBs on either sides of it
                            [datetime(2022, 11, 20, 17, 26, 35, 112000), datetime(2022, 11, 20, 17, 26, 40, 712000)], # Inverted-V two after the Primary-V
                            [datetime(2022, 11, 20, 17, 28, 17, 112000), datetime(2022, 11, 20, 17, 28, 34, 612000)] # Faint inverted-V on the most northside of the flight
                           ]
    elif wFlyerFit == 1: # ACES-I Data
        wRegion = 0  # pick the region below to use in the inverted-V times
        invertedV_times = [
                            [datetime(2009, 1, 29, 9, 54, 4, 0), datetime(2009, 1, 29, 9, 54, 29, 000)], # Very First ,Inverted-V, the high energy one
                            ]

##########################
# --- PRIMARY BEAM FIT ---
##########################
class primaryBeamToggles:

    # --- controlling the noise floor ---
    countNoiseLevel = 2

    # --- accelerating potential toggles ---
    engy_Thresh = 100  # minimum allowable energy of the inverted-V potential
    maxfev = int(1E4) # number of iterations the LM fit is allowed
    useNoGuess = True # use an initial guess?

    # --- Levenberg-Marquart Fit toggles ---
    wPitchsToFit = [2]
    wDistributionToFit = 'Kappa' # 'Maxwellian' or 'Kappa'
    numToAverageOver = 5 # HOW many datapoints are averaged together when fitting

    # Determine guesses for the fitted data
    V0_deviation = 0.18
    n_bounds = [0.001, 3]  # n [cm^-3]
    Te_bounds =  [10, 500]
    kappa_bounds = [1.5, 30]

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

class primaryBeamPlottingToggles:

    makeIndividualFitPlots = True
    makeFitParameterPlot = True

    # -- Fit Statistics Toggles ---
    chiSquare_ThreshRange = [0.1, 100]  # range that the ChiSquare must fall into in order to be counted

class secondaryBackScatterToggles:

    # z_high_flyer = 400
    # z_iono_cutoff = 300
    # alpha_cutoff = 80

    N_energyGrid = 1000 # number of points in the energy grid which covers the beam+backscatter/secondaries
    Niterations_secondaries = 10 # number of iterations for the secondaries calculations. >19 iterations is TOO many
    Niterations_backscatter = 10  # number of iterations for the secondaries calculations.












