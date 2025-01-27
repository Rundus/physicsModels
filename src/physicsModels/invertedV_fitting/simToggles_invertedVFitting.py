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

    # denote where the data should be taken from
    inputDataPath = GenToggles.input_diffNFiles[GenToggles.wFlyerFit]
    outputFolder = 'C:\Data\physicsModels\invertedV\primaryBeam_Fitting'
    targetTimes = [GenToggles.invertedV_times[GenToggles.wRegion][0],
                   GenToggles.invertedV_times[GenToggles.wRegion][1]]

    # --- controlling the noise floor ---
    countNoiseLevel = 2

    # --- accelerating potential toggles ---
    engy_Thresh = 100  # minimum allowable energy of the inverted-V potential
    maxfev = int(1E4) # number of iterations the LM fit is allowed
    useNoGuess = True # use an initial guess?

    # --- Levenberg-Marquart Fit toggles ---
    wPitchsToFit = [10, 20, 30, 40, 50, 60, 70, 80, 90] # give pitch angles in degrees
    wDistributionToFit = 'Kappa' # 'Maxwellian' or 'Kappa'
    numToAverageOver = 5 # HOW many datapoints are averaged together when fitting

    # Determine guesses for the fitted data
    V0_deviation = 0.18
    n_bounds = [0.001, 10]  # n [cm^-3]
    Te_bounds = [10, 300]
    kappa_bounds = [1.5, 101]

    if wDistributionToFit == 'Maxwellian':
        n_guess = 1
        T_guess = 100
        # can't do V0 guess, that's generated in the code itself
    elif wDistributionToFit == 'Kappa':
        n_guess = 1
        T_guess = 100
        kappa_guess = 20

    # --- fit refinement ---
    useFitRefinement = True
    beta_guess = 12 # altitude of the inverted-V
    n0guess_deviation = 0.99

class primaryBeamPlottingToggles:

    # -- Fit Statistics Toggles ---
    chiSquare_ThreshRange = [0.1, 100]  # range that the ChiSquare must fall into in order to be counted

class backScatterToggles:

    outputFolder = r'C:\Data\physicsModels\invertedV\backScatter'

    # --- ENERGY GRID ---
    N_energyGrid = 50
    model_energyGrid = np.linspace(10, 4E3, N_energyGrid)

    # --- model parameters ---
    modelParametersPitchAngle = 10 # scalar [degrees] - which pitch angle to use for the "primary beam"

    # --- Calculating backScatter ---
    betaChoice = 6 # which beta value to pick i.e. the height above the rocket of the invertedV
    niterations_backscatter = 4  # number of iterations for the secondaries calculations. >19 iterations is TOO many












