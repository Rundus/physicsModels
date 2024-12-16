# --- imports ---
import numpy as np
from numpy import linspace
from numpy import array


# TODO: PITCH ANGLE CALCULATION IS WRONG:
#  If you make a strong pulse that causes all particles to move downward, you have equal amounts of -180 and 180 particles.
#  But everything should be 0deg!

# --- USEFUL CONSTANTS ---
m_to_km = 1E3
R_REF = 6371.2 * m_to_km  # in meters


######################
# --- POWER SWITCH ---
######################
runFullSimulation = True # MUST BE  == TRUE TO RUN THE SIMULATION. Set this == True then run TestPArticle_simulation


######################
# ---GENERAL SETUP ---
######################
class GenToggles:
    simLen = 550 # how many delta T steps to simulate
    deltaT = 0.01 # in seconds
    simAltLow = 200*m_to_km # low altitude (in meters)
    simAltHigh = 10000*m_to_km # high altitude (in meters)
    obsHeight = 400*m_to_km # height of observation (in meters)
    alt_Rez = 2000 # number of points in the altitude grid

    # calculatd quantities
    simAlt = linspace(simAltLow, simAltHigh, alt_Rez)  # in METERS
    simTime = linspace(0, deltaT * simLen, simLen + 1)  # in seconds

    # extra
    fps = 10
    simColors = ['tab:purple', 'tab:orange', 'tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink']  # the color choices available for the simulation to use
    simFolderPath = r'C:\Users\cfelt\PycharmProjects\UIOWA_CDF_operator\ACESII_code\Science\AlfvenSingatureAnalysis\Simulations\TestParticle'
    simOutputPath = r'C:\Data\ACESII\science\simulations\TestParticle'


################################
# --- PARTICLE DISTRIBUTIONS ---
################################
class ptclToggles:
    seedChoice = 10 # some value to define the randomness seed
    ptclTemperature = 5 # distribution temperature in eV
    Z0_ptcl_ranges = array([0.5, 0.55, 0.6]) * m_to_km * 6371 # initial altitude of particles (in meters)
    N_ptcls = 500  # number of particles. Example: The real data at s3 has 10598 particles
    ptcl_mass = 9.11 * 10 ** (-31)  # mass of the particle
    ptcl_charge = 1.602176565 * 10 ** (-19)  # charge of the particle
    simEnergyRanges = [[0.01, 1], [1, 5], [5, 10], [10, 25], [25, 50], [50, 60]]  # The range of energies for each color (the rightmost linspace point is ignored)
    # calculated quantities
    totalNumberOfParticles = N_ptcls * len(Z0_ptcl_ranges)

###########################
# --- GEOMAGNETIC FIELD ---
###########################
class BgeoToggles:
    Lshell = 8.7
    ConstantBval = None # in tesla. Set == None to NOT use a constant Bval

########################
# --- ELECTRIC FIELD ---
########################
class EToggles:
    Z0_wave = (11000*m_to_km) # initial altitude of the wave (in meters)
    lambdaPerp0 = 3.2 * m_to_km  # lambdaPerp AT the Ionosphere (in meters)
    waveFreq_Hz = 4 # in Hz
    Eperp0 = 0.02  # V/m
    waveFraction = 2 # What fraction of the initial bipolar wave we want to keep. e.g. 2 --> Half the wave, 3 --> 1/3 of wave etc
    lambdaPerp_Rez = 11 # resolution of the x-direction (MUST BE ODD)

    # calculated quantities
    tau0 = 1 / waveFreq_Hz  # risetime of pulse (in seconds)
    kperp0 = 2 * np.pi / lambdaPerp0  # this is Kperp AT THE OBSERVATION POINT
    waveFreq_rad = 2 * np.pi * waveFreq_Hz

    # toggles
    static_Kperp = False
    flipEField = True
    staticDensity = False
    staticDensityVal = 15*(100**3)

