# --- imports ---
import numpy as np
from numpy import linspace
from numpy import array
from spaceToolsLib.variables import m_to_km


# TODO: PITCH ANGLE CALCULATION IS WRONG:
#  If you make a strong pulse that causes all particles to move downward, you have equal amounts of -180 and 180 particles.
#  But everything should be 0deg!

######################
# --- POWER SWITCH ---
######################
runFullSimulation = True # MUST BE  == TRUE TO RUN THE SIMULATION. Set this == True then run executable

######################
# ---GENERAL SETUP ---
######################
class GenToggles:
    simAltLow = 200*m_to_km # low altitude (in meters)
    simAltHigh = 1000*m_to_km # high altitude (in meters)
    obsHeight = 400*m_to_km # height of observation (in meters)
    alt_Rez = 2000 # number of points in the altitude grid

    # calculatd quantities
    simAlt = linspace(simAltLow, simAltHigh, alt_Rez)  # in METERS

    # extra
    fps = 10
    simColors = ['tab:purple', 'tab:orange', 'tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink']  # the color choices available for the simulation to use
    simFolderPath = r'C:\Data\physicsModels\ionosphere'

###########################
# --- GEOMAGNETIC FIELD ---
###########################
class BgeoToggles:
    Lshell = 8.7
    useConstantBval = False
    ConstantBval = (10000E-9) # in tesla

########################
# --- PLASMA DENSITY ---
########################
class plasmaDensity:
    useTanakaDensity = False
    useKletzingS33Density = True
    useChastonDensity = True



