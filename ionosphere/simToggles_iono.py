# --- imports ---
import numpy as np
from numpy import linspace
from numpy import array
from spaceToolsLib.variables import m_to_km
from datetime import datetime

######################
# ---GENERAL SETUP ---
######################
class GenToggles:
    target_Latitude = 70  # used to pull from the IRI model
    target_Longitude = 16  # used to pull from the IRI model
    target_time = datetime(2022, 11, 20, 17, 20)

    simAltLow = 100*m_to_km # low altitude (in meters)
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
    ConstantBval = 50000E-9 # in tesla. Set == None to NOT use a constant Bval

##########################
# --- NEUTRALS TOGGLES ---
##########################
class neutralsToggles:
    NRLMSIS_filePath = r'C:\Data\physicsModels\ionosphere\NRLMSIS\ACESII\NRLMSIS2.0.3D.2022324.nc'




########################
# --- PLASMA DENSITY ---
########################
class plasmaToggles:

    useIRI = True

    # --- --- --- ---
    ### ELECTRONS ###
    # --- --- --- ---
    # Temperature
    useIRI_Te_Profile = False if not useIRI else True
    useSchroeder_Te_Profile = False

    # Density
    useIRI_ne_Profile = False if not useIRI else True
    useTanaka_ne_Profile = False
    useKletzingS33_ne_Profile = False
    useChaston_ne_Profile = False
    useStatic_ne_Profile = False
    staticDensityVal = 15 * (100 ** 3)

    # --- --- --
    ### IONS ###
    # --- --- --
    # temperature
    useIRI_Ti_Profile = False if not useIRI else True

    # density
    useIRI_ni_Profile = False if not useIRI else True

    IRI_filePath = r'C:\Data\physicsModels\ionosphere\IRI\ACESII\IRI_3D_2022324.cdf'









