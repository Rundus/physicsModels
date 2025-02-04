# --- imports ---
import numpy as np
from spaceToolsLib.variables import m_to_km
from datetime import datetime

######################
# ---GENERAL SETUP ---
######################
class GenToggles:
    target_Latitude = 70  # used to pull from the IRI model
    target_Longitude = 16  # used to pull from the IRI model
    target_time = datetime(2022, 11, 20, 17, 20)
    # target_time = datetime(2012, 3, 30, 12, 00)

    simAltLow = 50*m_to_km # low altitude (in meters)
    simAltHigh = 1000*m_to_km # high altitude (in meters)
    obsHeight = 400*m_to_km # height of observation (in meters)
    alt_Rez = 2000 # number of points in the altitude grid

    # calculatd quantities
    simAlt = np.linspace(simAltLow, simAltHigh, alt_Rez)  # in METERS

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
    outputFolder = 'C:\Data\physicsModels\ionosphere\geomagneticField'

##########################
# --- NEUTRALS TOGGLES ---
##########################
class neutralsToggles:
    outputFolder = r'C:\Data\physicsModels\ionosphere\neutralEnvironment'
    NRLMSIS_filePath = r'C:\Data\physicsModels\ionosphere\NRLMSIS\ACESII\NRLMSIS2.0.3D.2022324.nc'
    wNeutrals = ['N2','O2','O'] # which neutrals to consider in the simulation, use the key format in spacetoolsLib



########################
# --- PLASMA DENSITY ---
########################
class plasmaToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\plasmaEnvironment'

    # --- --- --- ---
    ### ELECTRONS ###
    # --- --- --- ---
    useIRI_Te_Profile = True
    useIRI_ne_Profile = True

    useStatic_ne_Profile = False
    staticDensityVal = 15 * (100 ** 3)

    # --- --- --
    ### IONS ###
    # --- --- --
    useIRI_Ti_Profile = True
    useIRI_ni_Profile = True
    IRI_filePath = r'C:\Data\physicsModels\ionosphere\IRI\ACESII\IRI_3D_2022324.cdf'
    # IRI_filePath = r'C:\Data\physicsModels\ionosphere\IRI\Leda2019'

    wIons = ['NO+','O+','O2+']  # which neutrals to consider in the simulation, use the key format in spaceToolsLib
    # wIons = ['NO+','H+','N+','He+', 'O+', 'O2+']  # all ions



######################
# --- CONDUCTIVITY ---
######################
class conductivityToggles:
    useRealData = False # if True, uses the data/toggles from the /invertedV_fitting folder
    outputFolder = 'C:\Data\physicsModels\ionosphere\conductivity'



###########################
# --- HEIGHT IONIZATION ---
###########################
class ionizationRecombToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\ionizationRecomb'








