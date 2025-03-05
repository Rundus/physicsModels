# --- imports ---
import numpy as np
from spaceToolsLib.variables import m_to_km
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from datetime import datetime

######################
# ---GENERAL SETUP ---
######################
class GenToggles:
    fps = 10
    simColors = ['tab:purple', 'tab:orange', 'tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink']  # the color choices available for the simulation to use
    simFolderPath = r'C:\Data\physicsModels\ionosphere'



class SpatialToggles:

    # target_Latitude = 70  # used to pull from the IRI model
    # target_Longitude = 16  # used to pull from the IRI model
    target_time = datetime(2022, 11, 20, 17, 20)  # used to pull from the IRI model

    # Altitude Grid
    sim_alt_low = 50 * m_to_km  # low altitude (in meters)
    sim_alt_high = 400 * m_to_km  # high altitude (in meters)
    alt_rez = 5 * m_to_km  # number of points in the altitude grid
    simAlt = np.linspace(sim_alt_low, sim_alt_high, int((sim_alt_high - sim_alt_low) / alt_rez + 1))  # in METERS

    # ILat Grid
    sim_ILat_Low = 70
    sim_ILat_High = 73.5
    ilat_rez = 0.002 # there are 8659 records between 70ILat to 73.5 ILat on the HF. Choose an appropriate resolution.
    simILat = np.linspace(sim_ILat_Low, sim_ILat_High, int((sim_ILat_High - sim_ILat_Low) / ilat_rez + 1))  # in Degrees


###########################
# --- GEOMAGNETIC FIELD ---
###########################
class BgeoToggles:
    Lshell = 8.7
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

    wIons = ['NO+', 'O+', 'O2+']  # which neutrals to consider in the simulation, use the key format in spaceToolsLib
    # wIons = ['NO+','H+','N+','He+', 'O+', 'O2+']  # all ions



###########################
# --- HEIGHT IONIZATION ---
###########################
class ionizationRecombToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\ionizationRecomb'

    # --- BEAM n_E: which dataset to use for the n_e profile ---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_evans1974_beam = False
    use_eepaa_beam = False  # if True, uses the n_e profile derived from the High Flyer Data.


######################
# --- CONDUCTIVITY ---
######################
class conductivityToggles:

    outputFolder = 'C:\Data\physicsModels\ionosphere\conductivity'

    # --- BEAM n_E: which dataset to use for the n_e profile ---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_evans1974_beam = False
    use_eepaa_beam = False # if True, uses the n_e profile derived from the High Flyer Data.

    # --- BACKGROUND n_e: Which dataset to use for the n_e profile---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_IRI_background = False
    use_eepaa_background = False  # if True, uses the n_e profile derived from the High Flyer Data.











