# --- executable_ionosphere.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: regenerate the IONOSPHERE plasma environment toggles


#################
# --- IMPORTS ---
#################
import time
from src.physicsModels.ionosphere.simToggles_Ionosphere import *
import spaceToolsLib as stl
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()


#################
# --- TOGGLES ---
#################
regenSpatial = True
regenBgeo = False
regenPlasmaEnvironment = False
regenNeutralEnvironment = False
ionRecomb_ne_Calc = False
calc_IonoConductivity = False

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################

if regenSpatial:
    # spatial environment
    stl.prgMsg('Regenerating Spatial Environment')
    from src.physicsModels.ionosphere.spatialEnvironment.spatialEnvironment_Generator import generate_spatialEnvironment
    generate_spatialEnvironment()
    stl.Done(start_time)

if regenBgeo:
    # geomagnetic field
    stl.prgMsg('Regenerating Bgeo')
    from src.physicsModels.ionosphere.geomagneticField.test_scripts.verify_geomagneticField import generateGeomagneticField
    generateGeomagneticField(showPlot=True)
    stl.Done(start_time)

if regenPlasmaEnvironment:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment')
    from src.physicsModels.ionosphere.PlasmaEnvironment.plasmaEnvironment_Generator import generatePlasmaEnvironment
    generatePlasmaEnvironment(showPlot=True)
    stl.Done(start_time)

if regenNeutralEnvironment:
    # neutral atmosphere
    stl.prgMsg('Regenerating Neutral Environment')
    from src.physicsModels.ionosphere.neutralEnvironment.ionoNeutralEnvironment_Generator import generateNeutralEnvironment
    generateNeutralEnvironment(showPlot=True)
    stl.Done(start_time)

if ionRecomb_ne_Calc:
    stl.prgMsg('Generating electron density from Ionization/Recombination\n')
    from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_Generator import generateIonizationRecomb
    generateIonizationRecomb(GenToggles,ionizationRecombToggles)
    stl.Done(start_time)

if calc_IonoConductivity:
    # conductivity
    stl.prgMsg('Calculating Ionospheric Conductivity')
    from src.physicsModels.ionosphere.conductivity.conductivity_Generator import generateIonosphericConductivity
    generateIonosphericConductivity(GenToggles, conductivityToggles)
    stl.Done(start_time)