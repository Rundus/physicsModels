# --- executable_ionosphere.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: regenerate the IONOSPHERE plasma environment toggles


#################
# --- IMPORTS ---
#################
import time
from src.physicsModels.ionosphere.simToggles_Ionosphere import GenToggles,BgeoToggles,plasmaToggles,neutralsToggles,conductivityToggles,ionizationRecombToggles
import spaceToolsLib as stl
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()


#################
# --- TOGGLES ---
#################
regenBgeo = False
regenPlasmaEnvironment = False
regenNeutralEnvironment = False
regenHeightIonization = True
regenIonoConductivity = False

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################

if regenBgeo:
    # geomagnetic field
    stl.prgMsg('Regenerating Bgeo')
    from src.physicsModels.ionosphere.geomagneticField.geomagneticField_Generator import generateGeomagneticField
    generateGeomagneticField(outputData=True,GenToggles=GenToggles, BgeoToggles=BgeoToggles,showPlot=True)
    stl.Done(start_time)

if regenPlasmaEnvironment:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment')
    from src.physicsModels.ionosphere.PlasmaEnvironment.plasmaEnvironment_Generator import generatePlasmaEnvironment
    generatePlasmaEnvironment(True,GenToggles,plasmaToggles,showPlot=True)
    stl.Done(start_time)

if regenNeutralEnvironment:
    # neutral atmosphere
    stl.prgMsg('Regenerating Neutral Environment')
    from src.physicsModels.ionosphere.neutralEnvironment.ionoNeutralEnvironment_Generator import generateNeutralEnvironment
    generateNeutralEnvironment(GenToggles, neutralsToggles, showPlot=True)
    stl.Done(start_time)

if regenHeightIonization:
    # height ionization
    stl.prgMsg('Regenerating Height Ionization and Recombination')
    from src.physicsModels.ionosphere.ionizationRecomb.TestScripts.verify_IonizationRecombinationModels import generateHeightIonization
    generateHeightIonization(ionizationRecombToggles, showPlot=True)
    stl.Done(start_time)

if regenIonoConductivity:
    # conductivity
    stl.prgMsg('Regenerating Ionospheric Conductivity')
    from src.physicsModels.ionosphere.conductivity.conductivity_Generator import generateIonosphericConductivity
    generateIonosphericConductivity(True, GenToggles,conductivityToggles, showPlot=True)
    stl.Done(start_time)