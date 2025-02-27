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
regenBgeo = False
regenPlasmaEnvironment = False
regenNeutralEnvironment = False
ionRecomb_ne_Calc = False
calc_IonoConductivity = True

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################

if regenBgeo:
    # geomagnetic field
    stl.prgMsg('Regenerating Bgeo')
    from src.physicsModels.ionosphere.geomagneticField.geomagneticField_Generator import generateGeomagneticField
    generateGeomagneticField(GenToggles=GenToggles, BgeoToggles=BgeoToggles,showPlot=True)
    stl.Done(start_time)

if regenPlasmaEnvironment:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment')
    from src.physicsModels.ionosphere.PlasmaEnvironment.plasmaEnvironment_Generator import generatePlasmaEnvironment
    generatePlasmaEnvironment(GenToggles,plasmaToggles,showPlot=True)
    stl.Done(start_time)

if regenNeutralEnvironment:
    # neutral atmosphere
    stl.prgMsg('Regenerating Neutral Environment')
    from src.physicsModels.ionosphere.neutralEnvironment.ionoNeutralEnvironment_Generator import generateNeutralEnvironment
    generateNeutralEnvironment(GenToggles, neutralsToggles, showPlot=True)
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