# --- executable_iono.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: regenerate the IONOSPHERE plasma environment toggles


#################
# --- IMPORTS ---
#################
import time
from ionosphere.simToggles_iono import GenToggles,BgeoToggles,plasmaToggles,neutralsToggles,conductivityToggles,heightIonizationToggles
import spaceToolsLib as stl
start_time = time.time()


#################
# --- TOGGLES ---
#################
regenBgeo = False
regenPlasmaEnvironment = False
regenNeutralEnvironment = False
regenHeightIonization = False
regenIonoConductivity = False

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################

if regenBgeo:
    # geomagnetic field
    stl.prgMsg('Regenerating Bgeo')
    from ionosphere.geomagneticField.ionoGeomagneticField_Generator import generateGeomagneticField
    generateGeomagneticField(outputData=True,GenToggles=GenToggles, BgeoToggles=BgeoToggles,showPlot=True)
    stl.Done(start_time)

if regenPlasmaEnvironment:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment')
    from ionosphere.PlasmaEnvironment.ionoPlasmaEnvironment_Generator import generatePlasmaEnvironment
    generatePlasmaEnvironment(True,GenToggles,plasmaToggles,showPlot=True)
    stl.Done(start_time)

if regenNeutralEnvironment:
    # neutral atmosphere
    stl.prgMsg('Regenerating Neutral Environment')
    from ionosphere.neutralEnvironment.ionoNeutralEnvironment_Generator import generateNeutralEnvironment
    generateNeutralEnvironment(GenToggles, neutralsToggles, showPlot=True)
    stl.Done(start_time)

if regenHeightIonization:
    # height ionization
    stl.prgMsg('Regenerating Height Ionization')
    from ionosphere.heightIonization.heightIonization_Generator import generateHeightIonization
    generateHeightIonization(GenToggles, heightIonizationToggles, showPlot=True)
    stl.Done(start_time)

if regenIonoConductivity:
    # conductivity
    stl.prgMsg('Regenerating Ionospheric Conductivity')
    from ionosphere.conductivity.ionoConductivity_Generator import generateIonosphericConductivity
    generateIonosphericConductivity(True, GenToggles,conductivityToggles, showPlot=True)
    stl.Done(start_time)