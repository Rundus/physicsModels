# --- executable.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: regenerate the IONOSPHERE plasma environment toggles


#################
# --- IMPORTS ---
#################
import time
import spaceToolsLib as stl
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()


#################
# --- TOGGLES ---
#################
regenSpatial = False
regenBgeo = False
regenNeSpectrum = False
regenPlasmaEnvironment = True
regenNeutralEnvironment = False
ionRecomb_ne_Calc = False
calc_IonoConductivity = True

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################

if regenSpatial:
    # spatial environment
    stl.prgMsg('Regenerating Spatial Environment\n')
    from src.ionosphere_modelling.spatial_environment.spatial_environment_Generator import generate_spatialEnvironment
    generate_spatialEnvironment()
    stl.Done(start_time)
if regenBgeo:
    # geomagnetic field
    stl.prgMsg('Regenerating Bgeo\n')
    from src.ionosphere_modelling.geomagneticField.geomagneticField_Generator import generate_GeomagneticField
    generate_GeomagneticField()
    stl.Done(start_time)

if regenNeSpectrum:
    # ne spectrum
    stl.prgMsg('Regenerating ne spectrum\n')
    from src.ionosphere_modelling.plasma_environment.ACESII_Langmuir_ni_spectrogram.Langmuir_ni_spectrogram import langmuir_ni_spectrogram
    langmuir_ni_spectrogram()
    stl.Done(start_time)


if regenPlasmaEnvironment:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment\n')
    from src.ionosphere_modelling.plasma_environment.plasma_environment_Generator import generatePlasmaEnvironment
    generatePlasmaEnvironment()
    stl.Done(start_time)

if regenNeutralEnvironment:
    # neutral atmosphere
    stl.prgMsg('Regenerating Neutral Environment \n')
    from src.ionosphere_modelling.neutral_environment.neutral_environment_Generator import generateNeutralEnvironment
    generateNeutralEnvironment()
    stl.Done(start_time)

if ionRecomb_ne_Calc:
    stl.prgMsg('Generating electron density from Ionization/Recombination \n')
    from src.ionosphere_modelling.ionization_recombination.ionizationRecomb_Generator import generateIonizationRecomb
    generateIonizationRecomb()
    stl.Done(start_time)

if calc_IonoConductivity:
    # conductivity
    stl.prgMsg('Calculating Ionospheric Conductivity')
    from src.ionosphere_modelling.conductivity.conductivity_Generator import generateIonosphericConductivity
    generateIonosphericConductivity()
    stl.Done(start_time)