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
# 0 - False; Don't run this
# 1 - True; Run this

dict_executable = {
    'regen_EVERYTHING' : 0,
    'regenSpatial': 0,
    'regenBgeo': 0,
    'regenNeSpectrum' : 0,
    'regenPlasmaEnvironment': 0,
    'regenNeutralEnvironment': 0,
    'ionRecomb_ne_Calc' : 0,
    'calc_IonoConductivity': 0,
    'map_electrostatic_potential': 1,
    'calc_electricField': 0,
    'calc_IonoCurrents': 0
}

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################

# re-run everything
if dict_executable['regen_EVERYTHING']==1:
    for key in dict_executable.keys():
        dict_executable[key] = 1
# re-run specifics
if dict_executable['regenSpatial']==1:
    # spatial environment
    stl.prgMsg('Regenerating Spatial Environment\n')
    from src.ionosphere_modelling.spatial_environment.spatial_environment_Generator import generate_spatialEnvironment
    generate_spatialEnvironment()
    stl.Done(start_time)
if dict_executable['regenBgeo']==1:
    # geomagnetic field
    stl.prgMsg('Regenerating Bgeo\n')
    from src.ionosphere_modelling.geomagneticField.geomagneticField_Generator import generate_GeomagneticField
    generate_GeomagneticField()
    stl.Done(start_time)

if dict_executable['regenNeSpectrum']==1:
    # ne spectrum
    stl.prgMsg('Regenerating ne spectrum\n')

    from src.ionosphere_modelling.plasma_environment.plasma_toggles import plasmaToggles

    if plasmaToggles.useEISCAT_density_Profile:
        from src.ionosphere_modelling.plasma_environment.EISCAT_ne_spectrum.EISCAT_ne_spectrum import EISCAT_ne_spectrum
        EISCAT_ne_spectrum()
    elif plasmaToggles.useACESII_density_Profile:
        from src.ionosphere_modelling.plasma_environment.ACESII_Langmuir_ni_spectrogram.Langmuir_ni_spectrogram import langmuir_ni_spectrogram
        langmuir_ni_spectrogram()
    stl.Done(start_time)


if dict_executable['regenPlasmaEnvironment']==1:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment\n')
    from src.ionosphere_modelling.plasma_environment.plasma_environment_Generator import generatePlasmaEnvironment
    generatePlasmaEnvironment()
    stl.Done(start_time)

if dict_executable['regenNeutralEnvironment']==1:
    # neutral atmosphere
    stl.prgMsg('Regenerating Neutral Environment \n')
    from src.ionosphere_modelling.neutral_environment.neutral_environment_Generator import generateNeutralEnvironment
    generateNeutralEnvironment()
    stl.Done(start_time)

if dict_executable['ionRecomb_ne_Calc']==1:
    stl.prgMsg('Generating electron density from Ionization/Recombination \n')
    from src.ionosphere_modelling.ionization_recombination.ionizationRecomb_Generator import generateIonizationRecomb
    generateIonizationRecomb()
    stl.Done(start_time)

if dict_executable['calc_IonoConductivity']==1:
    # conductivity
    stl.prgMsg('Calculating Ionospheric Conductivity')
    from src.ionosphere_modelling.conductivity.conductivity_Generator import generateIonosphericConductivity
    generateIonosphericConductivity()
    stl.Done(start_time)


if dict_executable['map_electrostatic_potential']==1:
    # electrostatic potential mapping
    stl.prgMsg('Mapping Electrostatic Potential')
    from src.ionosphere_modelling.electrostaticPotential.electrostaticPotential_Generator import generateElectrostaticPotential
    generateElectrostaticPotential()
    stl.Done(start_time)

if dict_executable['calc_electricField']==1:
    # electric field
    stl.prgMsg('Calculating Electric Field Evironment')
    from src.ionosphere_modelling.electricField.electricField_Generator import generate_electricField
    generate_electricField()
    stl.Done(start_time)

if dict_executable['calc_IonoCurrents']==1:
    # conductivity
    stl.prgMsg('Calculating Ionospheric Currents')
    from src.ionosphere_modelling.currents.currents_Generator import generate_Currents
    generate_Currents()
    stl.Done(start_time)