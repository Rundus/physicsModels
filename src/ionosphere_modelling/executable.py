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
from src.ionosphere_modelling.execution_toggles import *

###############################
# --- --- --- --- --- --- --- -
# --- SIMULATION GENERATORS ---
# --- --- --- --- --- --- --- -
###############################

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

    from src.ionosphere_modelling.plasma_environment.plasma_toggles import PlasmaToggles

    if PlasmaToggles.useEISCAT_density_Profile:
        from src.ionosphere_modelling.plasma_environment.EISCAT_ne_spectrum.EISCAT_ne_spectrum import EISCAT_ne_spectrum
        EISCAT_ne_spectrum()
    elif PlasmaToggles.useACESII_density_Profile:
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
    stl.prgMsg('Mapping Electrostatic Potential\n')
    from src.ionosphere_modelling.electrostaticPotential.electrostaticPotential_Generator import generateElectrostaticPotential
    generateElectrostaticPotential()
    stl.Done(start_time)

if dict_executable['calc_electricField']==1:
    # electric field
    stl.prgMsg('Calculating Electric Field Evironment\n')
    from src.ionosphere_modelling.electricField.electricField_Generator import generate_electricField
    generate_electricField()
    stl.Done(start_time)

if dict_executable['filter_EFields_conductivity'] == 1:
    from src.ionosphere_modelling.currents.currents_filter_toggles import FilterToggles
    if FilterToggles.filter_data:

        stl.prgMsg('Filtering E-Field Data\n')
        from src.ionosphere_modelling.electricField.electricField_filtered_data_Generator import generate_filtered_EField
        generate_filtered_EField()
        stl.Done(start_time)

        stl.prgMsg('Filtering Conductivity Data\n')
        from src.ionosphere_modelling.conductivity.conductivity_filtered_data_Generator import generate_filtered_conductivity
        generate_filtered_conductivity()
        stl.Done(start_time)

if dict_executable['calc_IonoCurrents']==1:
    stl.prgMsg('Calculating Ionospheric Currents\n')
    from src.ionosphere_modelling.currents.currents_Generator import generate_Currents
    generate_Currents()
    stl.Done(start_time)

if dict_executable['calc_PoyntingFlux']==1:
    stl.prgMsg('Calculating Poynting Flux\n')
    from src.ionosphere_modelling.poynting_flux.poynting_flux_Generator import generatePoyntingFlux
    generatePoyntingFlux()
    stl.Done(start_time)

if dict_executable['calc_JouleHeating']==1:
    # Joule Heating
    stl.prgMsg('Calculating Joule Heating\n')
    from src.ionosphere_modelling.joule_heating.joule_heating_Generator import generate_JouleHeating
    generate_JouleHeating()
    stl.Done(start_time)
