# --- executable_invertedVFitting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: performs fits on diffNFlux data for ACES-II, but can be modified for other fluxes


#################
# --- IMPORTS ---
#################
import time
from simToggles_invertedVFitting import primaryBeamToggles,GenToggles
import spaceToolsLib as stl
start_time = time.time()

#################
# --- TOGGLES ---
#################
primaryBeam_fitting = False
primaryBeam_Plotting = True


################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################


if primaryBeam_fitting:
    stl.prgMsg('Generatorating Primary Beam Fit Parameters')
    from invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Generator import generatePrimaryBeamFit
    generatePrimaryBeamFit(GenToggles, primaryBeamToggles)
    stl.Done(start_time)

if primaryBeam_Plotting:
    stl.prgMsg('Plotting Primary Beam Fits')
    from invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Plotting import generatePrimaryBeamFitPlots
    generatePrimaryBeamFitPlots(GenToggles, primaryBeamToggles, showPlot=True)
    stl.Done(start_time)
