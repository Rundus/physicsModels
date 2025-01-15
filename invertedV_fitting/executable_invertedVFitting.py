# --- executable_invertedVFitting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: performs fits on diffNFlux data for ACES-II, but can be modified for other fluxes


#################
# --- IMPORTS ---
#################
import time
from simToggles_invertedVFitting import primaryBeamToggles,GenToggles,primaryBeamPlottingToggles,secondaryBackScatterToggles
import spaceToolsLib as stl
start_time = time.time()

#################
# --- TOGGLES ---
#################
primaryBeam_fitting = True
primaryBeam_Plotting = False
calcBackscatter = False


################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################


if primaryBeam_fitting:
    stl.prgMsg('Generating Primary Beam Fit Parameters\n')
    from invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Generator import generatePrimaryBeamFit
    generatePrimaryBeamFit(GenToggles, primaryBeamToggles)
    stl.Done(start_time)

if primaryBeam_Plotting:
    stl.prgMsg('Plotting Primary Beam Fits\n')
    from invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Plotting import generatePrimaryBeamFitPlots
    generatePrimaryBeamFitPlots(GenToggles, primaryBeamToggles, primaryBeamPlottingToggles, showPlot=True)
    stl.Done(start_time)

if calcBackscatter:
    stl.prgMsg('Generating Secondary/Backscatter Data')
    from invertedV_fitting.BackScatter.backScatter_Generator import generateSecondaryBackScatter
    generateSecondaryBackScatter(GenToggles, primaryBeamToggles, secondaryBackScatterToggles, showPlot=True)
    stl.Done(start_time)