# --- executable_invertedVFitting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: performs fits on diffNFlux data for ACES-II, but can be modified for other fluxes


#################
# --- IMPORTS ---
#################
import time
from simToggles_invertedVFitting import primaryBeamToggles,GenToggles,primaryBeamPlottingToggles, backScatterToggles
import spaceToolsLib as stl
start_time = time.time()


#################
# --- TOGGLES ---
#################
primaryBeam_fitting = True
primaryBeam_Plotting = True
backScatter_Calc = False
backScatter_Plotting = False



################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################


if primaryBeam_fitting:
    stl.prgMsg('Generating Primary Beam Fit Parameters\n')
    from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Generator import generatePrimaryBeamFit
    generatePrimaryBeamFit(GenToggles, primaryBeamToggles)
    stl.Done(start_time)

if primaryBeam_Plotting:
    stl.prgMsg('Plotting Primary Beam Fits\n')
    from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Plotting import generatePrimaryBeamFitPlots
    generatePrimaryBeamFitPlots(GenToggles, primaryBeamToggles, primaryBeamPlottingToggles, showPlot=True)
    stl.Done(start_time)

if backScatter_Calc:
    stl.prgMsg('Generating Secondary/Backscatter Data')
    from src.physicsModels.invertedV_fitting.BackScatter.backScatter_Generator import generateSecondaryBackScatter
    generateSecondaryBackScatter(GenToggles, primaryBeamToggles, backScatterToggles, showPlot=True)
    stl.Done(start_time)

if backScatter_Plotting:
    stl.prgMsg('Plotting BackScatter Beam Fits\n')
    from src.physicsModels.invertedV_fitting.BackScatter.backScatter_Plotting import generateBackScatterPlots
    generateBackScatterPlots(GenToggles)
    stl.Done(start_time)