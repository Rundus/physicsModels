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
primaryBeam_fitting = True


################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################


if primaryBeam_fitting:
    stl.prgMsg('Fitting Primary Beam')
    from invertedV_fitting.primaryBeam_fitting.primaryBeamFits_Generator import generatePrimaryBeamFit
    generatePrimaryBeamFit(GenToggles, primaryBeamToggles, showPlot=True)
    stl.Done(start_time)

