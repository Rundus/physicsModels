##########################
# --- NEUTRALS TOGGLES ---
##########################
class NeutralsToggles:
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = rf'{SimToggles.sim_root_path}/neutral_environment'
    NRLMSIS_filePath = rf'{SimToggles.sim_root_path}/NRLMSIS/ACESII/NRLMSIS2.0.3D.2022324.nc'
    wNeutrals = ['N2','O2','O'] # which neutrals to consider in the simulation, use the key format in spacetoolsLib
