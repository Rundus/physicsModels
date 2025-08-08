#######################################
# --- Electrostatic Potential FIELD ---
#######################################
class ElectroStaticToggles:
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}\electrostaticPotential'
    perform_mapping = True
    N_avg = 3 # number of points to average together for the electric potential mapping


    # relaxation method
    n_iter = 20 # number of times to iterate solution grid