#######################################
# --- Electrostatic Potential FIELD ---
#######################################
class ElectroStaticToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\electrostaticPotential'
    perform_mapping = True
    N_avg = 3 # number of points to average together for the electric potential mapping


    # relaxation method
    n_iter = 20 # number of times to iterate solution grid