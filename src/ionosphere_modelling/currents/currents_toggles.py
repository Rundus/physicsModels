###########################
# --- GEOMAGNETIC FIELD ---
###########################
class CurrentsToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\currents'


    smooth_data = True

    # savitz-golay
    use_savitz_golay = False
    polyorder = 3
    window = 500

    # boxcar
    use_boxcar = True
    N_boxcar = 10
