###########################
# --- GEOMAGNETIC FIELD ---
###########################
class CurrentsToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\currents'

    smooth_data = True

    # SSA data
    use_SSA_filter = True
    fH = 0.67 # High Flyer Spin Freq
    fL = 0.545 # Low Flyer Spin Freq
    T = (2 / (fH + fL)) / 0.05 # Combined Period
    num_of_spin_periods = 10 # use this many spin periods
    wLen = int(num_of_spin_periods * T)
    mirror_percent = 0.0
    DC_components = [0]
    AC_components = [i for i in range(wLen) if i not in [0,1]]

    # savitz-golay
    use_savitz_golay = False
    polyorder = 3
    window = 500

    # boxcar
    use_boxcar = False
    N_boxcar = 20
