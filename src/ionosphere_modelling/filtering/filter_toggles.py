class FilterToggles:
    # --- --- --- --- --- --- --- --- --- --- --- ---
    # --- FILTER THE CONDUCTIVITY AND E-FIELD DATA --
    # --- --- --- --- --- --- --- --- --- --- --- ---
    # Description: Use a choice of filters to clean-up the conductivity + E-Field data
    # before calculating the currents.

    # Main toggle
    filter_data = True

    # SSA data
    use_SSA_filter = True
    fH = 0.67  # High Flyer Spin Freq
    fL = 0.545  # Low Flyer Spin Freq
    T = (2 / (fH + fL)) / 0.05  # Averaged spin Period
    num_of_spin_periods = 10  # use this many spin periods
    wLen = int(num_of_spin_periods * T)
    mirror_percent = 0.0
    DC_components = [0,1]
    AC_components = [i for i in range(wLen) if i not in [0]]

    # savitz-golay
    use_savitz_golay = False
    polyorder = 3
    window = 500

    # boxcar
    use_boxcar = False
    N_boxcar = 20


    # FILE I/O
    if use_SSA_filter:
        filter_path = 'ssa_filtered'
    elif use_savitz_golay:
        filter_path = 'savitz_golay_filtered'
    elif use_boxcar:
        filter_path = 'boxcar_filtered'

    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}/filtered'