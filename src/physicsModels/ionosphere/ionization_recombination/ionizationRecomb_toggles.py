###########################
# --- HEIGHT IONIZATION ---
###########################
class ionizationRecombToggles:
    outputFolder = 'C:\Data\physicsModels\ionosphere\ionizationRecomb'

    flux_path = 'C:\Data\ACESII\L3\Energy_Flux\high'

    # --- BEAM n_E: which dataset to use for the n_e profile ---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_evans1974_beam = False
    use_eepaa_beam = False  # if True, uses the n_e profile derived from the High Flyer Data.