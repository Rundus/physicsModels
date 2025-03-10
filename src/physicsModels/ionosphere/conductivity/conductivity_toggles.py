######################
# --- CONDUCTIVITY ---
######################
class conductivityToggles:

    outputFolder = 'C:\Data\physicsModels\ionosphere\conductivity'

    # --- BEAM n_E: which dataset to use for the n_e profile ---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_evans1974_beam = False
    use_eepaa_beam = False # if True, uses the n_e profile derived from the High Flyer Data.

    # --- BACKGROUND n_e: Which dataset to use for the n_e profile---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_IRI_background = False
    use_eepaa_background = False  # if True, uses the n_e profile derived from the High Flyer Data.