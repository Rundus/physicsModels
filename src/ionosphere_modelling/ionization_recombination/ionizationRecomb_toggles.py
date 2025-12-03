###########################
# --- HEIGHT IONIZATION ---
###########################
class IonizationRecombToggles:
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}/ionizationRecomb'

    flux_path = f'{SimToggles.sim_root_path}/data_inputs/energy_flux/high'

    # --- BEAM n_E: which dataset to use for the n_e profile ---
    # Description: use Evans1974 beam model for n(z) OR a real-data derived model n(z)
    use_evans1974_beam = False
    use_eepaa_beam = False  # if True, uses the n_e profile derived from the High Flyer Data.