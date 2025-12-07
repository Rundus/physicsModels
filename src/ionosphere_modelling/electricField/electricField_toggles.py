########################
# --- ELECTRIC FIELD ---
########################
class EFieldToggles:
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}/electricField'

    include_neutral_winds = True
    useEFieldDetrend = True
    detrend_label = '_detrend' if useEFieldDetrend else ''
