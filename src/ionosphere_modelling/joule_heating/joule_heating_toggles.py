###########################
# --- GEOMAGNETIC FIELD ---
###########################
import datetime as dt
class JouleHeatingToggles:
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}/joule_heating'