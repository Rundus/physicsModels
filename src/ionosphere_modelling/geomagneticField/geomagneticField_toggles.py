###########################
# --- GEOMAGNETIC FIELD ---
###########################
import datetime as dt
class BgeoToggles:
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}\geomagneticField'
    targetTime = dt.datetime(2022,11,20,17)