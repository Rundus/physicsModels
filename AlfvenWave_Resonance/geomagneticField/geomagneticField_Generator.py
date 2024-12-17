# --- imports ---
from AlfvenWave_Resonance.TestParticle.simToggles import GenToggles,BgeoToggles, runFullSimulation


########################################
# --- GENERATE THE B-FIELD & TOGGLES ---
########################################
plot_BField = True

# --- OUTPUT DATA ------
outputData_Bool = True if not runFullSimulation else True

# --- EXECUTE ---
from ionosphere.geomagneticField.ionoGeomagneticField_Generator import generateGeomagneticField
generateGeomagneticField(outputData=outputData_Bool,GenToggles=GenToggles, BgeoToggles=BgeoToggles,plotting=plot_BField)

