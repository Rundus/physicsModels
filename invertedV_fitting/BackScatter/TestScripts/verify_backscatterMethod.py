# --- verify_backscatterMethod.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: use the digitized values from Evans 1974 Fig. 5 to check my
# Secondary/Backscatter code

#################
# --- IMPORTS ---
#################
from invertedV_fitting.BackScatter.Evans_Model.parameterizationCurves_Evans1974_classes import *
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

#################
# --- TOGGLES ---
#################
plot_Evans1974Curves = True
model_n = 1.5 # cm^-3
model_T = 800 # eV
model_V0 = 2000 # V
model_vPos = 2000 # inverted-V altitude (in km)


########################
# --- TEST FUNCTIONS ---
########################

def testModel_generate_BackScatterCurve_0deg():
    model= Evans1974()
    return CubicSpline(model.testModel_BackScatter_Energies_0deg,model.testModel_BackScatter_Flux_0deg)

def testModel_generate_PrimaryBeam_0deg():
    model= Evans1974()
    return CubicSpline(model.testModel_Beam_Energies_0deg, model.testModel_Beam_Flux_0deg)

def testModel_generate_BackScatterCurve_45deg():
    model= Evans1974()
    return CubicSpline(model.testModel_BackScatter_Energies_45deg, model.testModel_BackScatter_Flux_45deg)
def testModel_generate_PrimaryBeam_45deg():
    model= Evans1974()
    return CubicSpline(model.testModel_Beam_Energies_45deg, model.testModel_Beam_Flux_45deg)

###################################
# --- Generate the example data ---
###################################
model = Evans1974()

# backscatter - 0deg
minE, maxE = np.min(model.testModel_BackScatter_Energies_0deg),np.max(model.testModel_BackScatter_Energies_0deg)
xData_backscatter_0deg = np.linspace(minE,maxE,1000)
spline = testModel_generate_BackScatterCurve_0deg()
yData_backscatter_0deg = spline(xData_backscatter_0deg)

# beam - 0deg
minE, maxE = np.min(model.testModel_Beam_Energies_0deg), np.max(model.testModel_Beam_Energies_0deg)
xData_beam_0deg = np.linspace(minE, maxE, 1000)
spline = testModel_generate_PrimaryBeam_0deg()
yData_beam_0deg = spline(xData_beam_0deg)

# total - 0deg
xData_0deg = np.append(xData_backscatter_0deg,xData_beam_0deg)
yData_0deg = np.append(yData_backscatter_0deg, yData_beam_0deg)

# backscatter - 45deg
minE, maxE = np.min(model.testModel_BackScatter_Energies_45deg), np.max(model.testModel_BackScatter_Energies_45deg)
xData_backscatter_45deg = np.linspace(minE, maxE, 1000)
spline = testModel_generate_BackScatterCurve_45deg()
yData_backscatter_45deg = spline(xData_backscatter_45deg)

# beam - 0deg
minE, maxE = np.min(model.testModel_Beam_Energies_45deg), np.max(model.testModel_Beam_Energies_45deg)
xData_beam_45deg = np.linspace(minE, maxE, 1000)
spline = testModel_generate_PrimaryBeam_0deg()
yData_beam_45deg = spline(xData_beam_45deg)

# total - 45deg
xData_45deg = np.append(xData_backscatter_45deg, xData_beam_45deg)
yData_45deg = np.append(yData_backscatter_45deg, yData_beam_45deg)

#########################################
# --- Apply my Code to the Evans Data ---
#########################################


relevantPitchAngles = np.array([0 +i*10 for i in range(3)])
diffNFlux_Beam = np.array([yData_beam_0deg for i in range(len(relevantPitchAngles))])

# GENERATE BACKSCATTER - my method
omniFlux = helperFuncs().calcTotal_NFlux(diffNFlux=diffNFlux_Beam,
                                            pitchValues=relevantPitchAngles,
                                            energyValues=xData_beam_0deg)

# use OmniFlux to determine secondary response from the primary Beam. Out is shape=(len(Energy))
secondaryFlux_0deg = Evans1974().calcSecondaries(detectorEnergies=xData_0deg,
                                            InputOmniFlux=omniFlux,
                                            Niterations=secondaryBackScatterToggles.Niterations_secondaries,
                                            V0=model_V0)

# calculate the OmniFlux - Integrate over pitch angle
omniDiffFlux = helperFuncs().calcOmni_diffNFlux(diffNFlux=diffNFlux_Beam,
                                                   pitchValues=relevantPitchAngles,
                                                   energyValues=xData_beam_0deg)

backscatterFlux, secondaryFlux_backscatter = Evans1974().calcBackScatter(
                                                IncidentBeamEnergies=xData_beam_0deg,
                                                Incident_OmniDiffFlux=omniDiffFlux,
                                                Niterations=secondaryBackScatterToggles.Niterations_backscatter,
                                                V0=model_V0,
                                                detectorEnergies = xData_0deg)



totalResponse = backscatterFlux  + secondaryFlux_0deg



##################
# --- PLOTTING ---
##################

Plot_Linewidth = 2.5

if plot_Evans1974Curves:

    # Plot it all
    fig, ax = plt.subplots(ncols=2)

    # 0 deg
    ax[0].plot(xData_backscatter_0deg, yData_backscatter_0deg, label='Evans1974 - 0deg', color='black',linewidth=Plot_Linewidth)
    ax[0].plot(xData_beam_0deg, yData_beam_0deg, color='black',linewidth=Plot_Linewidth)

    ax[0].plot(xData_0deg, secondaryFlux_0deg, label='Secondary Flux (primary)', color='tab:red',linewidth=Plot_Linewidth)
    ax[0].plot(xData_0deg, secondaryFlux_backscatter, label='Secondary Flux (Backscatter)', color='tab:blue', linewidth=Plot_Linewidth)
    ax[0].plot(xData_0deg, backscatterFlux, label='Backscatter Flux (Backscatter)', color='tab:green', linewidth=Plot_Linewidth)
    ax[0].plot(xData_0deg,totalResponse, label='Total Response', color='magenta', linewidth=Plot_Linewidth)

    # 45 deg
    ax[1].plot(xData_backscatter_45deg, yData_backscatter_45deg, label='45deg',color='black',linewidth=Plot_Linewidth)
    ax[1].plot(xData_beam_45deg, yData_beam_45deg, color='black',linewidth=Plot_Linewidth)

    for i in range(2):
        ax[i].set_yscale('log')
        ax[i].set_ylim(1E4, 1E7)
        ax[i].set_xscale('log')
        ax[i].set_xlim(1E1, 1E4)
        ax[i].legend()

    plt.show()