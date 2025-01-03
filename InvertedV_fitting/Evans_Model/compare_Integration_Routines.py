# --- compare_Integration_Routines.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Use the Evans 1974 data to validate integration techniques for fluxes
import matplotlib.pyplot as plt

#################
# --- IMPORTS ---
#################
from ACESII_code.class_var_func import q0,m_e
import numpy as np
from ACESII_code.Science.InvertedV.Evans_class_var_funcs import dist_Maxwellian,calc_diffNFlux,velocitySpace_to_PitchEnergySpace,calc_DistributionMapping



#################
# --- TOGGLES ---
#################
# model_T = 800
# model_n = 1.5
# model_V0 = 2000
model_T = 125
model_n = 3
model_V0 = 275
N = 1000
EnergyRangeValue = 25000 # in eV


# plot the maxwellian distribution
plot_accelerated_Distribution = True
Vnorm = 1E7 # what to normalize velocitySpaceBy
Vunits = 'm/s' # what units for velocity space are you using
usePitchEnergy = False # if False then uses velocity space

####################################
# --- DEFINE THE MAXWELLIAN DATA ---
####################################

# define the undisturbed Maxwellian
Vperp_gridVals = np.linspace(-np.sqrt(2*EnergyRangeValue*q0/m_e), np.sqrt(2*EnergyRangeValue*q0/m_e), N)
Vpara_gridVals = np.linspace(0, np.sqrt(2*EnergyRangeValue*q0/m_e), N)
distGrid,VperpGrid, VparaGrid, diffNFluxGrid, VperpGrid_Accel, VparaGrid_Accel, diffNFluxGrid_Accel, VperpGrid_iono, VparaGrid_iono, diffNFluxGrid_iono = calc_DistributionMapping(Vperp_gridVals=Vperp_gridVals,
                                                                                                                                                                              Vpara_gridVals=Vpara_gridVals,
                                                                                                                                                                              model_T=model_T,
                                                                                                                                                                              model_n=model_n,
                                                                                                                                                                              model_V0=model_V0,
                                                                                                                                                                              beta=1)


# --- plot it ----
if plot_accelerated_Distribution:

    fig, ax = plt.subplots(2, 2)
    figure_width = 10  # in inches
    figure_height = 8  # in inches
    fig.set_size_inches(figure_width, figure_height)
    from my_matplotlib_Assets.colorbars.apl_rainbow_black0 import apl_rainbow_black0_cmap
    mycmap = apl_rainbow_black0_cmap()

    titles = ['Maxwellian', 'Accelerated']
    for k in range(2):

        if usePitchEnergy:
            EnergyBins_conversion = np.linspace(0, EnergyRangeValue, N)

            PitchBins_conversion = np.linspace(-90, 90, N)

            # non-accelerated
            distGrid_pE, EnergyGrid, PitchGrid = velocitySpace_to_PitchEnergySpace(VparaGrid=VparaGrid,VperpGrid=VperpGrid,ZGrid=distGrid,EnergyBins=EnergyBins_conversion,PitchBins=PitchBins_conversion)
            diffNFluxGrid_pE, EnergyGrid, PitchGrid = velocitySpace_to_PitchEnergySpace(VparaGrid=VparaGrid,VperpGrid=VperpGrid, ZGrid=diffNFluxGrid, EnergyBins=EnergyBins_conversion, PitchBins=PitchBins_conversion)

            # accelerated
            distGrid_Accel_pE, EnergyGrid_Accel, PitchGrid_Accel = velocitySpace_to_PitchEnergySpace(VparaGrid=VparaGrid_Accel, VperpGrid=VperpGrid_Accel, ZGrid=distGrid, EnergyBins=EnergyBins_conversion,PitchBins=PitchBins_conversion)
            diffNFluxGrid_Accel_pE, EnergyGrid_Accel, PitchGrid_Accel = velocitySpace_to_PitchEnergySpace(VparaGrid=VparaGrid_Accel, VperpGrid=VperpGrid_Accel, ZGrid=diffNFluxGrid_Accel, EnergyBins=EnergyBins_conversion, PitchBins=PitchBins_conversion)

            XYGrids = [[PitchGrid,EnergyGrid],[PitchGrid_Accel,EnergyGrid_Accel]]
            xLabel = 'Pitch'
            yLabel = 'Energy'
            xLims = [-100, 100]
            yLims = [1, 10000]
            yscale='log'
            xscale='linear'

        else:
            XYGrids = [[VperpGrid / (Vnorm), VparaGrid / (Vnorm)],[VperpGrid_Accel/ (Vnorm), VparaGrid_Accel/ (Vnorm)]]
            xLabel = f'Vperp [{format(Vnorm,".1E")} {Vunits}]'
            yLabel = f'Vpara [{format(Vnorm,".1E")} {Vunits}]'
            xLims = [-10,10]
            yLims = [0, 10]
            yscale = 'linear'
            xscale = 'linear'


        if k == 0:
            if usePitchEnergy:
                Zgrids = [distGrid_pE,distGrid_Accel_pE]
            else:
                Zgrids = [distGrid, distGrid]
            vmin, vmax = 1E-22, 1E-16
            cbarLabel = 'Distribution Function [$m^{-6}s^{-3}$]'

        else:
            if usePitchEnergy:
                Zgrids = [diffNFluxGrid_pE,diffNFluxGrid_Accel_pE]
            else:
                Zgrids = [diffNFluxGrid, diffNFluxGrid_Accel]

            vmin, vmax = 1E5, 1E18
            cbarLabel = 'diff_N_Flux [$cm^{-2}s^{-1}eV^{-1}str^{-1}$]'


        for i in [0, 1]:
            cmap = ax[i, k].pcolormesh(XYGrids[i][0] , XYGrids[i][1], Zgrids[i], cmap=mycmap, norm='log', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(cmap, ax=ax[i, k])
            cbar.set_label(cbarLabel)
            # if k in [1]:
            #     levels = [3.52E3, 1.04E4,1.56E4,2.08E4,9.87E4]
            #     CS = ax[i,k].contour(XYGrids[i][0] , XYGrids[i][1], Zgrids[i], levels=levels)
            #     ax[i,k].clabel(CS,inline=True,fontsize=7)
            # else:
            #     CS = ax[i, k].contour(XYGrids[i][0], XYGrids[i][1], Zgrids[i])
            #     ax[i, k].clabel(CS, inline=True, fontsize=7)


            ax[i, k].set_ylabel(yLabel)
            ax[i, k].set_xlabel(xLabel)
            ax[i, k].set_ylim(*yLims)
            ax[i, k].set_xlim(*xLims)
            if not usePitchEnergy:
                ax[i, k].invert_yaxis()
            ax[i, k].set_title(titles[i])
            ax[i, k].set_yscale(yscale)
            ax[i, k].set_xscale(xscale)

            if i in [1] and not usePitchEnergy:
                ax[i, k].axhline(np.sqrt(2 * model_V0 * q0 / m_e) / (1000 * 10000), color='red', label='$V_{0}$' + f'= {model_V0} eV')
    plt.tight_layout()
    plt.show()


#####################################
# --- CALCULATE TOTAL NUMBER FLUX ---
#####################################
# description: Determine total number flux: In units of electrons / cm^2 sec. To do this,
# we integrate the diffNFlux over velocity space using cylindrical coordinates since this constitutes an
# integral over both energy and pitch angle:

# Integrating in cyclindrical coordinates is the most obvious
from scipy.integrate import simpson
integratedValue_rowWise = []

for rowIdx in range(len(diffNFluxGrid_Accel)):
    integratedValue_rowWise.append(simpson(x=VperpGrid_Accel[rowIdx],y=diffNFluxGrid_Accel[rowIdx]))

print(simpson(x=VparaGrid_Accel.T[0],y=np.array(integratedValue_rowWise)))

integratedValue_colWise = []
newX = VparaGrid_Accel.T
newY = diffNFluxGrid_Accel.T

for colIdx in range(len(diffNFluxGrid_Accel)):
    integratedValue_colWise.append(simpson(x=newX[colIdx],y=newY[colIdx]))

print(simpson(x=VperpGrid_Accel[0],y=np.array(integratedValue_colWise)))