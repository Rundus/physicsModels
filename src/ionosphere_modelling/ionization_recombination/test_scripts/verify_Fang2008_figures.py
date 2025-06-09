# --- verify_Fang2008_figures.py ---
# Description: JUST plot the Fang 2008 Figures for a choice of inputs par

# --- imports ---
import time
start_time = time.time()
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.ionosphere_modelling.ionization_recombination.ionizationRecomb_toggles import ionizationRecombToggles
from src.ionosphere_modelling.ionization_recombination.ionizationRecomb_classes import *
from src.ionosphere_modelling.plasma_environment.plasma_environment_classes import *
from src.ionosphere_modelling.neutral_environment.neutral_toggles import neutralsToggles
import numpy as np
import datetime as dt
from numpy import datetime64, squeeze
import pymsis
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata

##################
# --- PLOTTING ---
##################
# --- Plotting Toggles ---
figure_width = 25  # in inches
figure_height = 20  # in inches
Title_FontSize = 25
Label_FontSize = 25
Tick_FontSize = 25
Tick_FontSize_minor = 20
Tick_Length = 10
Tick_Width = 2
Tick_Length_minor = 5
Tick_Width_minor = 1
Text_Fontsize = 20
Plot_LineWidth = 6.5
Legend_fontSize = 32
dpi = 100

xNorm = stl.m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'


def verify_Fang2008_figures():

    # Define the input parameters
    altRange = np.linspace(50,500,100)*stl.m_to_km
    monoEnergyProfile = np.array([round(0.1 + 2*i*0.1,2) for i in range(5)])  # 10eV to 1000keV, IN UNITS OF KEV
    energyFluxProfile = (6.242E8) * np.array([1 for i in range(len(monoEnergyProfile))])  # provide in ergs but convert from ergs/cm^-2s^-1 to keV/cm^-2s^-1
    energyFluxProfile = np.array([1.7E9,3E10, 6.6E10, 4E10,4E9])  # provide in ergs but convert from ergs/cm^-2s^-1 to keV/cm^-2s^-1

    # construct a vertical profile using NRLMSIS data
    f107 = 150  # the F10.7 (DON'T CHANGE)
    f107a = 150  # ap data (DON't CHANGE)
    ap = 7
    aps = [[ap] * 7]
    Tn = np.zeros(shape=(len(altRange)))
    m_eff_n = np.zeros(shape=(len(altRange)))
    rho_n = np.zeros(shape=(len(altRange)))
    nn = [[] for i in range(8)]



    for i in range(len(altRange)):

        lat = 70
        long = 16
        dt_targetTime = dt.datetime(2022, 11, 20, 17, 20)
        date = datetime64(f"{dt_targetTime.year}-{dt_targetTime.month}-{dt_targetTime.day}T{dt_targetTime.hour}:{dt_targetTime.minute}")
        NRLMSIS_data = squeeze(pymsis.calculate(date, long, lat, altRange[i]/stl.m_to_km, f107, f107a, aps))

        counter = 0
        for var in pymsis.Variable:

            dat = NRLMSIS_data[var]

            if var.name == 'MASS_DENSITY':
                rho_n[i] = dat

            elif var.name == 'TEMPERATURE':
                Tn[i] = dat
            elif var.name in ['N2','O2','O','HE','H','AR','N','NO']:
                nn[counter].append(dat)
                counter +=1

    masses = np.array([val for val in stl.netural_dict.values()])
    m_eff_n = np.array([ np.nansum(np.array([nn[i][j]*masses[i] for i in range(8)]))/np.nansum(np.array([nn[i][j] for i in range(8)]))   for j in range(len(altRange))])

    # CHOOSE THE MODEL
    model = fang2010(altRange= altRange,
                     Tn= Tn,
                     m_eff_n= m_eff_n,
                     rho_n= rho_n,
                     inputEnergies=monoEnergyProfile,
                     varPhi_E=energyFluxProfile)
    H = model.scaleHeight()
    y = model.atmColumnMass(monoEnergyProfile)
    f = model.f(y,model.calcCoefficents(monoEnergyProfile))
    q_profiles, q_total = model.ionizationRate() # in cm^-3 s^-1

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2,ncols=2)
    fig.set_size_inches(figure_width, figure_height)

    ax[0, 0].plot(H/(100*1000), altRange / xNorm,  linewidth=Plot_LineWidth, label='H')
    ax[0, 0].set_xlabel('Scale Height [km]', fontsize=Label_FontSize)
    ax[0, 1].set_xlabel('Column Mass (y)', fontsize=Label_FontSize)
    ax[0, 1].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
    ax[0, 1].set_xlim(1E-6, 1E6)

    for idx, profile in enumerate(q_profiles):
        ax[0, 1].plot(y[idx], altRange / xNorm, linewidth=Plot_LineWidth)
        # ax[1, 0].plot(f[idx], y[idx],  linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")
        ax[1, 0].plot(f[idx], altRange / xNorm, linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")
        ax[1, 1].plot(q_profiles[idx], altRange / xNorm,  linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")

    # ax[1, 1].plot(q_total, altRange / xNorm,  linewidth=Plot_LineWidth, linestyle='--', label=rf"Total")
    ax[1, 1].set_xlabel('Total ionization Rate [cm$^{-3}$s$^{-1}$]', fontsize=Label_FontSize)
    ax[1,1].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
    ax[0, 0].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
    ax[1, 0].set_ylabel(f'Column Mass (y)', fontsize=Label_FontSize)
    ax[1, 0].set_xlabel(f'Energy Dissipation (f)', fontsize=Label_FontSize)
    ax[1, 1].set_xlim(1E-1, 1E6)
    ax[1, 0].set_xlim(0, 0.8)

    for i in range(2):
        for j in range(2):
            if [i, j] != [1,0]:
                ax[i, j].yaxis.set_ticks(np.arange(0, 1000+50, 50))
                ax[i, j].set_ylim(50, 500)
                if [i, j] != [0, 0]:
                    ax[i, j].set_xscale('log')
                else:
                    ax[i, j].set_xlim(0,100)
            else:
                ax[i, j].set_ylim(50, 500)
                # ax[i, j].set_ylim(0.1, 10)
                # ax[i, j].set_yscale('log')

            ax[i, j].grid(True)
            ax[i, j].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                              length=Tick_Length)
            ax[i, j].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                              width=Tick_Width_minor, length=Tick_Length_minor)
            ax[i, j].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                              length=Tick_Length)
            ax[i, j].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                              width=Tick_Width_minor, length=Tick_Length_minor)
            if i == 1 and j ==1:
                ax[i, j].legend(fontsize=Legend_fontSize, loc='upper left')
            else:
                ax[i, j].legend(fontsize=Legend_fontSize, loc='upper right')
    plt.tight_layout()
    file_name = rf'{ionizationRecombToggles.outputFolder}\testScripts\MODEL_heightIonization.png'
    plt.savefig(file_name, dpi=dpi)






# --- EXECUTE ---
verify_Fang2008_figures()