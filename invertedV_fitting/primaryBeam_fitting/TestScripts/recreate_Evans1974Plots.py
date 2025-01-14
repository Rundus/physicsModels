
# --- Imports ---
import matplotlib.pyplot as plt
import numpy as np
from invertedV_fitting.primaryBeam_fitting.Evans_Model.parameterizationCurves_Evans1974_classes import Evans1974
import spaceToolsLib as stl

#TODO: Something isn't correct with the accelerated Maxwellian Example. It's TOO SMALL

#################
# --- TOGGLES ---
#################
show_Evans1974_reScaledCurves = False
show_Evans1974_Total_energy_flux_spectrums = False
recreate_Evans1974_Fig5 = True


##################
# --- Plotting ---
##################

if show_Evans1974_reScaledCurves:
    Label_fontsize = 25
    Title_FontSize = 40
    tick_LabelSize = 60
    tick_SubplotLabelSize = 15
    tick_Width = 2
    tick_Length = 4
    plot_LineWidth = 3

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 8)
    model = Evans1974()
    xData, yData = model.Energy_DegradedPrimary, model.NFlux_up_PeriE_DegradedPrimary
    ax[0].set_title('Primaries Backscatter', fontsize=Title_FontSize)
    ax[0].plot(xData, yData, color='black', linewidth=plot_LineWidth)
    ax[0].set_yscale('log')
    ax[0].set_ylim(1E-10, 1E-3)
    ax[0].set_xscale('log')
    ax[0].set_xlim(1E-2, 1)
    ax[0].set_ylabel('Upgoing Flux per Incident Electron\n $[E(Incident)/10000 ] \cdot [cm^{-2}sec^{-2}eV^{-1}$]',
                     fontsize=Label_fontsize)
    ax[0].set_xlabel('E(Backscatter)/E(Incident)', fontsize=Label_fontsize)

    model = Evans1974()
    xData, yData = model.Energy_secondaries, model.NFlux_up_PeriE_secondaries
    ax[1].set_title('Secondaries', fontsize=Title_FontSize)
    ax[1].plot(xData, yData, color='black', linewidth=plot_LineWidth)
    ax[1].set_yscale('log')
    ax[1].set_ylim(1E-8, 1E-1)
    ax[1].set_xscale('log')
    ax[1].set_xlim(1E1, 1E3)
    ax[1].set_ylabel('Upgoing Flux per Incident Electron \n $[cm^{-2}sec^{-2}eV^{-1}$]', fontsize=Label_fontsize)
    ax[1].set_xlabel('Energy (eV)', fontsize=Label_fontsize)

    for i in range(2):
        ax[i].tick_params(axis='y', which='major', labelsize=tick_SubplotLabelSize + 4, width=tick_Width,
                          length=tick_Length)
        ax[i].tick_params(axis='y', which='minor', labelsize=tick_SubplotLabelSize, width=tick_Width,
                          length=tick_Length / 2)
        ax[i].tick_params(axis='x', which='major', labelsize=tick_SubplotLabelSize, width=tick_Width,
                          length=tick_Length)
        ax[i].tick_params(axis='x', which='minor', labelsize=tick_SubplotLabelSize, width=tick_Width,
                          length=tick_Length / 2)

    plt.tight_layout()
    plt.show()

if show_Evans1974_Total_energy_flux_spectrums:

    E_values = [610, 1500, 7550, 10000]

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_ylim(1E-8,1E-2)
    ax.set_xscale('log')
    ax.set_xlim(1E1,2E4)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Flux (Upgoing)\n Electrons cm$^{-2}$sec$^{-1}$eV$^{-1}$ per Incident Electron')

    for eVal in E_values:
        model = Evans1974()

        energyRange = np.linspace(10,1E4,1000)

        # get the secondaries data
        spline = model.generate_SecondariesCurve()
        yData_secondaries = spline(energyRange)
        yData_secondaries[np.where(energyRange > eVal)[0]] = 0
        yData_secondaries[np.where(energyRange>1000)[0]]=0

        # get the degraded primaries data
        spline = model.generate_BackScatterCurve(eVal)
        yData_degradedPrimaries = spline(energyRange)
        yData_degradedPrimaries[np.where(energyRange<eVal*1E-2)[0]] = 0
        yData_degradedPrimaries[np.where(energyRange > eVal)[0]] = 0

        # plot everything
        ax.plot(energyRange, yData_secondaries+yData_degradedPrimaries,label=f'{eVal}')

    ax.legend()
    plt.show()

if recreate_Evans1974_Fig5:

    # --- Re-Create the Model Beam ---
    # maxwellian parameters
    model_n = 1.5  # cm^-3
    model_T = 800  # eV
    model_V0 = 2000  # V
    model_vPos = 2000  # inverted-V altitude (in km)

    def acceleratedMaxwellain(x, n0, T, V0):

        return stl.q0/(np.power(stl.cm_to_m,2))*((1E6)*n0/(2*np.power(np.pi,3/2))) * (stl.q0*(x + V0)/np.power(stl.q0*T,3/2)) * np.sqrt(2/stl.m_e) * np.exp(-x/T)

    energyRange = np.linspace(0, 1E4, 5000)
    yData_beam = acceleratedMaxwellain(energyRange,model_n,model_T,model_V0)
    yData_beam[np.where(energyRange<model_V0)[0]]=0


    # for each energy value in the beam, produce a reponse curve and add them
    responseCurve = np.zeros(shape=(len(energyRange)))
    beamEnergies = energyRange[np.where(energyRange>0)[0]]
    for eVal in beamEnergies:

        # find the idx of the omniflux for this energy
        omniFluxIdx = np.abs(energyRange - eVal).argmin()

        # get the secondaries data
        model = Evans1974()
        spline = model.generate_SecondariesCurve()
        yData_secondaries = spline(energyRange)
        yData_secondaries[np.where(energyRange > eVal)[0]] = 0
        yData_secondaries[np.where(energyRange > 1000)[0]] = 0
        yData_secondaries[np.where(energyRange > model_V0)[0]] = 0

        # get the degraded primaries data
        spline = model.generate_BackScatterCurve(eVal)
        yData_degradedPrimaries = spline(energyRange)
        yData_degradedPrimaries[np.where(energyRange < eVal * 1E-2)[0]] = 0
        yData_degradedPrimaries[np.where(energyRange > eVal)[0]] = 0
        yData_degradedPrimaries[np.where(energyRange > model_V0)[0]] = 0

        responseCurve+= 2*np.pi*yData_beam[omniFluxIdx]*(yData_secondaries+yData_degradedPrimaries)



    # Plot Everything
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_ylim(1E3, 1E7)
    ax.set_xscale('log')
    ax.set_xlim(1E1, 2E4)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Electrons cm$^{-2}$sec$^{-1}$eV$^{-1}$')

    ax.plot(energyRange,yData_beam)
    ax.plot(energyRange,responseCurve)
    plt.show()




