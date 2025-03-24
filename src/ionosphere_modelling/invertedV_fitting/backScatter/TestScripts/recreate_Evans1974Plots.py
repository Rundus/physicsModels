
# --- Imports ---
import matplotlib.pyplot as plt
import numpy as np
from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
from src.physicsModels.invertedV_fitting.backScatter.backScatter_classes import *
import spaceToolsLib as stl
from copy import deepcopy
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

#################
# --- TOGGLES ---
#################
show_Fig2_Fig3_backScatterCurves = False
show_Fig4_curveUsageSpectrums = False
show_Fig5_modelMaxwellianInvertedV_noBackscatter = False
show_Fig6_beamAngularWidths = False
compare_ParallelFLux_calculation = True
show_Fig5_modelMaxwellianInvertedV_withBackscatter = False


##################
# --- Plotting ---
##################
Label_fontsize = 20
Title_FontSize = 35
Tick_LabelSize = 60
Tick_SubplotLabelSize = 15
Tick_Width = 2
Tick_Length = 4
Plot_LineWidth = 3
Legend_Fontsize = 15

if show_Fig2_Fig3_backScatterCurves:
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 8)



    # --- Backscatter Curve ---
    model = Evans1974()
    xData, yData = model.Energy_BackScatter, model.NFlux_up_PeriE_BackScatter
    ax[0].set_title('Backscatter', fontsize=Title_FontSize)
    ax[0].plot(xData, yData, color='tab:red', linewidth=Plot_LineWidth,label='Digitized')
    ax[0].set_ylim(1E-10, 1E-3)
    ax[0].set_xlim(1E-2, 2)
    ax[0].set_ylabel('Upgoing Flux per Incident Electron\n $[E(Incident)/10000 ] \cdot [cm^{-2}sec^{-2}eV^{-1}$]', fontsize=Label_fontsize)
    ax[0].set_xlabel('E(Backscatter)/E(Incident)', fontsize=Label_fontsize)

    # plot image
    yLim = [-10, -3]
    xLim = [-2, np.log10(2)/np.log10(10)]
    imageFile = r'C:\Users\cfelt\PycharmProjects\physicsModels\src\physicsModels\invertedV_fitting\BackScatter\Evans_Model\degradedPrimaries.PNG'
    image = plt.imread(imageFile)
    dim1 = len(image)
    dim2 = len(image[0])
    X, Y = np.meshgrid(np.logspace(*xLim, base=10, num=dim2), np.logspace(*yLim, base=10, num=dim1))
    ax[0].pcolormesh(np.flip(X), Y, np.flip(image[:, :, 2]), cmap='gray',label='Evans1974')
    ax[0].legend(fontsize=Legend_Fontsize)

    # --- Secondaries Curve ---
    model = Evans1974()
    xData, yData = model.Energy_secondaries, model.NFlux_up_PeriE_secondaries
    ax[1].set_title('Secondaries', fontsize=Title_FontSize)
    ax[1].plot(xData, yData, color='tab:red', linewidth=Plot_LineWidth,label='Digitized')
    ax[1].set_ylabel('Upgoing Flux per Incident Electron \n $[cm^{-2}sec^{-2}eV^{-1}$]', fontsize=Label_fontsize)
    ax[1].set_xlabel('Energy (eV)', fontsize=Label_fontsize)
    ax[1].set_ylim(1E-8, 1E-1)
    ax[1].set_xlim(1E1, 1E3)
    ax[1].legend(fontsize=Legend_Fontsize)

    # image
    yLim = [-8, -1]
    xLim = [1, 3]
    imageFile = r'C:\Users\cfelt\PycharmProjects\physicsModels\src\physicsModels\invertedV_fitting\BackScatter\Evans_Model\secondaryies.PNG'
    image = plt.imread(imageFile)
    dim1 = len(image)
    dim2 = len(image[0])
    X, Y = np.meshgrid(np.logspace(*xLim, base=10, num=dim2), np.logspace(*yLim, base=10, num=dim1))
    ax[1].pcolormesh(np.flip(X), Y, np.flip(image[:, :, 2]), cmap='gray',label='Evans1974')


    for i in range(2):
        ax[i].tick_params(axis='y', which='major', labelsize=Tick_SubplotLabelSize + 4, width=Tick_Width, length=Tick_Length)
        ax[i].tick_params(axis='y', which='minor', labelsize=Tick_SubplotLabelSize, width=Tick_Width, length=Tick_Length / 2)
        ax[i].tick_params(axis='x', which='major', labelsize=Tick_SubplotLabelSize, width=Tick_Width, length=Tick_Length)
        ax[i].tick_params(axis='x', which='minor', labelsize=Tick_SubplotLabelSize, width=Tick_Width, length=Tick_Length / 2)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig(r'C:\Data\physicsModels\invertedV\backScatter\testScripts\Evans_Fig2_Fig3_curves.png')

if show_Fig4_curveUsageSpectrums:

    E_values = [610, 1500, 7550, 10000]

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_ylim(1E-8,1E-2)
    ax.set_xscale('log')
    ax.set_xlim(1E1, 2E4)
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

    # PLOT THE IMAGE
    yLim = [-8, -2]
    xLim = [1, np.log10(2E4)/np.log10(10)]
    imageFile0deg = r'C:\Users\cfelt\PycharmProjects\physicsModels\src\physicsModels\invertedV_fitting\BackScatter\Evans_Model\Evans_fig4.PNG'
    image = plt.imread(imageFile0deg)
    dim1 = len(image)
    dim2 = len(image[0])
    X, Y = np.meshgrid(np.logspace(*xLim, base=10, num=dim2), np.logspace(*yLim, base=10, num=dim1))
    ax.pcolormesh(np.flip(X), Y, np.flip(image[:, :, 2]), cmap='gray')
    plt.savefig(r'C:\Data\physicsModels\invertedV\backScatter\testScripts\Evans_Fig4_curveUsage.png')

if show_Fig5_modelMaxwellianInvertedV_noBackscatter:

    # --- Re-Create the Model Beam ---
    model_n = 1.5  # cm^-3
    model_T = 800  # eV
    model_V0 = 2000  # V

    # create a model beam at 0deg pitch
    N = 2000
    model_energyGrid = np.linspace(0, 1E4, N) # models the energies of the non-acceleration maxwellian
    beam_energyGrid = np.linspace(0, 1E4, N) + model_V0  # models the energies of the beam
    distributionFunc = distributions_class().generate_Maxwellian_Espace(n=model_n,
                                                                        T=model_T,
                                                                        energy_Grid=model_energyGrid)

    jN = distributions_class().calc_diffNFlux_Espace(dist=distributionFunc,
                                                     energy_Grid=beam_energyGrid)

    # --- calculate model at ARBITARY pitch ---
    # Description: Evans 1974 pointed out two parts to the beam pitch angle:
    # (1) The beam exiting the inverted-V will be collimated by alpha = arcsin(sqrt(E/(E + V0)))
    # (2) Magnetic mirroring effects will also widen the beam
    # At an arbitrary altitude the beam will widen due to (2), thus the beam itself may not be visible at certain eneriges for a given pitch angle
    # e.g. at low energies, the beam is really collimated, so low energies may not show up at ~60deg for a given altitude
    model_ZV = 2000  # inverted-V altitude (in km)
    model_Zatm = 100 # atmosphere altitude (in km)
    targetPitch = 85 # in deg

    # get the maximum pitch angles of the beam for a given energy
    alpha_m = (180/np.pi)*np.arcsin(np.sqrt(model_energyGrid/(model_energyGrid + model_V0)))

    # determine the maximum pitch angle at my chosen altitude via magnetic mirroring
    beta = np.power((stl.Re + model_ZV)/(stl.Re + model_Zatm),3)

    # pitch angle at the atmosphere of beam electron which had the highest pitch for a given energy
    alpha_atm = np.degrees(np.arcsin(np.sqrt(beta)*np.sin(np.radians(alpha_m)))  )
    alpha_atm = np.nan_to_num(alpha_atm, nan=90)

    # Initial pitch angle to required to reach 90deg at Zatm
    alpha_M_star = np.degrees(np.arcsin(1/np.sqrt(beta)))

    # for my specified target pitch angle, modified the beam to only allow electrons that have widened enough to reach that point
    # BE CAREFUL: alpha_m corresponds to the BEAM pitch angles, NOT the energy grid pitch angles. Account for this
    jN_targetPitch = deepcopy(jN)
    jN_targetPitch[np.where(alpha_atm<targetPitch)[0]] = 0

    # for i in range(len(alpha_m)):
    #     engyVal = str(round(beam_energyGrid[i]))
    #     startingPitch = str(round(alpha_m[i]))
    #     endPitch = round(alpha_atm[i])
    #     condition = endPitch >= targetPitch
    #     endPitch = str(endPitch)
    #     condition = str(condition)
    #     print(f"{''+engyVal:<5} {''+startingPitch:<5}{''+endPitch:<5}{''+condition:<5}")

    # Plot Everything
    fig, ax = plt.subplots(ncols=2)
    fig.suptitle('Observations at 100 km')

    # 0 Deg pitch
    ax[0].plot(beam_energyGrid, jN, label=r'$\alpha=0^{\circ}$',linewidth=Plot_LineWidth)

    # target Pitch Angle
    ax[1].plot(beam_energyGrid,jN_targetPitch, label=rf'$\alpha$={targetPitch}' r'$^{\circ}$', color='red',linewidth=Plot_LineWidth)

    for i in range(2):
        ax[i].grid(alpha=0.7,which='both')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1E3, 1E7)
        ax[i].set_xscale('log')
        ax[i].set_xlim(1E1, 2E4)
        ax[i].set_xlabel('Energy [eV]')
        ax[i].set_ylabel('Directional Flux [cm$^{-2}$sec$^{-1}$eV$^{-1}$str$^{-1}$]')
        ax[i].legend()

    plt.show()

if compare_ParallelFLux_calculation:

    # EVANS 1974 reported a paper with
    # beta = 1 , Number Flux = 7.1E8
    # beta = 2 , Number Flux = 1.37E9


    # --- Re-Create the Model Beam ---
    model_n = 1.5  # cm^-3
    model_T = 800  # eV
    model_V0 = 2000  # V

    # create a model beam at 0deg pitch
    N = 2000
    model_energyGrid = np.linspace(0, 1E4, N)  # models the energies of the non-acceleration maxwellian
    beam_energyGrid = np.linspace(0, 1E4, N) + model_V0  # models the energies of the beam
    distributionFunc = distributions_class().generate_Maxwellian_Espace(n=model_n,
                                                                        T=model_T,
                                                                        energy_Grid=model_energyGrid)

    jN = distributions_class().calc_diffNFlux_Espace(dist=distributionFunc,
                                                     energy_Grid=beam_energyGrid)

    # --- calculate model at ARBITARY pitch ---
    # Description: Evans 1974 pointed out two parts to the beam pitch angle:
    # (1) The beam exiting the inverted-V will be collimated by alpha = arcsin(sqrt(E/(E + V0)))
    # (2) Magnetic mirroring effects will also widen the beam
    # At an arbitrary altitude the beam will widen due to (2), thus the beam itself may not be visible at certain eneriges for a given pitch angle
    # e.g. at low energies, the beam is really collimated, so low energies may not show up at ~60deg for a given altitude
    model_ZV = 2000  # inverted-V altitude (in km)
    model_Zatm = 270  # atmosphere altitude (in km)
    targetPitch = 85  # in deg

    # get the maximum pitch angles of the beam for a given energy
    alpha_m = (180 / np.pi) * np.arcsin(np.sqrt(model_energyGrid / (model_energyGrid + model_V0)))

    beta = np.power((stl.Re + model_ZV) / (stl.Re + model_Zatm), 3)

    for betaVal in [1, beta]:
        # determine the maximum pitch angle at my chosen altitude via magnetic mirroring

        # pitch angle at the atmosphere of beam electron which had the highest pitch for a given energy
        alpha_atm = np.degrees(np.arcsin(np.sqrt(betaVal) * np.sin(np.radians(alpha_m))))
        alpha_atm = np.nan_to_num(alpha_atm, nan=90)

        # Initial pitch angle to required to reach 90deg at Zatm
        alpha_M_star = np.degrees(np.arcsin(1 / np.sqrt(betaVal)))

        # for my specified target pitch angle, modified the beam to only allow electrons that have widened enough to reach that point
        # BE CAREFUL: alpha_m corresponds to the BEAM pitch angles, NOT the energy grid pitch angles. Account for this
        jN_targetPitch = deepcopy(jN)
        jN_targetPitch[np.where(alpha_atm < targetPitch)[0]] = 0

        # --- diffNUMBER FLUX ---
        # Description: calculate the PARALLEL number flux by integrating over pitch angle.
        # Due to the inherent isotropy of jN, we only need to calculate: jN*pi*sin^(Gamma(E))
        # where Gamma(e) = alpha_m if alpha_m<alpha_M_star else arcsin(1/beta^1/2)
        # See my notebook for details.
        if betaVal != 1:
            Gamma = np.array([alpha_atm[i] if alpha_m[i] < alpha_M_star else 90 for i in range(len(alpha_atm))])
            varPhi_E = np.pi * jN * np.power(np.sin(np.radians(Gamma)), 2)
        else:
            varPhi_E = np.pi * jN * np.power(np.sin(np.radians(alpha_m)), 2)

        # --- number flux ---
        parallelFlux = simpson(x=beam_energyGrid, y=varPhi_E)
        print(f"Beta: {round(betaVal,1):<5} {'Flux [cm^-2s^-1]: '+'{:.2E}'.format(parallelFlux):<5}")

if show_Fig5_modelMaxwellianInvertedV_withBackscatter:

    ##################################
    # --- Re-Create the Model Beam ---
    ##################################
    model_n = 1.5  # cm^-3
    model_T = 800  # eV
    model_V0 = 2000  # eV

    # determine the maximum pitch angle at my chosen altitude via magnetic mirroring
    # beta = np.power((stl.Re + model_ZV) / (stl.Re + model_Zatm), 3)
    model_Zatm = 270
    beta = 2
    targetPitch = 45

    # create a model beam at 0deg pitch
    N = backScatterToggles.N_energyGrid
    model_energyGrid = np.linspace(10, 1E4, N)  # models the energies of the non-acceleration maxwellian
    beam_energyGrid = np.linspace(10, 1E4, N) + model_V0  # models the energies of the beam
    distributionFunc = distributions_class().generate_Maxwellian_Espace(n=model_n,
                                                                        T=model_T,
                                                                        energy_Grid=model_energyGrid)

    jN = distributions_class().calc_diffNFlux_Espace(dist=distributionFunc,
                                                     energy_Grid=beam_energyGrid)

    ############################################
    # --- Calculate the Ionospheric Response ---
    ############################################
    beam_num_flux, dgdPrim_num_flux, sec_num_flux = backScatter_class().calcIonosphericResponse(
        beta=beta,
        V0=model_V0,
        response_energy_grid=deepcopy(model_energyGrid),
        beam_energy_grid=deepcopy(beam_energyGrid),
        beam_jN = deepcopy(jN)
    )

    # --- determine the jN profile at various pitch angles ---
    # 0 deg
    dgdPrim_Total_0deg, sec_Total_0deg,jN_0deg = backScatter_class().calc_response_at_target_pitch(
        V0=model_V0,
        beta=beta,
        beam_jN=deepcopy(jN),
        beam_energy_grid = beam_energyGrid,
        sec_num_flux=sec_num_flux,
        dgdPrim_num_flux=dgdPrim_num_flux,
        energy_grid=model_energyGrid,
        target_pitch=0
    )

    # --- targetPitch ---
    dgdPrim_Total_targetPitch, sec_Total_targetPitch, jN_targetPitch = backScatter_class().calc_response_at_target_pitch(
        V0=model_V0,
        beta=beta,
        beam_jN=deepcopy(jN),
        beam_energy_grid=beam_energyGrid,
        sec_num_flux=sec_num_flux,
        dgdPrim_num_flux=dgdPrim_num_flux,
        energy_grid=model_energyGrid,
        target_pitch=targetPitch
    )

    ##################
    # --- PLOTTING ---
    ##################
    # Plot Everything
    fig, ax = plt.subplots(ncols=2)
    fig.suptitle(f'Observations at {model_Zatm} km\n '
                 f'N-Iterations: {backScatterToggles.niterations_backscatter}\n'
                 f'N-Energy GridPoint: {backScatterToggles.N_energyGrid}')
    fig.set_size_inches(16, 8)
    yLim = [4, 7]
    xLim = [1, 4]

    # --- 0 Deg pitch ---
    ax[0].plot(beam_energyGrid, jN_0deg, color='tab:red',label=r'$\alpha=0^{\circ}$', linewidth=Plot_LineWidth)
    ax[0].plot(model_energyGrid, dgdPrim_Total_0deg, color='tab:green', linewidth=Plot_LineWidth, label='Degraded Prim.')
    ax[0].plot(model_energyGrid, sec_Total_0deg, color='tab:blue', linewidth=Plot_LineWidth, label='Secondaries')
    ax[0].plot(model_energyGrid, (sec_Total_0deg+dgdPrim_Total_0deg), color='tab:orange', linewidth=Plot_LineWidth, label='Total Response')

    # --- target Pitch Angle ---
    ax[1].plot(beam_energyGrid, jN_targetPitch, label=rf'$\alpha$={targetPitch}' r'$^{\circ}$', color='tab:red', linewidth=Plot_LineWidth)
    ax[1].plot(model_energyGrid, dgdPrim_Total_targetPitch, color='tab:green', linewidth=Plot_LineWidth, label='Degraded Prim.')
    ax[1].plot(model_energyGrid, sec_Total_targetPitch, color='tab:blue', linewidth=Plot_LineWidth, label='Secondaries')
    ax[1].plot(model_energyGrid, (sec_Total_targetPitch + dgdPrim_Total_targetPitch), color='tab:orange', linewidth=Plot_LineWidth, label='Total Response')

    # IMAGES
    imageFile0deg = r'C:\Users\cfelt\PycharmProjects\physicsModels\src\physicsModels\invertedV_fitting\BackScatter\Evans_Model\EvansOutput_0deg.PNG'
    image = plt.imread(imageFile0deg)
    dim1 = len(image)
    dim2 = len(image[0])
    X, Y = np.meshgrid(np.logspace(*xLim,base=10,num=dim2), np.logspace(*yLim,base=10,num=dim1))
    ax[0].pcolormesh(np.flip(X),Y,np.flip(image[:,:,2]),cmap='gray')

    imageFile45deg = r'C:\Users\cfelt\PycharmProjects\physicsModels\src\physicsModels\invertedV_fitting\BackScatter\Evans_Model\EvansOutput_45deg.PNG'
    image = plt.imread(imageFile45deg)
    dim1 = len(image)
    dim2 = len(image[0])
    X, Y = np.meshgrid(np.logspace(*xLim, base=10, num=dim2), np.logspace(*yLim, base=10, num=dim1))
    ax[1].pcolormesh(np.flip(X), Y, np.flip(image[:, :, 2]),cmap='gray')

    for i in range(2):
        ax[i].grid(alpha=0.7, which='both')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1E4, 1E7)
        ax[i].set_xscale('log')
        ax[i].set_xlim(1E1, 1E4)
        ax[i].set_xlabel('Energy [eV]')
        ax[i].set_ylabel('Directional Flux [cm$^{-2}$sec$^{-1}$eV$^{-1}$sr$^{-1}$]')
        ax[i].legend(fontsize=Legend_Fontsize)

    plt.savefig(rf'C:\Data\physicsModels\invertedV\backScatter\testScripts\Evans1974_Fig5_compare.png')






