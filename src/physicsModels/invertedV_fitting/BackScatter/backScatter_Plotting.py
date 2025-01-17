# --- backScatter_Plotting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: get the data from the backscatter Curves and plot the data WITHOUT regenerating all the curves again

from src.physicsModels.invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import *
import spaceToolsLib as stl
import matplotlib.pyplot as plt

# TODO: FIgure out why /np.pi isn't giving correct result WHEN IT SHOULD. Have to divide by pi to get sr-1 back since we're assuming isotropy
# TODO: so that J_N = varPHI(E)/pi

#################
# --- TOGGLES ---
#################
makeIndividualPlts = True
norm = 1 # SHOULD be np.pi

##################
# --- PLOTTING ---
##################
Figure_width = 8 # in inches
Figure_height = 6  # in inches
Title_FontSize = 20
Label_FontSize = 20
Label_Padding = 8
Text_FontSize = 20

Tick_FontSize = 22
Tick_Length = 3
Tick_Width = 1.5
Tick_FontSize_minor = 15
Tick_Length_minor = 1
Tick_Width_minor = 1
Plot_LineWidth = 1.5
plot_MarkerSize = 12
Legend_FontSize = 14
dpi = 200

def generateBackScatterPlots(GenToggles, **kwargs):

    ##########################
    # --- --- --- --- --- ---
    # --- LOADING THE DATA ---
    # --- --- --- --- --- ---
    ##########################
    data_dict_diffFlux = stl.loadDictFromFile(inputFilePath=GenToggles.input_diffNFiles[GenToggles.wFlyerFit])
    data_dict = stl.loadDictFromFile(inputFilePath=rf'C:\Data\physicsModels\invertedV\backScatter\ionosphericResponse.cdf')
    data_dict_beamFits = stl.loadDictFromFile(inputFilePath=r"C:\Data\physicsModels\invertedV\primaryBeam_Fitting\primaryBeam_fitting_parameters.cdf")
    EpochFitData, diffNFlux_avg, stdDev_avg = helperFuncs().groupAverageData(data_dict_diffFlux=data_dict_diffFlux,
                                                                             GenToggles=GenToggles,
                                                                             N_avg=primaryBeamToggles.numToAverageOver)

    def plotIndividualBackScatters(data_dict):

        energyGrid = data_dict['energy_Grid'][0]
        Epoch = data_dict['Epoch'][0]

        for tmeIdx in range(len(Epoch)):
            dPrim_Flux = data_dict['Phi_dPrim'][0][tmeIdx]
            secondaries_Flux = data_dict['Phi_sec'][0][tmeIdx]
            beam_Flux = data_dict['Phi_Beam'][0][tmeIdx]

            fig, ax = plt.subplots()
            fig.set_figwidth(Figure_width)
            fig.set_figheight(Figure_height)
            ax.set_title(r'$\alpha = 10^{\circ}$ - Primary Beam Only'+f'\n{Epoch[tmeIdx]} UTC', fontsize=Title_FontSize)

            # plot the raw data
            ax.plot(data_dict_diffFlux['Energy'][0], diffNFlux_avg[tmeIdx][2], 'o-', color='black')

            # plot the data which was fitted
            fittedIndicies = np.where(data_dict_beamFits['dataIdxs'][0][2][tmeIdx]>0)[0]
            ax.plot(data_dict_diffFlux['Energy'][0][fittedIndicies], diffNFlux_avg[tmeIdx][2][fittedIndicies], 'o-', color='tab:red')

            # plot the noise level
            ax.plot(helperFuncs().generateNoiseLevel(energyData=energyGrid,primaryBeamToggles=primaryBeamToggles),'--',color='black',label=f'{primaryBeamToggles.countNoiseLevel}-Count Noise')

            # plot the model beam reconstructed
            ax.plot(energyGrid, beam_Flux / norm, color='tab:red', label='Beam', linewidth=Plot_LineWidth)

            # plot the degraded primaries
            ax.plot(energyGrid, dPrim_Flux/norm, color='tab:red',label='Upward Degraded Prim.', linewidth=Plot_LineWidth)

            # plot the secondary flux
            ax.plot(energyGrid, secondaries_Flux / norm, color='tab:green', label='Upward Secondaries', linewidth=Plot_LineWidth)

            # plot the total response
            ax.plot(energyGrid, dPrim_Flux / norm + secondaries_Flux/norm , color='tab:blue', label='Total Response', linewidth=Plot_LineWidth)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('[cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$]', fontsize=Label_FontSize)
            ax.set_ylim(1E4, 5E7)
            ax.set_xlim(20, 3E3)
            ax.set_xlabel('Energy [eV]', fontsize=Label_FontSize)
            ax.legend(fontsize=Legend_FontSize)
            ax.grid(alpha=0.5)
            ax.tick_params(axis='y', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
            ax.tick_params(axis='x', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
            plt.savefig(rf'C:\Data\physicsModels\invertedV\backScatter\firstBounce\firstBounce_{tmeIdx}.png')




    if makeIndividualPlts:
        plotIndividualBackScatters(data_dict)