# --- betaFit.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: using the data from primaryBeam_fitting we can generate distributions at various altitudes
# to see which height our data most closely matches to

# --- --- --- ---
# --- IMPORTS ---
# --- --- --- ---
from src.physicsModels.my_Imports import *
from my_matplotlib_Assets.colorbars.apl_rainbow_black0 import apl_rainbow_black0_cmap
from myspaceToolsLib.physicsVariables import q0,m_e
from Science.InvertedV.Evans_class_var_funcs import velocitySpace_to_PitchEnergySpace, loadDiffNFluxData,mapping_VelSpace_magMirror,calc_velSpace_DistFuncDiffNFluxGrid
from Science.InvertedV.Evans_class_var_funcs import diffNFlux_for_mappedMaxwellian
from functools import partial
plt.rcParams["font.family"] = "Arial"
start_time = time.time()
# --- --- --- --- ---



#################
# --- TOGGLES ---
#################


# --- Model Distribution Toggles ---
useKaepplerData = False
N = 251 # velcoity space grid density
modifyInitialBeam = True
beamPitchThreshold = 30
beamEnergyThreshold = 400
cbarVmin, cbarVmax = 1E5, 5E6
beta_model = [6] # If I wanted distance values between 1-2 R_E, then beta should be between 3 to 24

# --- mapped distribution to different betas ---
Plot_modelDistribution = True

# --- beam widening fits for beta ---
Plot_beamWidthBetaFit = False
threshEngy = 200
betaIdxToPlot = 2 # Even though I'm trying to find beta, I still might want to plot it in order to compare

# --- beta pitch slice comparison plots ---
Plot_BetaSliceComparison = False

# --- Plot toggles - General ---
figure_width = 10 # in inches
figure_height =8 # in inches
Title_FontSize = 20
Label_FontSize = 20
Label_Padding = 8
Tick_FontSize = 12
Tick_Length = 1
Tick_Width = 1
Tick_FontSize_minor = 10
Tick_Length_minor = 1
Tick_Width_minor = 1
Plot_LineWidth = 0.5
plot_MarkerSize = 14
legend_fontSize = 15
mycmap = apl_rainbow_black0_cmap()

##########################
# --- LOADING THE DATA ---
##########################
prgMsg('Loading Data')
invertedV_TargetTimes_data = [[dt.datetime(2022,11,20,17,25,1,212210), dt.datetime(2022,11,20,17,25,1,212210)]]

# note: these values come from the pitch = 10deg fit
if useKaepplerData:
    # ACESI setup
    ACESI_EEPAA_path = r'C:\Data\ACESII\science\invertedV\ACESI_EEPAA.cdf'
    data_dict_ACESI = loadDictFromFile(ACESI_EEPAA_path)
    diffEFlux = data_dict_ACESI['diff_flux'][0]
    Epoch, Energy, Pitch = data_dict_ACESI['Epoch'][0], data_dict_ACESI['energy_cal'][0][0], data_dict_ACESI['PA_bin'][0][0]
    diffNFlux = deepcopy(diffEFlux)
    Energy = np.array(Energy)

    # Convert Kaeppler's diffEFlux to diffNFlux
    ranges = [range(len(diffEFlux)), range(len(diffEFlux[0])), range(len(diffEFlux[0][0]))]
    for tme, ptch, engy in itertools.product(*ranges):
        val = diffEFlux[tme][ptch][engy]
        if val >= 0:
            diffNFlux[tme][ptch][engy] = val/Energy[engy]

    EnergyBins = Energy
    PitchBins = np.array([-180 + i * 15 for i in range(24 + 1)])
    compareThesePitches = [2,4,6,8,10]
    paramSet = 0
else:
    diffNFlux, Epoch, Energy, Pitch = loadDiffNFluxData()
    EnergyBins = Energy
    compareThesePitches = [2,4,6,8,10]
    PitchBins = np.array([-180 + i * 10 for i in range(36 + 1)])
    paramSet = 2

modelParams = [['2009-01-29 09:54:57:673000', 1.25, 1150, 3400],# kappler
               ['2022-11-20 17:25:01:312207', 1.5, 800, 2000],  # Evans 1974
               ['2022-11-20 17:25:01:412211', 2.7, 124.6, 284.8] # OUR DATA: the GOOD SLICE
               ] # format: time, density [cm^-3], temperature [eV], [potential eV]

Done(start_time)


#################################

# --- --- --- --- --- --- --- ---
# --- GENERATE THE MODEL DATA ---
# --- --- --- --- --- --- --- ---
#################################
# remove the old files in the folder
for file in glob(r'C:\Data\ACESII\science\invertedV\BeamWidth\*.png*'):
    os.remove(file)

# collect the data
low_idx, high_idx = np.abs(Epoch - invertedV_TargetTimes_data[0][0]).argmin(), np.abs(Epoch - invertedV_TargetTimes_data[0][1]).argmin()
EpochFitData = Epoch[low_idx:high_idx + 1]

for tmeIdx in range(len(EpochFitData)):

    # --- --- --- --- --- --- --- --- --- --- --- ----
    # --- FIT THE 0DEG pitch data for model PARAMS ---
    # --- --- --- --- --- --- --- --- --- --- --- ----
    # get the real data at the timeslice
    diffNFluxSlice_real = np.array(diffNFlux[np.abs(Epoch - EpochFitData[tmeIdx]).argmin()])
    EngyIdx = np.abs(Energy - threshEngy).argmin()
    peakDiffNVal = diffNFluxSlice_real[2][:EngyIdx].max()
    peakDiffNVal_index = np.argmax(diffNFluxSlice_real[2][:EngyIdx])

    # get the subset of data to fit to and fit it. Only include data with non-zero points
    xData_fit = np.array(Energy[:peakDiffNVal_index + 1])
    yData_fit = np.array(diffNFluxSlice_real[2][:peakDiffNVal_index + 1])
    nonZeroIndicies = np.where(yData_fit != 0)[0]
    xData_fit = xData_fit[nonZeroIndicies]
    yData_fit = yData_fit[nonZeroIndicies]

    deviation = 0.18
    guess = [1.55, 20000000, 100]  # observed plasma at dispersive region is 0.5E5 cm^-3 BUT this doesn't make sense to use as the kappa fit since the kappa fit comes from MUCH less dense populations above
    boundVals = [[0.001, 30],  # n [cm^-3]
                 [10, 500],  # T [eV]
                 [(1 - deviation) * Energy[peakDiffNVal_index], (1 + deviation) * Energy[peakDiffNVal_index]]]  # V [eV]

    bounds = tuple([[boundVals[i][0] for i in range(len(boundVals))], [boundVals[i][1] for i in range(len(boundVals))]])
    fitFuncAtPitch = partial(diffNFlux_for_mappedMaxwellian, alpha=Pitch[2])
    params, cov = curve_fit(fitFuncAtPitch, xData_fit, yData_fit, maxfev=int(1E9), bounds=bounds)

    # Choose the input beam paramaters
    model_T = params[1]  # in eV
    model_n = params[0]  # in cm^-3
    model_V0 = params[2]

    # Containers to store data
    VparaGrids_iono_model = []
    VperpGrids_iono_model = []
    distFunc_model = []
    diffNFluxGrids_iono_model=[]
    Epoch_model =[dt.datetime.strptime(st[0], "%Y-%m-%d %H:%M:%S:%f") for st in modelParams]

    for betaChoice in beta_model:

        # --- Define a grid of intial velocities ---
        Vperp_gridVals = np.linspace(-np.sqrt(2*Energy.max()*q0/m_e), np.sqrt(2*Energy.max()*q0/m_e), N)
        Vpara_gridVals = np.linspace(0, np.sqrt(2*Energy.max()*q0/m_e), N)
        VperpGrid_Accel, VparaGrid_Accel, distGrid, diffNFluxGrid_Accel = calc_velSpace_DistFuncDiffNFluxGrid(Vperp_gridVals=Vperp_gridVals,
                                                                                                        Vpara_gridVals=Vpara_gridVals,
                                                                                                        model_Params=[model_n,model_T,model_V0],
                                                                                                        initalBeamParams=[beamPitchThreshold,beamEnergyThreshold])

        # --- Calculate the grids of the accelerate and mirror_mapped distributions ---
        prgMsg(rf'Creating Model Data for beta = {betaChoice}')
        targetAlt = 400
        betaValueAlt = Re*(np.power(betaChoice, 1/3) - 1) + targetAlt
        VperpGrid_mapped, VparaGrid_mapped, diffNFlux_mapped = mapping_VelSpace_magMirror(VperpGrid=VperpGrid_Accel,
                                                                                          VparaGrid=VparaGrid_Accel,
                                                                                          distFuncGrid=distGrid,
                                                                                          targetAlt=targetAlt,
                                                                                          startingAlt=betaValueAlt,
                                                                                          mapToMagSph=False)

        # Store the data for various beta values
        VperpGrids_iono_model.append(VperpGrid_mapped)
        VparaGrids_iono_model.append(VparaGrid_mapped)
        distFunc_model.append(distGrid)
        diffNFluxGrids_iono_model.append(diffNFlux_mapped)

        Done(start_time)

    #####################################
    # --- Distribution Function Plots ---
    #####################################
    if Plot_modelDistribution:

        # remove the old files in the folder
        for file in glob(r'C:\Data\ACESII\science\invertedV\betaMappingPlots\*.png*'):
            os.remove(file)


        for betaIdx,betaChoice in enumerate(beta_model):

            prgMsg('Creating the Plot')

            # --- Plot it ---
            fig, ax = plt.subplots(3, 2)
            fig.set_size_inches(figure_width, figure_height)

            titles = ['Plasma Sheet Model','Accelerated', 'Observed Ionosphere Model']
            fig.suptitle(rf'$\beta$ = {betaChoice}')
            for k in range(2):

                if k == 0:
                    grids = [[VperpGrid_Accel, VparaGrid_Accel, distGrid], [VperpGrids_iono_model[betaIdx], VparaGrids_iono_model[betaIdx], distGrid]]
                    vmin, vmax = 1E-22, 1E-14
                    cbarLabel = 'Distribution Function'

                else:
                    grids = [[VperpGrid_Accel, VparaGrid_Accel, diffNFluxGrid_Accel], [VperpGrids_iono_model[betaIdx], VparaGrids_iono_model[betaIdx], diffNFluxGrids_iono_model[betaIdx]]]
                    vmin, vmax = 1E2, 1E8
                    cbarLabel = 'diff_N_Flux'

                for i in [0, 1, 2]:
                    cmap = ax[i, k].pcolormesh(grids[i][0]/(1E7), grids[i][1]/(1E7), grids[i][2], cmap=mycmap, norm='log', vmin=vmin, vmax=vmax)
                    cbar = plt.colorbar(cmap, ax=ax[i,k])
                    cbar.set_label(cbarLabel)
                    ax[i, k].set_ylabel('Vpara')
                    ax[i, k].set_xlabel('Vperp')
                    ax[i, k].set_ylim(0, 8)
                    ax[i, k].set_xlim(-8, 8)
                    ax[i, k].invert_yaxis()
                    ax[i, k].set_title(titles[i])

                    if i in [1, 2]:
                        # plot the 110 deg line
                        ax[i, k].axhline(np.sqrt(2*model_V0*q0/m_e)/(1000*10000),color='red', label='$V_{0}$'+f'= {model_V0} eV')
                        ax[i, k].legend()
            plt.tight_layout()
            plt.savefig(rf'C:\Data\ACESII\science\invertedV\betaMappingPlots\BetaFit_{betaChoice}.png')
            plt.close()
            Done(start_time)

    ###########################################
    # --- BETA PITCH SLICE COMPARISON PLOTS ---
    ###########################################
    if Plot_BetaSliceComparison:
        prgMsg('Creating Beta Slice Comparison Plots')
        # remove the old files in the folder
        for file in glob(r'C:\Data\ACESII\science\invertedV\betaComparisonPlots\*.png*'):
            os.remove(file)


        for betaIdx in range(len(distFunc_model)):

            diffNFlux_model_pE, EnergyGrid, PitchGrid = velocitySpace_to_PitchEnergySpace(EnergyBins=EnergyBins,
                                                                                 PitchBins=PitchBins,
                                                                                 VperpGrid=VperpGrids_iono_model[betaIdx],
                                                                                 VparaGrid=VparaGrids_iono_model[betaIdx],
                                                                                 ZGrid=diffNFluxGrids_iono_model[betaIdx])


            # --- now create some comparison plots between the real data and your model ---

            # get the real data at the timeslice
            diffNFluxSlice = diffNFlux[np.abs(Epoch - Epoch_model[paramSet]).argmin()]

            for ptchIdxval in compareThesePitches:

                realData = diffNFluxSlice[ptchIdxval][:]
                closestPitch = np.abs(PitchBins-Pitch[ptchIdxval]).argmin()
                modelData = diffNFlux_model_pE[closestPitch][:]

                fig, ax = plt.subplots()
                fig.suptitle(rf'$\beta$={beta_model[betaIdx]}' + f'\n Pitch = {Pitch[ptchIdxval]}' + f'\n{Epoch_model[paramSet]}')
                ax.plot(Energy,realData,color='black',label='real Data',marker='.')
                ax.plot(EnergyBins,modelData,color='red',label='model',marker='.')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_ylim(1E4,1E7)
                ax.set_xlim(1E1,1E4)
                ax.legend()
                plt.savefig(rf'C:\Data\ACESII\science\invertedV\betaComparisonPlots\Beta_{beta_model[betaIdx]}_pitch_{Pitch[ptchIdxval]}.png')
                plt.close()

        outputPath = rf'C:\Data\ACESII\science\invertedV\BetaFit_data.cdf'
        Done(start_time)

    #########################################
    # --- BETA BEAM PITCH WIDENNING PLOTS ---
    #########################################
    if Plot_beamWidthBetaFit:
        prgMsg('Creating Beta Slice Comparison Plots')



        for i,betaIdxToPlot in enumerate(beta_model):

            # Get the model data (non-mirrored) in Pitch-Energy Space
            diffNFlux_model_pE, EnergyGrid, PitchGrid = velocitySpace_to_PitchEnergySpace(EnergyBins=EnergyBins,
                                                                                 PitchBins=PitchBins,
                                                                                 VperpGrid=VperpGrid_Accel,
                                                                                 VparaGrid=VparaGrid_Accel,
                                                                                 ZGrid=diffNFluxGrid_Accel,
                                                                                 method='average')

            diffNFlux_model_mirrored_pE, EnergyGrid_mirrored, PitchGrid_mirrored = velocitySpace_to_PitchEnergySpace(EnergyBins=EnergyBins,
                                                                                          PitchBins=PitchBins,
                                                                                          VperpGrid=VperpGrids_iono_model[i],
                                                                                          VparaGrid=VparaGrids_iono_model[i],
                                                                                          ZGrid=diffNFluxGrids_iono_model[i],
                                                                                          method='average')




            # For each pitch angle in the REAL data, get the peak in the diffNFlux after some threshold, exactly like the fitting procedure for diffNFit
            BeamEnergyChoice = []
            BeamPitchChoice = []
            for ptch in [1,2,3,4,5,6,7,8,9]: # PITCH INDICIES
                try:
                    EngyIdx = np.abs(Energy - threshEngy).argmin()
                    peakDiffNVal_index = np.argmax(diffNFluxSlice_real[ptch][:EngyIdx])
                    BeamEnergyChoice.append(Energy[peakDiffNVal_index])
                    BeamPitchChoice.append(Pitch[ptch])
                except:
                    print('no Data for Pitch Acquisition')


            # get the maximum angle for each pitch angle in the model beam
            BeamPitchModel_AllPitch = []
            BeamEnergyModel_AllEnergy = []
            searchData = diffNFlux_model_pE.T
            for engy in range(len(searchData)):
                if EnergyBins[engy] < beamEnergyThreshold:
                    pitchSlice = searchData[engy]

                    # find the largest pitch angle with a non-zero diffNFlux value. Search only positive pitch angles
                    try: # if data contains some non-zero data
                        nonZeroIndicies = np.nonzero(pitchSlice)
                        positivePitchMaxNonZeroFlux_Idx = nonZeroIndicies[0][-1]
                        BeamPitchModel_AllPitch.append(PitchBins[positivePitchMaxNonZeroFlux_Idx])
                        BeamEnergyModel_AllEnergy.append(EnergyBins[engy])
                    except: # if data is only zero
                        BeamPitchModel_AllPitch.append(-10000)
                        BeamEnergyModel_AllEnergy.append(EnergyBins[engy])

            # reduce the model data to only those shared by the real data
            BeamPitchModel, BeamEnergyModel = [], []
            for idx, pitchVal in enumerate(BeamPitchModel_AllPitch):
                if pitchVal >= -20 :
                    BeamPitchModel.append(BeamPitchModel_AllPitch[idx])
                    BeamEnergyModel.append(BeamEnergyModel_AllEnergy[idx])

            BeamEnergyModel, BeamPitchModel = list(zip(*sorted(zip(BeamEnergyModel, BeamPitchModel))))

            # convert things to numpy arrays
            BeamEnergyModel, BeamPitchModel = np.array(BeamEnergyModel), np.array(BeamPitchModel)
            BeamEnergyChoice, BeamPitchChoice = np.array(BeamEnergyChoice), np.array(BeamPitchChoice)


            # if the model data has less points than the real data, reduce the real data in order to compare them fot beta fit
            BeamPitchChosenValues = [     10,     30,     40,     50,     60]
            BeamEnergyChosenValues = [210.54, 245.74, 286.82, 286.82, 334.77]
            print(BeamPitchModel)
            print(BeamEnergyModel)


            if len(BeamEnergyModel) != len(BeamEnergyChosenValues):

                # find which energies they have in common
                commonEnergies = list(set(BeamEnergyChoice).intersection(BeamEnergyModel))
                engyIndicies = np.array([np.where(BeamEnergyChoice == engy)[0][0] for engy in commonEnergies])
                BeamEnergyChoice = BeamEnergyChoice[engyIndicies]
                BeamPitchChoice = BeamPitchChoice[engyIndicies]


            # # calculate the sin^2(alpha) values
            xData_betaFit = [np.power(np.sin(np.radians(val)), 2) for val in BeamPitchModel]
            yData_betaFit = [np.power(np.sin(np.radians(val)), 2) for val in BeamPitchChosenValues]

            # Linear fit the data
            def fitFuncLinear(x,A):
                return A*x
            params, cov = curve_fit(fitFuncLinear, xData_betaFit, yData_betaFit)
            xData_fitted = np.linspace(min(xData_betaFit),max(xData_betaFit),10)
            yData_fitted = fitFuncLinear(xData_fitted,*params)

            # --- --- --- --- --- ----
            # Make the Beam-Width Plot
            # --- --- --- --- --- ----
            fig, ax = plt.subplots(4)
            fig.set_size_inches(figure_width, 2*figure_height)

            # Accelerated Model Data
            fig.suptitle(f'Time {EpochFitData[tmeIdx]} UTC', fontsize=Title_FontSize)
            ax[0].set_title('Accelerated Model',fontsize=Label_FontSize)
            cmapObj=ax[0].pcolormesh(PitchGrid, EnergyGrid, np.array(diffNFlux_model_pE),vmin=cbarVmin, vmax=cbarVmax, norm='log',cmap=mycmap)
            ax[0].scatter(BeamPitchModel, BeamEnergyModel)
            ax[0].set_yscale('log')
            ax[0].set_ylim(90, 2000)
            ax[0].set_xlim(-15, 100)
            ax[0].set_ylabel('Energy', fontsize=Label_FontSize)
            cbar = plt.colorbar(cmapObj)

            # Mirrored Model Data
            ax[1].set_title(rf'Mirrored Model $\beta$ = {beta_model[i]}',fontsize=Label_FontSize)
            cmapObj = ax[1].pcolormesh(PitchGrid_mirrored, EnergyGrid_mirrored, np.array(diffNFlux_model_mirrored_pE),vmin=cbarVmin, vmax=cbarVmax, norm='log', cmap=mycmap)
            ax[1].scatter(BeamPitchChoice, BeamEnergyChoice, color='black')
            ax[1].set_yscale('log')
            ax[1].set_ylim(90, 2000)
            ax[1].set_xlim(-15, 100)
            ax[1].set_ylabel('Energy', fontsize=Label_FontSize)
            cbar = plt.colorbar(cmapObj)

            # Real Data + Beam Choice
            ax[2].set_title('Real Data',fontsize=Label_FontSize)
            cmapObj=ax[2].pcolormesh(Pitch, Energy, diffNFluxSlice_real.T,vmin=cbarVmin, vmax=cbarVmax, norm='log',cmap=mycmap)
            ax[2].scatter(BeamPitchChoice, BeamEnergyChoice, color='black')
            ax[2].set_yscale('log')
            ax[2].set_ylim(90, 2000)
            ax[2].set_xlim(-15, 100)
            ax[2].set_ylabel('Energy')
            ax[2].grid(alpha=0.5, which='minor', axis='both')
            cbar = plt.colorbar(cmapObj)

            # Beta fit from Beam Choice
            ax[3].plot(xData_fitted,yData_fitted,label=rf'$\beta =$ {params}')
            ax[3].scatter(xData_betaFit, yData_betaFit)
            ax[3].set_ylabel(r'$\sin^{2}(\alpha_{I})$',fontsize=Label_FontSize)
            ax[3].set_xlabel(r'$\sin^{2}(\alpha_{Model})$',fontsize=Label_FontSize)
            ax[3].legend()
            plt.tight_layout()
            plt.savefig(rf'C:\Data\ACESII\science\invertedV\BeamWidth\BeamWidthAnalysis_{tmeIdx}_{beta_model[i]}.png')


        Done(start_time)
