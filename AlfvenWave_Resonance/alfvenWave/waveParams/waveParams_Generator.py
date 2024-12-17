# --- imports ---
from ACESII_code.Science.Simulations.TestParticle.simToggles import m_to_km, R_REF, GenToggles, EToggles, runFullSimulation
from myspaceToolsLib.physicsVariables import lightSpeed,u0,m_e,ep0,cm_to_m,IonMasses,q0
from myspaceToolsLib.CDF_load import loadDictFromFile
from numpy import exp, sqrt, array, pi, abs, tanh
simulationAlt = GenToggles.simAlt

# TODO: import the ionosphere environment parameters and re-path the data to each of the wave parameter profiles

##################
# --- PLOTTING ---
##################
plotting = False
useTanakaDensity = False
xNorm = R_REF # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == R_REF else 'km'
plottingDict = {
                'lambdaPerp': True,
                'skinDepth': True,
                'alfSpdMHD': True,
                'kineticTerms': True,
                'lambdaPara': True,
                'alfSpdInertial': True}

# --- Output Data ---
outputData = True if not runFullSimulation else True

# get the geomagnetic field data dict
data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simOutputPath}\geomagneticField\geomagneticField.cdf')


def generatePlasmaEnvironment(outputData, **kwargs):
    plotting = kwargs.get('showPlot', False)


    # --- Plotting Toggles ---
    figure_width = 14  # in inches
    figure_height = 8.5  # in inches
    Title_FontSize = 25
    Label_FontSize = 25
    Tick_FontSize = 25
    Tick_FontSize_minor = 20
    Tick_Length = 10
    Tick_Width = 2
    Tick_Length_minor = 5
    Tick_Width_minor = 1
    Text_Fontsize = 20
    Plot_LineWidth = 2.5
    Legend_fontSize = 16
    dpi = 100


    # --- Kperp ---
    def lambdaPerpProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        Bgeo,Bgrad = data_dict_Bgeo['Bgeo'][0], data_dict_Bgeo['Bgrad'][0]
        initindex = abs(altRange - GenToggles.obsHeight).argmin() # the index of the startpoint of the Wave
        initBgeo = Bgeo[initindex] # <--- This determines where the scaling begins
        LambdaPerp = EToggles.lambdaPerp0*sqrt(initBgeo/Bgeo) if not EToggles.static_Kperp else array([EToggles.lambdaPerp0 for i in range(len(altRange))])
        kperp = 2*pi/LambdaPerp

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 1 + figure_height/2)
            ax.plot(altRange / xNorm, LambdaPerp/1000, linewidth=Plot_LineWidth, color='black',label='$\lambda_{\perp}$')
            ax.set_title('$\lambda_{\perp}$, $k_{\perp}$ vs Altitude \n'
                            '$\lambda_{\perp 0}$=' +f'{EToggles.lambdaPerp0/1000} km',fontsize=Title_FontSize)
            ax.set_ylabel('$\lambda_{\perp}$ [km]',fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axvline(x=400000 / xNorm, label='Observation Height', color='red',linewidth=Plot_LineWidth)
            ax.plot(altRange / xNorm, LambdaPerp/1000, linewidth=Plot_LineWidth, color='black', linestyle='--', label='$k_{\perp}$')
            ax.grid(True)
            ax.legend(fontsize=Legend_fontSize,loc='right')

            axKperp = ax.twinx()
            axKperp.plot(altRange / xNorm, kperp, linewidth=Plot_LineWidth,color='black',linestyle='--',label='$k_{\perp}$')
            axKperp.set_ylabel('$k_{\perp}$ [m$^{-1}$]',fontsize=Label_FontSize)
            axKperp.axvline(x=400000 / xNorm, label='Observation Height', color='red',linewidth=Plot_LineWidth)

            axes = [ax,axKperp]
            for axesE in axes:

                axesE.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                axesE.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)
                axesE.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                axesE.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)

            plt.subplots_adjust(left=0.1, bottom=0.2, right=0.85, top=0.82, wspace=None, hspace=None)
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_kperp.png',dpi=dpi)

        return LambdaPerp, kperp

    # --- SKIN DEPTH ---
    def skinDepthProfile(altRange,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaFreq = plasmaFreqProfile(altRange)
        LambdaPerp, kperp = lambdaPerpProfile(altRange)
        skinDepth = array([lightSpeed/plasmaFreq[i] for i in range(len(plasmaFreq))])

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 2.5+ figure_height/2)
            ax.plot(altRange/xNorm, skinDepth, color='blue',label='SkinDepth',linewidth=Plot_LineWidth)
            ax.plot(altRange / xNorm, LambdaPerp, color='black', label=r'$\lambda_{\perp}$',linewidth=Plot_LineWidth)
            ax.set_title('$\lambda_{e}$ vs Altitude\n' + '$\lambda_{\perp}$= ' + rf'{EToggles.lambdaPerp0}m', fontsize=Title_FontSize)
            ax.set_ylabel('Skin Depth [m]',fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axvline(x=400000/xNorm, label='Observation Height', color='red',linewidth=Plot_LineWidth)
            ax.set_yscale('log')
            ax.set_ylim(10, 1E5)
            ax.margins(0)
            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_skinDepth.png',dpi=dpi)

        return skinDepth

    # --- MHD Alfven Speed ---
    def MHD_alfvenSpeedProfile(altRange,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaDensity = plasmaDensityProfile(altRange)
        Bgeo,Bgrad = data_dict_Bgeo['Bgeo'][0], data_dict_Bgeo['Bgrad'][0]
        n_Op, n_Hp, m_eff_i = ionMassProfile(altRange)
        VA_MHD = array(Bgeo/sqrt(u0*m_eff_i*plasmaDensity))

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(altRange / xNorm, VA_MHD/(10000*m_to_km), label='$V_{A} (MHD)$',linewidth=Plot_LineWidth)
            ax.set_title(r'$V_{A}$ vs Altitude',fontsize=Title_FontSize)
            ax.set_ylabel('MHD Alfven Speed  [10,000 km/s]',fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axvline(x=400000 / xNorm, label='Observation Height', color='red', linestyle='--',linewidth=Plot_LineWidth)

            # plot some thermal velocity comparisons
            Vth_low = sqrt(8*q0*1/(9.11E-31))/(10000*m_to_km)
            ax.axhline(y=Vth_low, color='black',linewidth=Plot_LineWidth)
            ax.text(x=R_REF/xNorm,y=Vth_low*1.3,s='$V_{th_{e}}$ (1 eV)', color='black',fontsize=Text_Fontsize)

            Vth_high = sqrt(8 * q0 * 50 / (9.11E-31))/(10000*m_to_km)
            ax.axhline(y=Vth_high, color='black',linewidth=Plot_LineWidth)
            ax.text(x=R_REF/xNorm, y=Vth_high * 1.1, s='$V_{th_{e}}$ (50 eV)', color='black',fontsize=Text_Fontsize)

            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_alfMHD.png',dpi=dpi)

        return VA_MHD

    # --- 3 Kinetic Terms ---
    def kineticTermsProfiles(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        # collect profiles
        lambdaPerp, kperp = lambdaPerpProfile(altRange)
        ionCyclotron, ionCyclotron_Op, ionCyclotron_Hp = ionCyclotronProfile(altRange)
        ionLarmorRadius, ionLarmorRadius_Op, ionLarmorRadius_Hp = ionLarmorRadiusProfile(altRange)
        skinDepth = skinDepthProfile(altRange)
        alfSpdMHD = MHD_alfvenSpeedProfile(altRange)
        inertialTerm = 1 + (kperp*skinDepth)**2
        finiteFreqTerm = 1 - (EToggles.waveFreq_rad/ionCyclotron)**2
        LarmorTerm = 1 + (kperp*ionLarmorRadius)**2

        if plotBool:

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2, ncols=2,sharex=True)
            fig.set_size_inches(figure_width, figure_height)

            # Alfven Velocity
            ax[0, 0].plot(altRange / (xNorm), alfSpdMHD/(10000*m_to_km),linewidth=Plot_LineWidth)
            ax[0, 0].set_title('Alfven velocity (MHD)',fontsize=Title_FontSize)
            ax[0, 0].set_ylabel('Velocity [10,000 km/s]',fontsize=Label_FontSize)
            # ax[0, 0].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[0, 0].axvline(x=400000 / xNorm, label='Observation Height', color='red')
            ax[0, 0].set_ylim(0, 5)

            # inerital term
            ax[0, 1].plot(altRange / xNorm, inertialTerm,linewidth=Plot_LineWidth,label='$\lambda_{\perp}$ =' + f'{EToggles.lambdaPerp0} [m]')
            ax[0, 1].set_title('(1 + $k_{\perp}^{2}\lambda_{e}^{2}$)',fontsize=Title_FontSize)
            # ax[0, 1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[0, 1].axvline(x=400000 / xNorm, label='Observation Height', color='red')
            ax[0, 1].set_ylim(0, 3.5)
            ax[0,1].legend(fontsize=Legend_fontSize)

            # larmor radius term
            ax[1, 0].plot(altRange / xNorm, LarmorTerm,linewidth=Plot_LineWidth,label='$\lambda_{\perp}$ =' + f'{EToggles.lambdaPerp0} [m]')
            ax[1, 0].set_title(r'(1 + $k_{\perp}^{2}\rho_{i}^{2}$)',fontsize=Title_FontSize)
            # ax[1, 0].set_ylabel('Ion Larmor radius',fontsize=Label_FontSize)
            ax[1, 0].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[1, 0].axvline(x=400000 / xNorm, label='Observation Height', color='red')
            ax[1, 0].set_ylim(0, 3.5)
            ax[1,0].legend(fontsize=Legend_fontSize)

            # finite frequency term
            ax[1, 1].plot(altRange / xNorm, finiteFreqTerm,linewidth=Plot_LineWidth,label='$f_{wave}$ =' + f'{EToggles.waveFreq_Hz} [Hz]')
            ax[1, 1].set_title('(1 - $\omega^{2}/\omega_{ci}^{2}$)',fontsize=Title_FontSize)
            # ax[1, 1].set_ylabel('',fontsize=Label_FontSize)
            ax[1, 1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[1, 1].axvline(x=400000 / xNorm, label='Observation Height', color='red')
            ax[1, 1].set_ylim(0, 3.5)
            ax[1,1].legend(fontsize=Legend_fontSize)


            for i in range(2):
                for j in range(2):
                    ax[i,j].grid(True)
                    ax[i, j].margins(0)
                    ax[i,j].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
                    ax[i,j].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
                    ax[i,j].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
                    ax[i,j].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)

            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_kineticTerms.png',dpi=dpi)

        return inertialTerm, finiteFreqTerm, LarmorTerm

    # --- Lambda Parallel Wavelength ---
    def lambdaParallelProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        # collect profiles
        alfSpdMHD = MHD_alfvenSpeedProfile(altRange)
        inertialTerm, finiteFreqTerm, LarmorTerm = kineticTermsProfiles(altRange)
        LambdaPara = 2*pi*alfSpdMHD*sqrt(finiteFreqTerm)*sqrt(LarmorTerm)/(EToggles.waveFreq_rad*sqrt(inertialTerm))
        kpara = 2*pi/LambdaPara

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            ax[0].plot(altRange / xNorm, LambdaPara/m_to_km, linewidth=Plot_LineWidth)
            ax[0].set_title('$\lambda_{\parallel}$ vs Altitude',fontsize=Title_FontSize)
            ax[0].set_ylabel('$\lambda_{\parallel}$ [km]',fontsize=Label_FontSize)
            # ax[0].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='red',linewidth=Plot_LineWidth)
            ax[0].legend(fontsize=Legend_fontSize)

            ax[1].plot(altRange / xNorm, kpara, linewidth=Plot_LineWidth)
            ax[1].set_title('$k_{\parallel}$ vs Altitude',fontsize=Title_FontSize)
            ax[1].set_ylabel('$k_{\parallel}$ [m$^{-1}$]',fontsize=Label_FontSize)
            ax[1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red',linewidth=Plot_LineWidth)
            ax[1].legend(fontsize=Legend_fontSize)
            for i in range(2):
                ax[i].grid(True)
                ax[i].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
                ax[i].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
                ax[i].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
                ax[i].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)

            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_lambdaPara.png', dpi=dpi)

        return LambdaPara,kpara

    # --- Kinetic Alfven Speed ---
    def Intertial_alfvenSpeedProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        # collect profiles
        inertialTerm, finiteFreqTerm, LarmorTerm = kineticTermsProfiles(altRange)
        alfSpdMHD = MHD_alfvenSpeedProfile(altRange)
        kineticAlfSpeed = alfSpdMHD * sqrt(finiteFreqTerm) *sqrt(LarmorTerm)/sqrt(inertialTerm)

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(altRange / xNorm, kineticAlfSpeed/(10000*m_to_km), label='DAW Speed', color='blue',linewidth=Plot_LineWidth)
            ax.set_title(r'$\omega_{wave}/k_{\parallel}$ vs Altitude' +
                         '\n' + r'$f_{wave}$=' + f'{round(EToggles.waveFreq_rad/(2*pi),1)} Hz, '
                                                       '$\lambda_{\perp 0}$ ='+f'{EToggles.lambdaPerp0/1000} km', fontsize=Title_FontSize)
            ax.set_ylabel('DAW Speed  [10,000 km/s]', fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axvline(x=400000 / xNorm, label='Observation Height', color='red', linestyle='--',linewidth=Plot_LineWidth)

            # plot the MHD alfven speed
            ax.plot(altRange / xNorm, alfSpdMHD/(10000*m_to_km), color='red',linewidth=Plot_LineWidth, label='MHD Speed')

            # plot some thermal velocity comparisons
            Vth_low = sqrt(8*q0*1/(9.11E-31))/(m_to_km*10000)
            ax.axhline(y=Vth_low, color='black',linewidth=Plot_LineWidth)
            ax.text(x=R_REF/xNorm,y=Vth_low*2,s='$V_{th_{e}}$ (1 eV)', color='black',fontsize=Text_Fontsize)

            Vth_high = sqrt(8 * q0 * 50 / (9.11E-31))/(m_to_km*10000)
            ax.axhline(y=Vth_high, color='black',linewidth=Plot_LineWidth)
            ax.text(x=R_REF/xNorm, y=Vth_high * 1.3, s='$V_{th_{e}}$ (50 eV)', color='black',fontsize=Text_Fontsize)

            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_DAWaflSpd.png',dpi=dpi)

        return kineticAlfSpeed

    # --- collect all the functions ---
    profileFuncs = [temperatureProfile,
                    lambdaPerpProfile,
                    plasmaDensityProfile,
                    ionMassProfile,
                    plasmaBetaProfile,
                    plasmaFreqProfile,
                    skinDepthProfile,
                    ionCyclotronProfile,
                    ionLarmorRadiusProfile,
                    MHD_alfvenSpeedProfile,
                    kineticTermsProfiles,
                    lambdaParallelProfile,
                    Intertial_alfvenSpeedProfile]

    ##################
    # --- PLOTTING ---
    ##################
    if plotting:
        counter = 0
        for key, val in plottingDict.items():
            if val and key == 'Density':
                profileFuncs[counter](altRange=GenToggles.simAlt, showPlot=True, useTanakaDensity= useTanakaDensity)
            elif val:
                profileFuncs[counter](altRange = GenToggles.simAlt, showPlot = True)
            counter+= 1


    #####################
    # --- OUTPUT DATA ---
    #####################
    if outputData:

        # get all the variables
        Bgeo, Bgrad  = data_dict_Bgeo['Bgeo'][0],data_dict_Bgeo['Bgrad'][0]
        Temp = temperatureProfile(GenToggles.simAlt)
        lambdaPerp, kperp = lambdaPerpProfile(GenToggles.simAlt)
        plasmaDensity = plasmaDensityProfile(GenToggles.simAlt)
        n_Op, n_Hp, m_eff_i = ionMassProfile(GenToggles.simAlt)
        beta = plasmaBetaProfile(GenToggles.simAlt)
        plasmaFreq = plasmaFreqProfile(GenToggles.simAlt)
        skinDepth = skinDepthProfile(GenToggles.simAlt)
        ionCyclotron, ionCyclotron_Op, ionCyclotron_Hp = ionCyclotronProfile(GenToggles.simAlt)
        ionLarmorRadius, ionLarmorRadius_Op, ionLarmorRadius_Hp = ionLarmorRadiusProfile(GenToggles.simAlt)
        alfSpdMHD = MHD_alfvenSpeedProfile(GenToggles.simAlt)
        inertialTerm, finiteFreqTerm, LarmorTerm = kineticTermsProfiles(GenToggles.simAlt)
        lambdaPara, kpara = lambdaParallelProfile(GenToggles.simAlt)
        alfSpdInertial = Intertial_alfvenSpeedProfile(GenToggles.simAlt)

        if outputData:

            from copy import deepcopy
            from myspaceToolsLib.CDF_load import outputCDFdata

            # --- Construct the Data Dict ---
            exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                          'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                          'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}

            data_dict = {'Bgeo': [Bgeo, {'DEPEND_0': 'simAlt', 'UNITS': 'T', 'LABLAXIS': 'Bgeo'}],
                         'Bgrad': [Bgrad, {'DEPEND_0': 'simAlt', 'UNITS': 'T', 'LABLAXIS': 'Bgrad'}],
                         'Temp': [Temp, {'DEPEND_0': 'simAlt', 'UNITS': 'eV', 'LABLAXIS': 'Temperature'}],
                         'lambdaPerp': [lambdaPerp, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'LambdaPerp'}],
                         'kperp': [kperp, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-1!N', 'LABLAXIS': 'kperp'}],
                         'plasmaDensity': [plasmaDensity, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'plasmaDensity'}],
                         'n_Op': [n_Op, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'n_Op'}],
                         'n_Hp': [n_Hp, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'n_Hp'}],
                         'm_eff_i': [m_eff_i, {'DEPEND_0': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i'}],
                         'beta': [beta, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta'}],
                         'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}],
                         'skinDepth': [skinDepth, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'skinDepth'}],
                         'ionCyclotron': [ionCyclotron, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron'}],
                         'ionCyclotron_Op': [ionCyclotron_Op, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron_Op'}],
                         'ionLarmorRadius_Hp': [ionLarmorRadius_Hp, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionLarmorRadius_Hp'}],
                         'alfSpdMHD': [alfSpdMHD, {'DEPEND_0': 'simAlt', 'UNITS': 'm/s', 'LABLAXIS': 'alfSpdMHD'}],
                         'inertialTerm': [inertialTerm, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'inertialTerm'}],
                         'finiteFreqTerm': [finiteFreqTerm, {'DEPEND_0': 'finiteFreqTerm', 'UNITS': None, 'LABLAXIS': 'finiteFreqTerm'}],
                         'LarmorTerm': [LarmorTerm, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'LarmorTerm'}],
                         'lambdaPara': [lambdaPara, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'LambdaPara'}],
                         'kpara': [kpara, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-1!N', 'LABLAXIS': 'kpara'}],
                         'alfSpdInertial': [alfSpdInertial, {'DEPEND_0': 'simAlt', 'UNITS': 'm/s', 'LABLAXIS': 'alfSpdInertial'}],
                         'simAlt': [GenToggles.simAlt, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}]}

            # update the data dict attrs
            for key, val in data_dict.items():
                newAttrs = deepcopy(exampleVar)

                for subKey, subVal in data_dict[key][1].items():
                    newAttrs[subKey] = subVal

                data_dict[key][1] = newAttrs

            outputPath = rf'{GenToggles.simOutputPath}\plasmaEnvironment\plasmaEnvironment.cdf'
            outputCDFdata(outputPath, data_dict)



#################
# --- EXECUTE ---
#################
generatePlasmaEnvironment(outputData=outputData, showPlot=plotting)