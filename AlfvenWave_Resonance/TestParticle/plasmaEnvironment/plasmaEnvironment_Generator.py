# --- imports ---
from ACESII_code.Science.Simulations.TestParticle.simToggles import m_to_km, R_REF, GenToggles, EToggles, runFullSimulation
from myspaceToolsLib.physicsVariables import lightSpeed,u0,m_e,ep0,cm_to_m,IonMasses,q0
from myspaceToolsLib.CDF_load import loadDictFromFile
from numpy import exp, sqrt, array, pi, abs, tanh
simulationAlt = GenToggles.simAlt


##################
# --- PLOTTING ---
##################
plotting = False
useTanakaDensity = False
xNorm = R_REF # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == R_REF else 'km'
plottingDict = {'Temperature': True,
                'lambdaPerp': True,
                'Density': True,
                'ionMass': True,
                'Beta': True,
                'plasmaFreq': True,
                'skinDepth': True,
                'ionCyclotron': True,
                'ionLarmorRadius': True,
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

    # --- Temperature ---
    def temperatureProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        # --- Ionosphere Temperature Profile ---
        # ASSUMES Ions and electrons have same temperature profile
        T0 = 2.5 # Temperature at the Ionospher (in eV)
        T1 = 0.0135 # (in eV)
        h0 = 2000*m_to_km # scale height (in meters)
        T_iono = T1*exp(altRange/h0) + T0
        deltaZ = 0.3*R_REF
        T_ps = 2000 # temperature of plasma sheet (in eV)
        # T_ps = 105  # temperature of plasma sheet (in eV)
        z_ps = 3.75*R_REF # height of plasma sheet (in meters)
        w = 0.5*(1 - tanh((altRange - z_ps)/deltaZ)) # models the transition to the plasma sheet

        # determine the overall temperature profile
        T_e = array([T_iono[i]*w[i] + T_ps*(1 - w[i]) for i in range(len(altRange))])

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3,sharex=True)
            fig.set_size_inches(figure_width, figure_height*(3/2))

            ax[0].plot(altRange / xNorm, T_iono,linewidth=Plot_LineWidth)
            ax[0].set_title('Ionospheric Temperature Profile vs Altitude', fontsize=Title_FontSize)
            ax[0].set_ylabel('Temperature [eV]',fontsize=Label_FontSize)
            ax[0].set_yscale('log')
            ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='red')
            ax[0].grid(True)

            ax[1].plot(altRange / xNorm, w,linewidth=Plot_LineWidth)
            ax[1].set_title('Weighting Function vs Altitude', fontsize=Title_FontSize)
            ax[1].set_ylabel('Weighting Function',fontsize=Label_FontSize)
            # ax[1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red')

            ax[2].plot(altRange / xNorm, T_e,linewidth=Plot_LineWidth)
            ax[2].set_yscale('log')
            ax[2].set_title('Total Electron Temperature vs Altitude', fontsize=Title_FontSize)
            ax[2].set_ylabel('Electron Temperature [eV]',fontsize=Label_FontSize)
            ax[2].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[2].axvline(x=400000 / xNorm, label='Observation Height', color='red')
            ax[2].grid(True)

            for i in range(3):
                ax[i].grid(True)
                ax[i].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                ax[i].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)
                ax[i].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                ax[i].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_Temperature.png',dpi=dpi)

        return T_e

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

    # --- PLASMA DENSITY ---
    # uses the Klezting Model to return an array of plasma density (in m^-3) from [Alt_low, ..., Alt_High]
    def plasmaDensityProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)
        useTanakaDensity = kwargs.get('useTanakaDensity', False)

        if useTanakaDensity:

            ##### TANAKA FIT #####
            # --- determine the density over all altitudes ---
            # Description: returns density for altitude "z [km]" in m^-3
            n0 = 24000000
            n1 = 2000000
            z0 = 600  # in km from E's surface
            h = 680  # in km from E's surface
            H = -0.74
            a = 0.0003656481654202569

            def fitFunc(x, n0, n1, z0, h, H, a):
                return a * (n0 * exp(-1 * (x - z0) / h) + n1 * (x ** (H)))
            n_density = (cm_to_m ** 3) * array([ fitFunc(alt/m_to_km, n0, n1, z0, h, H, a) for alt in altRange])  # calculated density (in m^-3)

        elif EToggles.staticDensity:
            n_density = array([EToggles.staticDensity for alt in altRange])

        else:
            #### KLETZING AND TORBERT MODEL ####
            # --- determine the density over all altitudes ---
            # Description: returns density for altitude "z [km]" in m^-3
            h = 0.06 * (R_REF / m_to_km)  # in km from E's surface
            n0 = 6E4
            n1 = 1.34E7
            z0 = 0.05 * (R_REF / m_to_km)  # in km from E's surface
            n_density = (cm_to_m**3)*array([(n0 * exp(-1 * ((alt / m_to_km) - z0) / h) + n1 * ((alt / m_to_km) ** (-1.55))) for alt in altRange])  # calculated density (in m^-3)

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width,1 + figure_height/2)
            ax.plot(altRange/xNorm, n_density/(cm_to_m**3),linewidth=Plot_LineWidth)
            ax.set_title('$n$ vs Altitude', fontsize = Title_FontSize)
            ax.set_ylabel('Density [$cm^{-3}$]', fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axvline(x=400000/xNorm,label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax.set_yscale('log')
            ax.set_ylim(1E-2, 1E6)
            ax.margins(0)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.grid(True)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_density',dpi=dpi)


        return n_density

    # --- Ion Mass ---
    def ionMassProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaDensity_total = plasmaDensityProfile(altRange)
        z_i = 2370*m_to_km  #
        h_i = 1800*m_to_km  # height of plasma sheet (in meters)
        n_Op = array([plasmaDensity_total[i]*0.5 * (1 - tanh((altRange[i] - z_i) / h_i)) for i in range(len(altRange))])
        n_Hp = plasmaDensity_total - n_Op
        m_Op = IonMasses[1]
        m_Hp = IonMasses[2]
        m_eff_i = array([ m_Hp*0.5*(1 + tanh( (altRange[i] - z_i)/h_i )) + m_Op*0.5*(1 - tanh( (altRange[i] - z_i)/h_i )) for i in range(len(altRange))])

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            ax[0].plot(altRange / xNorm, n_Op, color='blue', label='$n_{0^{+}}$ [$m^{-3}$]', linewidth=Plot_LineWidth)
            ax[0].plot(altRange / xNorm, n_Hp, color='red', label='$n_{H^{+}}$ [$m^{-3}$]', linewidth=Plot_LineWidth)
            ax[0].set_title('Plasma densities vs Altitude', fontsize=Title_FontSize)
            ax[0].set_ylabel(r'Density [m$^{-3}$]', fontsize=Label_FontSize)
            ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='black', linewidth=Plot_LineWidth)
            ax[0].set_yscale('log')
            ax[0].legend(fontsize=Legend_fontSize)

            ax[1].plot(altRange / xNorm, m_eff_i, linewidth=Plot_LineWidth)
            ax[1].set_ylabel('$m_{eff_{i}}$ [kg]', fontsize=Label_FontSize)
            ax[1].set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            for i in range(2):
                ax[i].grid(True)
                ax[i].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
                ax[i].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
                ax[i].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
                ax[i].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_ionMass.png',dpi=dpi)

        return n_Op, n_Hp, m_eff_i

    # --- PLASMA BETA ---
    def plasmaBetaProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaDensity = plasmaDensityProfile(altRange)
        Bgeo,Bgrad = data_dict_Bgeo['Bgeo'][0],data_dict_Bgeo['Bgrad'][0]
        Te = 50
        plasmaBeta = array([(plasmaDensity[i]*q0*Te)/(Bgeo[i]**2 /(2*u0)) for i in range(len(altRange))])
        n_Op, n_Hp, m_eff_i = ionMassProfile(altRange)
        ratio = m_e/m_eff_i

        if plotBool:
            import matplotlib.pyplot as plt

            Temps = [1, 10, 50]
            colors= ['tab:red','tab:blue','tab:green']

            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)

            for k,temp in enumerate(Temps):
                plasmaBeta = array([(plasmaDensity[i] * q0 * temp) / (Bgeo[i] ** 2 / (2 * u0)) for i in range(len(altRange))])
                ax.plot(altRange/xNorm, plasmaBeta/ratio, color=colors[k], label=f'T_e = {temp} eV',linewidth=Plot_LineWidth)

            ax.set_title(r'$\beta$ vs Altitude',fontsize=Title_FontSize)
            ax.set_ylabel('Plasma Beta / (m_e/m_i)',fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axvline(x=400000/xNorm, label='Observation Height', color='red',linestyle='--',linewidth=Plot_LineWidth)
            ax.axhline(y=1, color='black')
            ax.set_yscale('log')
            plt.legend(fontsize=Legend_fontSize)
            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_beta.png',dpi=dpi)


        return plasmaBeta

    # --- PLASMA FREQ ---
    def plasmaFreqProfile(altRange,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaDensity = plasmaDensityProfile(altRange)
        plasmaFreq = array([sqrt(plasmaDensity[i]* (q0*q0) / (ep0*m_e)) for i in range(len(plasmaDensity))])

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 1+ figure_height/2)
            ax.plot(altRange/xNorm, plasmaFreq,linewidth=Plot_LineWidth)
            ax.set_title('$\omega_{pe}$ vs Altitude',fontsize=Title_FontSize)
            ax.set_ylabel('Plasma Freq [rad/s]',fontsize=Label_FontSize)
            ax.set_yscale('log')
            ax.set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axvline(x=400000/xNorm,label='Observation Height',color='red',linewidth=Plot_LineWidth)
            plt.legend(fontsize=Legend_fontSize)
            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_plasFreq.png',dpi=dpi)

        return plasmaFreq

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

    # --- ION CYCLOTRON FREQ ---
    def ionCyclotronProfile(altRange,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        Bgeo,Bgrad = data_dict_Bgeo['Bgeo'][0], data_dict_Bgeo['Bgrad'][0]
        plasmaDensity = plasmaDensityProfile(altRange)
        n_Op, n_Hp, m_eff_i = ionMassProfile(altRange)
        m_Op = IonMasses[1]
        m_Hp = IonMasses[2]
        ionCyclotron_Op = q0 * Bgeo / m_Op
        ionCyclotron_Hp = q0 * Bgeo / m_Hp
        ionCyclotron = (n_Op*ionCyclotron_Op + n_Hp*ionCyclotron_Hp)/plasmaDensity

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            ax[0].plot(altRange/xNorm, ionCyclotron, color='blue', label='$\omega_{total}$',linewidth=Plot_LineWidth)
            ax[0].plot(altRange / xNorm, ionCyclotron_Op, color='black', label='$\omega_{Op}$',linewidth=Plot_LineWidth)
            ax[0].plot(altRange / xNorm, ionCyclotron_Hp, color='red', label='$\omega_{Hp}$',linewidth=Plot_LineWidth)
            ax[0].set_title('$\omega_{ci}$ vs Altitude',fontsize=Title_FontSize)
            ax[0].set_ylabel('$\omega_{ci}$ [rad/s]',fontsize=Label_FontSize)

            ax[0].axvline(x=400000/xNorm, label='Observation Height',color='red',linewidth=Plot_LineWidth)
            ax[0].set_yscale('log')
            ax[0].set_ylim(0.1, 1E4)
            ax[0].set_xlim(0,GenToggles.simAltHigh/xNorm)
            ax[0].grid(True)
            ax[0].margins(0)
            ax[0].legend(fontsize=Legend_fontSize)

            ax[1].plot(altRange / xNorm, ionCyclotron / (2*pi), color='blue', label='$f_{avg}$',linewidth=Plot_LineWidth)
            ax[1].plot(altRange / xNorm, ionCyclotron_Op/ (2*pi), color='black', label='$f_{Op}$',linewidth=Plot_LineWidth)
            ax[1].plot(altRange / xNorm, ionCyclotron_Hp/ (2*pi), color='green', label='$f_{Hp}$',linewidth=Plot_LineWidth)
            ax[1].set_title('$f_{ci}$ vs Altitude',fontsize=Title_FontSize)
            ax[1].set_ylabel('$f_{ci}$ [Hz]',fontsize=Label_FontSize)
            ax[1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red',linewidth=Plot_LineWidth)
            ax[1].set_yscale('log')
            ax[1].set_ylim(0.1, 1000)
            ax[1].set_xlim(0, GenToggles.simAltHigh/xNorm)
            ax[1].margins(0)
            ax[1].grid(True)
            ax[1].legend(fontsize=Legend_fontSize)


            for i in range(2):
                ax[i].grid(True)
                ax[i].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                ax[i].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)
                ax[i].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                ax[i].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_ionCyclo.png',dpi=dpi)

        return ionCyclotron, ionCyclotron_Op, ionCyclotron_Hp

    # --- Ion Larmor Radius ---
    def ionLarmorRadiusProfile(altRange,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        ionCyclotron, ionCyclotron_Op, ionCyclotron_Hp = ionCyclotronProfile(altRange)
        n_Op, n_Hp, m_eff_i = ionMassProfile(altRange)
        plasmaDensity = plasmaDensityProfile(altRange)
        Ti = temperatureProfile(altRange)
        vth_Op = sqrt(2)*sqrt(8 * q0 * Ti /IonMasses[1]) # the sqrt(2) comes from the vector sum of two dimensions
        vth_Hp = sqrt(2)*sqrt(8 * q0 * Ti/ IonMasses[2])

        ionLarmorRadius_Op = vth_Op / ionCyclotron_Op
        ionLarmorRadius_Hp = vth_Hp / ionCyclotron_Hp
        ionLarmorRadius = (n_Op*ionLarmorRadius_Op + n_Hp*ionLarmorRadius_Hp)/plasmaDensity

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 1+ figure_height/2)
            ax.plot(altRange / xNorm, ionLarmorRadius, label=r'$\rho_{avg}$',linewidth=Plot_LineWidth)
            ax.plot(altRange / xNorm, ionLarmorRadius_Op, label=r'$\rho_{Op}$',linewidth=Plot_LineWidth)
            ax.plot(altRange / xNorm, ionLarmorRadius_Hp, label=r'$\rho_{Hp}$',linewidth=Plot_LineWidth)
            ax.set_title(r'$\rho_{i}$ vs Altitude',fontsize=Title_FontSize)
            ax.set_ylabel(r'$\rho_{i}$ [m]',fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axvline(x=400000 / xNorm, label='Observation Height', color='red',linestyle='--',linewidth=Plot_LineWidth)
            # ax.axvline(x=10000000 / xNorm, label='Magnetosheath Proton Limit', color='tab:green',linestyle='--',linewidth=Plot_LineWidth)
            ax.set_yscale('log')
            ax.set_ylim(1, 4E5)
            ax.margins(0)
            plt.legend(fontsize=Legend_fontSize)
            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            plt.tight_layout()
            plt.savefig('C:\Data\ACESII\science\simulations\TestParticle\plasmaEnvironment\MODEL_ionLarmor.png',dpi=dpi)

        return ionLarmorRadius, ionLarmorRadius_Op, ionLarmorRadius_Hp

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