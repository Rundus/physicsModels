# --- imports ---
from ionosphere.simToggles_iono import GenToggles, plasmaToggles
from spaceToolsLib.variables import u0,m_e,ep0,cm_to_m,IonMasses,q0, m_to_km,Re,kB,m_Hp,m_Op,m_Np,m_Hep,m_NOp,m_O2p,m_N2p
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from numpy import exp, sqrt, array, pi, tanh,power,abs,sum
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata
from scipy.interpolate import CubicSpline


# TODO: Convert each model into a class object with functions representing its various variables. This should standardize some of these functions and allow for expansion.
# TODO: add electron Larmor Length

##################
# --- PLOTTING ---
##################
xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'

# get the geomagnetic field data dict
data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simFolderPath}\geomagneticField\geomagneticField.cdf')

def generatePlasmaEnvironment(outputData,GenToggles,plasmaToggles, **kwargs):
    plotting = kwargs.get('showPlot', False)
    data_dict = {}

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

    if plasmaToggles.useIRI:
        Inames = ['O+', 'H+', 'He+', 'O2+', 'NO+', 'N+']
        Ikeys = ['Op', 'Hp', 'Hep', 'O2p', 'NOp', 'Np']
        Imasses = array(IonMasses[1:6+1])
    else:
        Inames = ['O+', 'H+']
        Ikeys = ['Op', 'Hp']
        Imasses = array(IonMasses[1:3])


    # --- IRI ---
    # if you use the IRI data anywhere, downsample it at the right lat/long/alt/time and interpolate it onto the simulation
    if True in [plasmaToggles.useIRI_ne_Profile,plasmaToggles.useIRI_Ti_Profile,plasmaToggles.useIRI_Te_Profile,plasmaToggles.useIRI_ni_Profile]:

        # collect the IRI data
        data_dict_IRI = deepcopy(loadDictFromFile(plasmaToggles.IRI_filePath))

        # get the IRI at the specific Lat/Long/Time
        dt_targetTime = GenToggles.target_time
        time_idx = abs(data_dict_IRI['time'][0] - int(dt_targetTime.hour * 60 + dt_targetTime.minute)).argmin()
        lat_idx = abs(data_dict_IRI['lat'][0] - GenToggles.target_Latitude).argmin()
        long_idx = abs(data_dict_IRI['lon'][0] - GenToggles.target_Longitude).argmin()
        alt_low_idx, alt_high_idx = abs(data_dict_IRI['ht'][0] - GenToggles.simAltLow/m_to_km).argmin(), abs(data_dict_IRI['ht'][0] - GenToggles.simAltHigh/m_to_km).argmin()

        for varname in data_dict_IRI.keys():
            if varname not in ['time', 'ht', 'lat', 'lon']:

                reducedData = deepcopy(array(data_dict_IRI[varname][0][time_idx,alt_low_idx:alt_high_idx,lat_idx,long_idx]))

                # interpolate data onto ionosphere altitude range
                # --- cubic interpolation ---
                splCub = CubicSpline(data_dict_IRI['ht'][0][alt_low_idx:alt_high_idx], reducedData)

                # evaluate the interpolation and store the interpolated data
                data_dict_IRI[varname][0] = array([splCub(val) for val in GenToggles.simAlt])

    # --- Temperature ---
    def electron_temperatureProfile(altRange,data_dict,**kwargs):

        if plasmaToggles.useSchroeder_Te_Profile:
            # --- Ionosphere Temperature Profile ---
            # ASSUMES Ions and electrons have same temperature profile
            T0 = 2.5 # Temperature at the Ionospher (in eV)
            T1 = 0.0135 # (in eV)
            h0 = 2000*m_to_km # scale height (in meters)
            T_iono = T1*exp(altRange/h0) + T0
            deltaZ = 0.3*Re
            T_ps = 2000 # temperature of plasma sheet (in eV)
            # T_ps = 105  # temperature of plasma sheet (in eV)
            z_ps = 3.75*Re # height of plasma sheet (in meters)
            w = 0.5*(1 - tanh((altRange - z_ps)/deltaZ)) # models the transition to the plasma sheet

            # determine the overall temperature profile
            T_e = array([T_iono[i]*w[i] + T_ps*(1 - w[i]) for i in range(len(altRange))])

            if kwargs.get('showPlot', False):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(3, sharex=True)
                fig.set_size_inches(figure_width, figure_height * (3 / 2))

                ax[0].plot(altRange / xNorm, T_iono, linewidth=Plot_LineWidth)
                ax[0].set_title('Electron Ionospheric Temperature Profile vs Altitude', fontsize=Title_FontSize)
                ax[0].set_ylabel('Temperature [eV]', fontsize=Label_FontSize)
                ax[0].set_yscale('log')
                ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='red')
                ax[0].grid(True)

                ax[1].plot(altRange / xNorm, w, linewidth=Plot_LineWidth)
                ax[1].set_title('Weighting Function vs Altitude', fontsize=Title_FontSize)
                ax[1].set_ylabel('Weighting Function', fontsize=Label_FontSize)
                # ax[1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
                ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red')

                ax[2].plot(altRange / xNorm, T_e, linewidth=Plot_LineWidth)
                ax[2].set_yscale('log')
                ax[2].set_title('Total Electron Temperature vs Altitude', fontsize=Title_FontSize)
                ax[2].set_ylabel('Electron Temperature [eV]', fontsize=Label_FontSize)
                ax[2].set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
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
                plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_electron_Temperature.png', dpi=dpi)

        elif plasmaToggles.useIRI_Te_Profile:
            T_e = data_dict_IRI['Te'][0]*(kB/q0)
            T_iono = T_e


        # update the data_dict
        return {**data_dict,**{'Te': [T_e, {'DEPEND_0': 'simAlt', 'UNITS': 'eV', 'LABLAXIS': 'Te'}]}}

    # --- Temperature ---
    def ion_temperatureProfile(altRange,data_dict,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        if plasmaToggles.useSchroeder_Te_Profile:
            # --- Ionosphere Temperature Profile ---
            # ASSUMES Ions and electrons have same temperature profile
            T0 = 2.5 # Temperature at the Ionospher (in eV)
            T1 = 0.0135 # (in eV)
            h0 = 2000*m_to_km # scale height (in meters)
            T_iono = T1*exp(altRange/h0) + T0
            deltaZ = 0.3*Re
            T_ps = 2000 # temperature of plasma sheet (in eV)
            # T_ps = 105  # temperature of plasma sheet (in eV)
            z_ps = 3.75*Re # height of plasma sheet (in meters)
            w = 0.5*(1 - tanh((altRange - z_ps)/deltaZ)) # models the transition to the plasma sheet

            # determine the overall temperature profile
            T_i = array([T_iono[i]*w[i] + T_ps*(1 - w[i]) for i in range(len(altRange))])

            if plotBool:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(3, sharex=True)
                fig.set_size_inches(figure_width, figure_height * (3 / 2))

                ax[0].plot(altRange / xNorm, T_iono, linewidth=Plot_LineWidth)
                ax[0].set_title('Ion Ionospheric Temperature Profile vs Altitude', fontsize=Title_FontSize)
                ax[0].set_ylabel('Temperature [eV]', fontsize=Label_FontSize)
                ax[0].set_yscale('log')
                ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='red')
                ax[0].grid(True)

                ax[1].plot(altRange / xNorm, w, linewidth=Plot_LineWidth)
                ax[1].set_title('Weighting Function vs Altitude', fontsize=Title_FontSize)
                ax[1].set_ylabel('Weighting Function', fontsize=Label_FontSize)
                # ax[1].set_xlabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
                ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red')

                ax[2].plot(altRange / xNorm, T_i, linewidth=Plot_LineWidth)
                ax[2].set_yscale('log')
                ax[2].set_title('Total Ion Temperature vs Altitude', fontsize=Title_FontSize)
                ax[2].set_ylabel('Electron Temperature [eV]', fontsize=Label_FontSize)
                ax[2].set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
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
                plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_Ion_Temperature.png', dpi=dpi)

        elif plasmaToggles.useIRI_Te_Profile:
            T_i = data_dict_IRI['Te'][0]*kB/q0
            T_iono = T_i



        return {**data_dict,**{'T_i': [T_i, {'DEPEND_0': 'simAlt', 'UNITS': 'eV', 'LABLAXIS': 'Ion Temperature'}]}}

    # --- PLASMA DENSITY ---
    # uses the Kletzing Model to return an array of plasma density (in m^-3) from [Alt_low, ..., Alt_High]
    def electron_plasmaDensityProfile(altRange,data_dict,**kwargs):

        if plasmaToggles.useTanaka_ne_Profile:
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
            ne_density = (cm_to_m ** 3) * array([ fitFunc(alt/m_to_km, n0, n1, z0, h, H, a) for alt in altRange])  # calculated density (in m^-3)

        elif plasmaToggles.useKletzingS33_Profile:
            #### KLETZING AND TORBERT MODEL ####
            # --- determine the density over all altitudes ---
            # Description: returns density for altitude "z [km]" in m^-3
            h = 0.06 * (Re / m_to_km)  # in km from E's surface
            n0 = 6E4
            n1 = 1.34E7
            z0 = 0.05 * (Re / m_to_km)  # in km from E's surface
            ne_density = (cm_to_m**3)*array([(n0 * exp(-1 * ((alt / m_to_km) - z0) / h) + n1 * ((alt / m_to_km) ** (-1.55))) for alt in altRange])  # calculated density (in m^-3)

        elif plasmaToggles.useChaston_Profile:
            raise Exception('No Chaston Profile Available yet!')

        elif plasmaToggles.useStatic_ne_Profile:
            ne_density = array([plasmaToggles.staticDensity for alt in altRange])

        elif plasmaToggles.useIRI_ne_Profile:
            ne_density = deepcopy(data_dict_IRI['Ne'][0])

        data_dict = {**data_dict,**{'ne': [ne_density, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'ne'}]}}


        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width,1 + figure_height/2)
            ax.plot(altRange/xNorm, ne_density/(cm_to_m**3),linewidth=Plot_LineWidth)
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
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_density',dpi=dpi)

        return data_dict

    # --- Ion Mass ---
    def ion_plasmaDensityProfile(altRange,data_dict,**kwargs):
        if plasmaToggles.useSchroeder_Te_Profile:
            plasmaDensity_total = data_dict['ne'][0]
            z_i = 2370*m_to_km  #
            h_i = 1800*m_to_km  # height of plasma sheet (in meters)
            n_Op = array([plasmaDensity_total[i]*0.5 * (1 - tanh((altRange[i] - z_i) / h_i)) for i in range(len(altRange))])
            n_Hp = plasmaDensity_total - n_Op
            m_Op = IonMasses[1]
            m_Hp = IonMasses[2]
            m_eff_i = array([ m_Hp*0.5*(1 + tanh( (altRange[i] - z_i)/h_i )) + m_Op*0.5*(1 - tanh( (altRange[i] - z_i)/h_i )) for i in range(len(altRange))])

            if kwargs.get('showPlot', False):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, sharex=True)
                fig.set_size_inches(figure_width, figure_height)
                ax[0].plot(altRange / xNorm, n_Op, color='blue', label='$n_{0^{+}}$ [$m^{-3}$]',
                           linewidth=Plot_LineWidth)
                ax[0].plot(altRange / xNorm, n_Hp, color='red', label='$n_{H^{+}}$ [$m^{-3}$]',
                           linewidth=Plot_LineWidth)
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
                    ax[i].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                      length=Tick_Length)
                    ax[i].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,
                                      length=Tick_Length_minor)
                    ax[i].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                      length=Tick_Length)
                    ax[i].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,
                                      length=Tick_Length_minor)

                plt.legend(fontsize=Legend_fontSize)
                plt.tight_layout()
                plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_ionMass.png', dpi=dpi)

            return {**data_dict, **{'n_Op': [n_Op, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'n_Op'}],
                 'n_Hp': [n_Hp, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'n_Hp'}],
                 'm_eff_i': [m_eff_i, {'DEPEND_0': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i'}]  }}

        elif plasmaToggles.useIRI_ni_Profile:

            n_ions = 1E6*array([data_dict_IRI[f"{key}"][0] for key in Inames])
            m_eff_i = (sum((n_ions.T*Imasses).T,axis=0) / (sum(n_ions,axis=0)))
            ni_total = sum(n_ions,axis=0)
            data_dict = {**data_dict,
                         **{'ni': [ni_total, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'ni'}]},
                         **{'m_eff_i': [m_eff_i, {'DEPEND_0': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i'}]},
                         **{f'n_{key}': [n_ions[idx], {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': f'n_{key}'}] for idx, key in enumerate(Ikeys)}
                         }

            if kwargs.get('showPlot', False):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, sharex=True)
                fig.set_size_inches(figure_width, figure_height)
                for idx,thing in enumerate(n_ions):
                    ax[0].plot(altRange / xNorm, thing, label=f'$n_{Inames[idx]}$ [$m^{-3}$]', linewidth=Plot_LineWidth)
                ax[0].plot(altRange / xNorm, ni_total, label=f'$ni$ [$m^{-3}$]', linewidth=Plot_LineWidth)
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
                    ax[i].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                      length=Tick_Length)
                    ax[i].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,
                                      length=Tick_Length_minor)
                    ax[i].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                      length=Tick_Length)
                    ax[i].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,
                                      length=Tick_Length_minor)

                plt.legend(fontsize=Legend_fontSize)
                plt.tight_layout()
                plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_ionMass.png', dpi=dpi)

        return data_dict

    # --- PLASMA BETA ---
    def plasmaBetaProfile(altRange,data_dict,**kwargs):

        m_eff_i = data_dict['m_eff_i'][0]
        plasmaBeta = (data_dict['ne'][0]*q0*data_dict['Te'][0])/((data_dict_Bgeo['Bgeo'][0])**2 /(2*u0))
        ratio = m_e/m_eff_i
        data_dict = {**data_dict, **{'beta_e': [plasmaBeta, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta_e'}]}}

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            colors= ['tab:red','tab:blue','tab:green']
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(altRange/xNorm, plasmaBeta/ratio, color=next(iter(colors)), label=rf'$\beta_{i}$',linewidth=Plot_LineWidth)
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
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_beta.png',dpi=dpi)

        return data_dict

    # --- PLASMA FREQ ---
    def plasmaFreqProfile(altRange,data_dict,**kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaDensity = data_dict['ne'][0]
        plasmaFreq = array([sqrt(plasmaDensity[i]* (q0*q0) / (ep0*m_e)) for i in range(len(plasmaDensity))])

        data_dict = {**data_dict,**{'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}]}}

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
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_plasFreq.png',dpi=dpi)

        return data_dict

    # --- ION CYCLOTRON FREQ ---
    def ionCyclotronProfile(altRange,data_dict,**kwargs):
        n_ions = array([data_dict[f"n_{key}"][0] for key in Ikeys])
        ionCyclotron_ions = array([q0 * data_dict_Bgeo['Bgeo'][0]  / mass for mass in Imasses])
        ionCyclotron_eff = (ionCyclotron_ions*n_ions)/data_dict['ni'][0]

        data_dict = {**data_dict,
                     **{'ionCyclotron_eff': [ionCyclotron_eff, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron_eff'}]},
                     **{f'ionCyclotron_{key}': [ionCyclotron_ions[idx], {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': f'ionCyclotron_{key}'}] for idx, key in enumerate(Ikeys)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            for thing in ionCyclotron_ions:
                ax[0].plot(altRange / xNorm, thing, label=f'$\Omega_{next(iter(Inames))}$ [rad/s]', linewidth=Plot_LineWidth)

            ax[0].plot(altRange / xNorm, ionCyclotron_eff, label='$\Omega_{eff}$ [rad/s]', linewidth=Plot_LineWidth)
            ax[0].set_title('$\Omega_{ci}$ vs Altitude', fontsize=Title_FontSize)
            ax[0].set_ylabel('$\Omega_{ci}$ [rad/s]', fontsize=Label_FontSize)

            ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax[0].set_yscale('log')
            ax[0].set_ylim(0.1, 1E4)
            ax[0].set_xlim(0, GenToggles.simAltHigh / xNorm)
            ax[0].grid(True)
            ax[0].margins(0)
            ax[0].legend(fontsize=Legend_fontSize)

            for thing in ionCyclotron:
                ax[1].plot(altRange / xNorm, thing/ (2 * pi), label=f'$f_{next(Inames)}$ [Hz]', linewidth=Plot_LineWidth)

            ax[1].plot(altRange / xNorm, ionCyclotron_eff / (2 * pi), color='blue', label='$f_{avg}$', linewidth=Plot_LineWidth)
            ax[1].set_title('$f_{ci}$ vs Altitude', fontsize=Title_FontSize)
            ax[1].set_ylabel('$f_{ci}$ [Hz]', fontsize=Label_FontSize)
            ax[1].set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax[1].set_yscale('log')
            ax[1].set_ylim(0.1, 1000)
            ax[1].set_xlim(0, GenToggles.simAltHigh / xNorm)
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
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_ionCyclo.png', dpi=dpi)


        return data_dict

    # --- Ion Larmor Radius ---
    def ionLarmorRadiusProfile(altRange,data_dict, **kwargs):

        Ti = data_dict['Ti'][0]
        n_ions = array([data_dict[f"n_{key}"] for idx, key in enumerate(Ikeys)])
        vth_ions = array([sqrt(2)*sqrt(8 * q0 * Ti /mass) for mass in Imasses]) # the sqrt(2) comes from the vector sum of two dimensions
        ionLarmorRadius_ions = array([vth_ions[idx] / data_dict[f"ionCyclotron_{key}"][0] for idx,key in enumerate(Ikeys)])
        ionLarmorRadius_eff = n_ions*ionLarmorRadius_ions/data_dict['ne'][0]
        data_dict = {**data_dict,
                     **{'ionLarmorRadius_eff': [ionLarmorRadius_eff, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'ionLarmorRadius_eff'}]}
                     **{f'ionLarmorRadius_{key}': [ionLarmorRadius_ions[idx], {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': f'ionLarmorRadius_{key}'}] for idx, key in enumerate(Ikeys)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 1+ figure_height/2)
            for idx, ion in enumerate(Ikeys):
                ax.plot(altRange / xNorm, data_dict[f"ionLarmorRadius_{ion}"], label=rf"$\rho_{ion}$",linewidth=Plot_LineWidth)

            ax.plot(altRange / xNorm, ionLarmorRadius_eff, label=r'$\rho_{avg}$', linewidth=Plot_LineWidth)
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
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_ionLarmor.png',dpi=dpi)

        return data_dict

    # --- MHD Alfven Speed ---
    def MHD_alfvenSpeedProfile(altRange,data_dict,**kwargs):

        ion_plasmaDensity = data_dict['ni'][0]
        Bgeo = data_dict_Bgeo['Bgeo'][0]
        m_eff_i = data_dict['m_eff_i'][0]
        VA_MHD = array(Bgeo/sqrt(u0*m_eff_i*ion_plasmaDensity))

        data_dict = {**data_dict,
                     **{'alfSpdMHD': [VA_MHD, {'DEPEND_0': 'simAlt', 'UNITS': 'm/s', 'LABLAXIS': 'alfSpdMHD'}]}
                     }

        if kwargs.get('showPlot', False):
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
            ax.text(x=Re/xNorm,y=Vth_low*1.3,s='$V_{th_{e}}$ (1 eV)', color='black',fontsize=Text_Fontsize)

            Vth_high = sqrt(8 * q0 * 50 / (9.11E-31))/(10000*m_to_km)
            ax.axhline(y=Vth_high, color='black',linewidth=Plot_LineWidth)
            ax.text(x=Re/xNorm, y=Vth_high * 1.1, s='$V_{th_{e}}$ (50 eV)', color='black',fontsize=Text_Fontsize)

            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,length=Tick_Length_minor)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_alfMHD.png',dpi=dpi)



        return data_dict


    ##################
    # --- PLOTTING ---
    ##################
    if plotting:

        # --- collect all the functions ---
        profileFuncs = [electron_temperatureProfile,
                        ion_temperatureProfile,
                        electron_plasmaDensityProfile,
                        ion_plasmaDensityProfile,
                        plasmaBetaProfile,
                        plasmaFreqProfile,
                        ionCyclotronProfile,
                        ionLarmorRadiusProfile,
                        MHD_alfvenSpeedProfile]

        for i in range(len(profileFuncs)):
            data_dict = profileFuncs[i](altRange=GenToggles.simAlt, data_dict=data_dict, showPlot=True)


    #####################
    # --- OUTPUT DATA ---
    #####################
    if outputData:

        # include the simulation altt
        data_dict = {**data_dict,**{'simAlt': [GenToggles.simAlt, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}]}}

        # --- Construct the Data Dict ---
        exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                      'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                      'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}


        # update the data dict attrs
        for key, val in data_dict.items():
            newAttrs = deepcopy(exampleVar)

            for subKey, subVal in data_dict[key][1].items():
                newAttrs[subKey] = subVal

            data_dict[key][1] = newAttrs

        outputPath = rf'{GenToggles.simFolderPath}\plasmaEnvironment\plasmaEnvironment.cdf'
        outputCDFdata(outputPath, data_dict)
