# --- imports ---
from src.physicsModels.ionosphere.simToggles_iono import GenToggles
from spaceToolsLib.variables import u0,m_e,ep0,q0, Re,kB,ion_dict
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from spaceToolsLib.tools.CDF_output import outputCDFdata
from src.physicsModels.ionosphere.PlasmaEnvironment.model_plasmaEnvironment_classes import *
import numpy as np
from copy import deepcopy


##################
# --- PLOTTING ---
##################
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
Legend_fontSize = 20
dpi = 100
xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'

# get the geomagnetic field data dict
data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simFolderPath}\geomagneticField\geomagneticField.cdf')

def generatePlasmaEnvironment(outputData,GenToggles,plasmaToggles, **kwargs):
    plotting = kwargs.get('showPlot', False)
    data_dict = {}
    Imasses = np.array([ion_dict[key] for key in plasmaToggles.wIons])
    Ikeys = np.array([key for key in plasmaToggles.wIons])

    # --- IRI ---
    data_dict_IRI = deepcopy(loadDictFromFile(plasmaToggles.IRI_filePath)) # collect the IRI data

    # get the IRI at the specific Lat/Long/Time
    dt_targetTime = GenToggles.target_time
    time_idx = abs(data_dict_IRI['time'][0] - int(dt_targetTime.hour * 60 + dt_targetTime.minute)).argmin()
    lat_idx = abs(data_dict_IRI['lat'][0] - GenToggles.target_Latitude).argmin()
    long_idx = abs(data_dict_IRI['lon'][0] - GenToggles.target_Longitude).argmin()
    alt_low_idx, alt_high_idx = abs(data_dict_IRI['ht'][0] - GenToggles.simAltLow/m_to_km).argmin(), abs(data_dict_IRI['ht'][0] - GenToggles.simAltHigh/m_to_km).argmin()

    for varname in data_dict_IRI.keys():
        if varname not in ['time', 'ht', 'lat', 'lon']:

            reducedData = deepcopy(np.array(data_dict_IRI[varname][0][time_idx,alt_low_idx:alt_high_idx,lat_idx,long_idx]))

            # --- linear 1D interpolate data to assist the cubic interpolation ---
            interpolated_result = np.interp(GenToggles.simAlt,data_dict_IRI['ht'][0][alt_low_idx:alt_high_idx]*m_to_km, reducedData)
            data_dict_IRI[varname][0] = interpolated_result

    # --- Temperature ---
    def electron_temperatureProfile(altRange,data_dict,**kwargs):

        if plasmaToggles.useIRI_Te_Profile:
            T_e = data_dict_IRI['Te'][0]

        data_dict = {**data_dict, **{'Te': [T_e, {'DEPEND_0': 'simAlt', 'UNITS': 'K', 'LABLAXIS': 'Te'}]}}

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))

            ax.plot(data_dict['Te'][0], altRange / xNorm,  linewidth=Plot_LineWidth)
            ax.set_title('Electron Ionospheric Temperature Profile vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('Temperature [K]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.grid(True)
            ax.set_xlim(100,3500)

            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                              length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                              width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                              length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                              width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_Electron_Temperature.png', dpi=dpi)

        # update the data_dict
        return data_dict

    # --- Temperature ---
    def ion_temperatureProfile(altRange,data_dict,**kwargs):

        if plasmaToggles.useIRI_Te_Profile:
            Ti = data_dict_IRI['Ti'][0]

        data_dict = {**data_dict,**{'Ti': [Ti, {'DEPEND_0': 'simAlt', 'UNITS': 'K', 'LABLAXIS': 'Ti'}]}}

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))

            ax.plot(data_dict['Ti'][0], altRange / xNorm,  linewidth=Plot_LineWidth)
            ax.set_title('Ion Ionospheric Temperature Profile vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('Temperature [K]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.grid(True)
            ax.set_xlim(100, 3500)

            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                              length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                              width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                              length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                              width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_Ion_Temperature.png', dpi=dpi)

        return data_dict
    def recombinationRate(altRange, data_dict, **kwargs):

        model = vickrey1982()
        recombRate_vickrey = model.calcRecombinationRate(altRange, data_dict)
        model = schunkNagy2009()
        recombRate_schunkNagy = model.calcRecombinationRate(altRange, data_dict)

        data_dict = {**data_dict, **{'recombRate': [recombRate_vickrey, {'DEPEND_0': 'simAlt', 'UNITS': 'm^3s^-1',
                                                                            'LABLAXIS': 'Recombination Rate'}]} }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            ax.plot(recombRate_vickrey, altRange / xNorm, linewidth=Plot_LineWidth, label=r"Vickrey 1982")
            ax.plot(recombRate_schunkNagy, altRange / xNorm, linewidth=Plot_LineWidth, label=r"S&N 2009")
            ax.set_title('Ionospheric Recombination Rate', fontsize=Title_FontSize)
            ax.set_xlabel('Recombination Rate [m$^{3}$s$^{-1}$ ]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xscale('log')
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000 + 50, 100))
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                           length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                           width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                           length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                           width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_recombinationRate.png', dpi=dpi)

        return data_dict

    # --- PLASMA DENSITY ---
    # uses the Kletzing Model to return an np.array of plasma density (in m^-3) from [Alt_low, ..., Alt_High]
    def electron_plasmaDensityProfile(altRange,data_dict,**kwargs):


        if plasmaToggles.useStatic_ne_Profile:
            ne_density = np.array([plasmaToggles.staticDensity for alt in altRange])

        elif plasmaToggles.useIRI_ne_Profile:
            ne_density = 1E6*deepcopy(data_dict_IRI['Ne'][0]) # convert data into m^-3

        data_dict = {**data_dict,**{'ne': [ne_density, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'ne'}]}}


        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width,1 + figure_height/2)
            ax.plot(altRange/xNorm, ne_density,linewidth=Plot_LineWidth)
            ax.set_title('$n_{e}$ vs Altitude', fontsize = Title_FontSize)
            ax.set_ylabel('Electron Density [m$^{-3}$]', fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axvline(x=400000/xNorm,label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax.set_yscale('log')
            # ax.set_ylim(1E2, 1E6)
            ax.margins(0)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.grid(True)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_electron_density',dpi=dpi)

        return data_dict

    # --- Ion Mass ---
    def ion_plasmaDensityProfile(altRange,data_dict, **kwargs):

        if plasmaToggles.useIRI_ni_Profile:
            n_ions = 1E6*np.array([data_dict_IRI[f"{key}"][0] for key in plasmaToggles.wIons])# get the ion densities and convert them to m^-3
            m_eff_i = (np.sum((n_ions.T*Imasses).T, axis=0) / (np.sum(n_ions, axis=0)))
            ni_total = np.sum(n_ions, axis=0)

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
                ax[0].plot(altRange / xNorm, thing, label=f'${Ikeys[idx]}$ [$m^{-3}$]', linewidth=Plot_LineWidth)
            ax[0].plot(altRange / xNorm, data_dict['ne'][0], label=f'$ne$ [$m^{-3}$]', linewidth=Plot_LineWidth)
            ax[0].plot(altRange / xNorm, ni_total, label=f'$ni$ [$m^{-3}$]', linewidth=Plot_LineWidth)
            ax[0].set_title('Ion Plasma density vs Altitude', fontsize=Title_FontSize)
            ax[0].set_ylabel(r'Density [m$^{-3}$]', fontsize=Label_FontSize)
            ax[0].axvline(x=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax[0].set_yscale('log')
            ax[0].set_ylim(1E6,1E12)
            ax[0].legend(fontsize=Legend_fontSize,loc='upper right')

            ax[1].plot(altRange / xNorm, m_eff_i/ion_dict['proton'], linewidth=Plot_LineWidth)
            ax[1].set_ylabel('$m_{eff_{i}}/m_{Hp}$ [kg]', fontsize=Label_FontSize)
            ax[1].set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1].axvline(x=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax[1].set_ylim(10,35)
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
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_ionDensity.png', dpi=dpi)

        return data_dict

    # --- PLASMA BETA ---
    def plasmaBetaProfile(altRange,data_dict,**kwargs):

        m_eff_i = data_dict['m_eff_i'][0]
        plasmaBeta = (data_dict['ne'][0]*kB*data_dict['Te'][0])/((data_dict_Bgeo['Bgeo'][0])**2 /(2*u0))
        ratio = m_e/m_eff_i
        data_dict = {**data_dict, **{'beta_e': [plasmaBeta, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta_e'}]}}

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            colors= ['tab:red','tab:blue','tab:green']
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(plasmaBeta/ratio, altRange/xNorm, color=next(iter(colors)), label=r'$\beta_{i}$',linewidth=Plot_LineWidth)
            ax.set_title(r'$\beta$ vs Altitude',fontsize=Title_FontSize)
            ax.set_xlabel('Plasma Beta / (m_e/m_i)',fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axhline(y=400000/xNorm, label='Observation Height', color='red',linestyle='--',linewidth=Plot_LineWidth)
            ax.axvline(x=1, color='black')
            ax.set_xscale('log')
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
        plasmaDensity = data_dict['ne'][0]
        plasmaFreq = np.array([np.sqrt(plasmaDensity[i]* (q0*q0) / (ep0*m_e)) for i in range(len(plasmaDensity))])
        data_dict = {**data_dict,**{'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}]}}

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 1+ figure_height/2)
            ax.plot(plasmaFreq, altRange/xNorm, linewidth=Plot_LineWidth)
            ax.set_title('$\omega_{pe}$ vs Altitude',fontsize=Title_FontSize)
            ax.set_xlabel('Plasma Freq [rad/s]',fontsize=Label_FontSize)
            ax.set_xscale('log')
            ax.set_ylabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axhline(y=400000/xNorm,label='Observation Height',color='red',linewidth=Plot_LineWidth)
            plt.legend(fontsize=Legend_fontSize,loc='upper right')
            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\plasmaEnvironment\MODEL_plasFreq.png',dpi=dpi)

        return data_dict

    # --- ION CYCLOTRON FREQ ---
    def cyclotronProfile(altRange,data_dict,**kwargs):
        n_ions = np.array([data_dict[f"n_{key}"][0] for key in Ikeys])
        ionCyclotron_ions = np.array([q0 * data_dict_Bgeo['Bgeo'][0]  / mass for mass in Imasses])
        ionCyclotron_eff = np.sum(ionCyclotron_ions*n_ions,axis=0)/data_dict['ni'][0]
        electronCyclotron = q0 * data_dict_Bgeo['Bgeo'][0]/ m_e
        data_dict = {**data_dict,
                     **{'Omega_e': [electronCyclotron, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'electronCyclotron'}]},
                     **{'Omega_i_eff': [ionCyclotron_eff, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron_eff'}]},
                     **{f'Omega_{key}': [ionCyclotron_ions[idx], {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': f'ionCyclotron_{key}'}] for idx, key in enumerate(Ikeys)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            for idx,thing in enumerate(ionCyclotron_ions):
                ax[0].plot(thing, altRange / xNorm,  label=f'${Ikeys[idx]}$ [rad/s]', linewidth=Plot_LineWidth)

            ax[0].plot(ionCyclotron_eff, altRange / xNorm,  label='$\Omega_{eff}$ [rad/s]', linewidth=Plot_LineWidth)
            ax[0].set_title('$\Omega_{ci}$ vs Altitude', fontsize=Title_FontSize)
            ax[0].set_xlabel('$\Omega_{ci}$ [rad/s]', fontsize=Label_FontSize)
            ax[0].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[0].axhline(y=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax[0].set_xscale('log')
            ax[0].set_xlim(1E2, 1E4)
            ax[0].set_ylim(0, GenToggles.simAltHigh / xNorm)
            ax[0].grid(True)
            ax[0].margins(0)
            ax[0].legend(fontsize=Legend_fontSize,loc='upper right')

            for idx,thing in enumerate(ionCyclotron_ions):
                ax[1].plot(thing/ (2 * np.pi),altRange / xNorm,  label=f'${Ikeys[idx]}$ [Hz]', linewidth=Plot_LineWidth)

            ax[1].plot( ionCyclotron_eff / (2 * np.pi),altRange / xNorm, color='blue', label='$f_{avg}$', linewidth=Plot_LineWidth)
            ax[1].set_title('$f_{ci}$ vs Altitude', fontsize=Title_FontSize)
            ax[1].set_xlabel('$f_{ci}$ [Hz]', fontsize=Label_FontSize)
            ax[1].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1].axhline(y=400000 / xNorm, label='Observation Height', color='red', linewidth=Plot_LineWidth)
            ax[1].set_xscale('log')
            ax[1].set_xlim(1E2, 1E4)
            ax[1].set_ylim(0, GenToggles.simAltHigh / xNorm)
            ax[1].margins(0)
            ax[1].grid(True)
            ax[1].legend(fontsize=Legend_fontSize,loc='upper right')

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
        n_ions = np.array([data_dict[f"n_{key}"][0] for idx, key in enumerate(Ikeys)])
        vth_ions = np.array([np.sqrt(2)*np.sqrt(8 *kB* Ti /mass) for mass in Imasses]) # the np.sqrt(2) comes from the vector sum of two dimensions
        ionLarmorRadius_ions = np.array([vth_ions[idx] / data_dict[f"Omega_{key}"][0] for idx,key in enumerate(Ikeys)])
        ionLarmorRadius_eff = np.sum(n_ions*ionLarmorRadius_ions,axis=0)/data_dict['ni'][0]
        data_dict = {**data_dict,
                     **{'ionLarmorRadius_eff': [ionLarmorRadius_eff, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'ionLarmorRadius_eff'}]},
                     **{f'ionLarmorRadius_{key}': [ionLarmorRadius_ions[idx], {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': f'ionLarmorRadius_{key}'}] for idx, key in enumerate(Ikeys)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, 1+ figure_height/2)
            for idx, ion in enumerate(Ikeys):
                ax.plot(data_dict[f"ionLarmorRadius_{ion}"][0], altRange / xNorm,  label=rf"$\rho_{ion}$",linewidth=Plot_LineWidth)

            ax.plot(ionLarmorRadius_eff, altRange / xNorm,  label=r'$\rho_{avg}$', linewidth=Plot_LineWidth)
            ax.set_title(r'$\rho_{i}$ vs Altitude',fontsize=Title_FontSize)
            ax.set_xlabel(r'$\rho_{i}$ [m]',fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red',linestyle='--',linewidth=Plot_LineWidth)
            ax.set_xscale('log')
            ax.set_ylim(0, GenToggles.simAltHigh / xNorm)
            ax.margins(0)
            ax.set_xlim(1E0,2E2)
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
        VA_MHD = np.array(Bgeo/np.sqrt(u0*m_eff_i*ion_plasmaDensity))

        data_dict = {**data_dict,
                     **{'alfSpdMHD': [VA_MHD, {'DEPEND_0': 'simAlt', 'UNITS': 'm/s', 'LABLAXIS': 'alfSpdMHD'}]}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(VA_MHD/(10000*m_to_km), altRange / xNorm,  label='$V_{A} (MHD)$',linewidth=Plot_LineWidth)
            ax.set_title(r'$V_{A}$ vs Altitude',fontsize=Title_FontSize)
            ax.set_xlabel('MHD Alfven Speed  [10,000 km/s]',fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]',fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red', linestyle='--',linewidth=Plot_LineWidth)

            # plot some thermal velocity comparisons
            Vth_low = np.sqrt(8*q0*1/(9.11E-31))/(10000*m_to_km)
            ax.axvline(x=Vth_low, color='black',linewidth=Plot_LineWidth)
            ax.text(y=Re/xNorm,x=Vth_low*1.3,s='$V_{th_{e}}$ (1 eV)', color='black',fontsize=Text_Fontsize)

            Vth_high = np.sqrt(8 * q0 * 50 / (9.11E-31))/(10000*m_to_km)
            ax.axvline(x=Vth_high, color='black',linewidth=Plot_LineWidth)
            ax.text(y=Re/xNorm, x=Vth_high * 1.1, s='$V_{th_{e}}$ (50 eV)', color='black',fontsize=Text_Fontsize)

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
                        recombinationRate,
                        plasmaBetaProfile,
                        plasmaFreqProfile,
                        cyclotronProfile,
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
