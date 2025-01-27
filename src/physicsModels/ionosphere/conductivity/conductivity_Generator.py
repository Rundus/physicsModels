# --- conductivity_Generator.py ---
# Description: Model the ionospheric conductivity



# --- imports ---

from src.physicsModels.ionosphere.simToggles_Ionosphere import GenToggles, plasmaToggles,neutralsToggles
from spaceToolsLib.variables import m_to_km,Re
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.physicsModels.ionosphere.conductivity.conductivity_classes import *
import numpy as np
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata


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
Legend_fontSize = 30
dpi = 100

xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'

# get the geomagnetic field data dict
data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simFolderPath}\geomagneticField\geomagneticField.cdf')

# get the ionospheric neutral data dict
data_dict_neutral = loadDictFromFile(rf'{GenToggles.simFolderPath}\neutralEnvironment\neutralEnvironment.cdf')

# get the ionospheric plasma data dict
data_dict_plasma = loadDictFromFile(rf'{GenToggles.simFolderPath}\plasmaEnvironment\plasmaEnvironment.cdf')

def generateIonosphericConductivity(outputData,GenToggles, conductivityToggles, **kwargs):
    data_dict = {}

    # --- Electron mobility ---
    def electron_collisionFreqProfile(altRange,data_dict,**kwargs):

        # electron-neutral collisions
        model = Leda2019()
        nu_en = [model.electronNeutral_CollisionFreq(data_dict_neutral= data_dict_neutral,
                                                     data_dict_plasma= data_dict_plasma,
                                                     neutralKey=key) for key in neutralsToggles.wNeutrals]

        # electron-ion collisions
        model = Johnson1961()
        nu_ei = model.electronIon_CollisionFreq(data_dict_neutral,data_dict_plasma)


        # total collision fre
        nu_e_total = nu_ei + np.sum(nu_en, axis=0)

        data_dict = {**data_dict,
                     **{'nu_e_total': [nu_e_total, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'nu_e'}]}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            for idx,key in enumerate(neutralsToggles.wNeutrals):
                ax.plot(nu_en[idx],altRange / xNorm,  linewidth=Plot_LineWidth,label=fr"$\nu$: e-n ({key})")
            ax.plot(nu_ei,altRange / xNorm,  linewidth=Plot_LineWidth, label=rf"$\nu$: e-i")
            ax.plot(nu_e_total,altRange / xNorm,  linewidth=Plot_LineWidth, label=rf"$\nu$: e-total")
            ax.set_title('Electron Collision Freq. vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('Collision Freq [1/s]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.set_xscale('log')
            ax.set_xlim(1E-2, 1E5)
            ax.yaxis.set_ticks(np.arange(0, 1000+50, 100))
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
            plt.savefig(f'{GenToggles.simFolderPath}\conductivity\MODEL_Electron_CollisionFreq.png', dpi=dpi)

        # update the data_dict
        return data_dict

    def ion_collisionFreqProfile(altRange, data_dict, **kwargs):

        model = Leda2019()
        nu_in = [model.ionNeutral_CollisionsFreq(data_dict_neutral= data_dict_neutral,
                                                 data_dict_plasma= data_dict_plasma,
                                                 ionKey=key) for key in plasmaToggles.wIons] # NOp, Op, O2p

        # total collision freq
        nu_i_total = np.sum(nu_in,axis=0)

        data_dict = {**data_dict,
                     **{'nu_i_total': [nu_i_total, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'Total Ion Collisions'}]},
                     **{f'nu_{key}': [nu_in[idx], {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': f'{key} Collisions'}] for idx,key in enumerate(plasmaToggles.wIons)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))

            for idx, key in enumerate(plasmaToggles.wIons):
                ax.plot(data_dict[f"nu_{key}"][0], altRange / xNorm, linewidth=Plot_LineWidth, label=rf"$\nu$: i-n ({key})")

            ax.plot(data_dict['nu_e_total'][0], altRange / xNorm, linewidth=Plot_LineWidth, label=rf"$\nu$: e-total")
            ax.plot(data_dict['nu_i_total'][0], altRange / xNorm,  linewidth=Plot_LineWidth, label=rf"$\nu$: i-total")
            ax.set_title('Ion Collision Freq. vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('Collision Freq [1/s]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.set_xscale('log')
            ax.set_xlim(1E-2, 1E5)
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000+50, 100))

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
            plt.savefig(f'{GenToggles.simFolderPath}\conductivity\MODEL_Ion_CollisionFreq.png', dpi=dpi)

        # update the data_dict
        return data_dict

    def mobilityProfile(altRange, data_dict,**kwargs):

        # electrons
        nu_e = data_dict['nu_e_total'][0]
        Omega_e = data_dict_plasma['Omega_e'][0]
        elecMobility = Omega_e / nu_e

        # ions
        ionMobility = [ data_dict_plasma[f"Omega_{key}"][0]/data_dict[f"nu_{key}"][0] for key in plasmaToggles.wIons]
        ionMobility_eff = data_dict_plasma['Omega_i_eff'][0]/data_dict["nu_i_total"][0]

        data_dict = {**data_dict,
                     **{f'kappa_i_eff': [ionMobility_eff, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': f'ion Mobility (effective)'}]},
                     **{f'kappa_{key}': [ionMobility[idx], {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': f'{key} Mobility'}] for idx,key in enumerate(plasmaToggles.wIons)},
                     **{'kappa_e': [elecMobility, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'elecMobility'}]},
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            for idx, key in enumerate(plasmaToggles.wIons):
                ax.plot(data_dict[f"kappa_{key}"][0], altRange / xNorm, linewidth=Plot_LineWidth, label=rf"$\kappa_{key}$")
            ax.plot(data_dict['kappa_i_eff'][0], altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\kappa_{i}$")
            ax.plot(data_dict['kappa_e'][0], altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\kappa_{e}$")
            ax.set_title('Ion/Electron Mobility vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('Mobility', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xscale('log')
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000+50, 100))
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\conductivity\MODEL_mobility.png', dpi=dpi)

        # update the data_dict
        return data_dict

    def ionosphericConductivityProfile(altRange, data_dict, **kwargs):

        B_geo = data_dict_Bgeo['Bgeo'][0]

        # calculate the specific sigmas for each ion/electron with format: # [ [parallel,Pedersen and Hall], [...], [...] ]
        ionsSigmas = [ (data_dict_plasma[f"n_{key}"][0] * q0 / B_geo) * np.array([data_dict[f'kappa_{key}'][0], data_dict[f'kappa_{key}'][0]/(1 + np.power(data_dict[f'kappa_{key}'][0],2)), np.power(data_dict[f'kappa_{key}'][0],2)/(1 + np.power(data_dict[f'kappa_{key}'][0],2))]) for key in plasmaToggles.wIons]
        elecSigmas = (data_dict_plasma[f"ne"][0] * q0 / B_geo) * np.array([data_dict[f'kappa_e'][0], data_dict[f'kappa_e'][0]/(1 + np.power(data_dict[f'kappa_e'][0],2)), np.power(data_dict[f'kappa_e'][0],2)/(1 + np.power(data_dict[f'kappa_e'][0],2))])

        sigmaPedersen = np.sum([ionsSigmas[idx][1] for idx in range(len(plasmaToggles.wIons))],axis=0) + elecSigmas[1]
        sigmaHall =  - np.sum([ionsSigmas[idx][2] for idx in range(len(plasmaToggles.wIons))],axis=0) + elecSigmas[2]
        sigmaParallel = np.sum([ionsSigmas[idx][0] for idx in range(len(plasmaToggles.wIons))],axis=0) + elecSigmas[0]

        # height-integrated conductivities
        from scipy.integrate import trapz
        heightIntegratedHall = np.array([trapz(y=sigmaHall,x=altRange)])
        heightIntegratedPedersen = np.array([trapz(y=sigmaPedersen,x=altRange)])
        data_dict = {**data_dict,
                     **{'sigma_P_total': [sigmaPedersen, {'DEPEND_0': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Pedersen Conductivity'}]},
                     **{'sigma_H_total': [sigmaHall, {'DEPEND_0': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Hall Conductivity'}]},
                     **{'heightIntegrated_H': [heightIntegratedHall, {'DEPEND_0': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                     **{'heightIntegrated_P': [heightIntegratedPedersen, {'DEPEND_0': 'simAlt', 'UNITS': 'S', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                     **{'sigma_Par': [sigmaParallel, {'DEPEND_0': 'simAlt', 'UNITS': 'S/m', 'LABLAXIS': 'Parallel Conductivity'}]},
                     **{f"sigma_P_{key}": [ionsSigmas[idx][1], {'DEPEND_0': 'simAlt', 'UNITS': 'S', 'LABLAXIS': f'Pedersen Conductivity ({key})'}] for idx,key in enumerate(plasmaToggles.wIons)},
                     **{f"sigma_H_{key}": [ionsSigmas[idx][2], {'DEPEND_0': 'simAlt', 'UNITS': 'S', 'LABLAXIS': f'Hall Conductivity ({key})'}] for idx, key in enumerate(plasmaToggles.wIons)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            ax.plot(data_dict['sigma_P_total'][0]*1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\sigma_{P}$",color='red')
            ax.plot(data_dict['sigma_H_total'][0]*1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\sigma_{H}$",color='blue')
            ax.plot(data_dict['sigma_Par'][0]*1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\sigma_{0}$                                                                                                                                                                                                                 ", color='green')
            ax.axvline(data_dict['heightIntegrated_P'][0], linewidth=Plot_LineWidth, label=r"$\Sigma_{P}$",color='tab:red')
            ax.axvline(data_dict['heightIntegrated_H'][0], linewidth=Plot_LineWidth, label=r"$\Sigma_{H}$",color='tab:blue')
            ax.set_title('Ionospheric Conductivity vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('conductivity [$10^{-6}$ S/m]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xscale('log')
            ax.set_xlim(1E-2,1E8)
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000+50, 100))

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
            plt.savefig(f'{GenToggles.simFolderPath}\conductivity\MODEL_conductivity.png', dpi=dpi)

        return data_dict




    ##################
    # --- PLOTTING ---
    ##################

    def makePlotOfModel(altRange, data_dict, **kwargs):


        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2,ncols=3)
            fig.set_size_inches(12, 8)

            conductivityScaleFactor = 1E4

            # plot all the data
            for key in neutralsToggles.wNeutrals:
                ax[0, 0].plot(data_dict_neutral[f'{key}'][0], altRange / xNorm, linewidth=Plot_LineWidth,label=key)  # neutrals

            for key in plasmaToggles.wIons:
                ax[0, 1].plot(data_dict_plasma[f'n_{key}'][0], altRange / xNorm, linewidth=Plot_LineWidth,label=key) # ions
                ax[1, 1].plot(data_dict[f'sigma_P_{key}'][0]*conductivityScaleFactor, altRange / xNorm, linewidth=Plot_LineWidth,label=key) # ion pedersen conductivites
                ax[1, 2].plot(data_dict[f'sigma_H_{key}'][0]*conductivityScaleFactor, altRange / xNorm, linewidth=Plot_LineWidth,label=key)  # ion Hall conductivites


            ax[1,1].set_xlabel('Pedersen Conductivity [S/10km]')

            # neutrals - fine adjustments
            ax[0, 0].set_ylabel('Altitude [km]')
            ax[0, 0].set_xlim(1E12, 1E21)
            ax[0, 0].set_xscale('log')
            ax[0, 0].set_xlabel('Number density [m$^{-3}$]')

            # ions - fine adjustments
            ax[0, 1].set_xlim(1E9, 1E12)
            ax[0, 1].set_xscale('log')
            ax[0, 1].set_xlabel('Number density [m$^{-3}$]')

            # temperature
            ax[0,2].set_xlim(0, 1800)
            ax[0, 2].plot(data_dict_plasma[f'Ti'][0], altRange / xNorm, linewidth=Plot_LineWidth,label='$T_{i}$')  # temperature
            ax[0, 2].plot(data_dict_plasma[f'Te'][0], altRange / xNorm, linewidth=Plot_LineWidth,label='$T_{e}$')  # temperature
            ax[0, 2].plot(data_dict_neutral[f'Tn'][0], altRange / xNorm, linewidth=Plot_LineWidth,label='$T_{n}$')  # temperature

            # Total Pedersen/Hall Conductivity
            ax[1, 0].plot(data_dict['sigma_P_total'][0]*conductivityScaleFactor, altRange / xNorm, linewidth=Plot_LineWidth,label='$\sigma_{P}$')
            ax[1, 0].plot(data_dict['sigma_H_total'][0]*conductivityScaleFactor, altRange / xNorm, linewidth=Plot_LineWidth,label='$\sigma_{H}$')
            ax[1, 0].set_xlabel('Conductivity [S/10km]')

            # Hall conductivity
            ax[1, 2].set_xlabel('Hall Conductivity [S/10km]')

            for i in range(2):
                for j in range(3):
                    ax[i,j].set_ylim(60,300)
                    ax[i,j].legend()

            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\conductivity\MODEL_profile.png', dpi=dpi)

        return data_dict



    if kwargs.get('showPlot', False):

        # --- collect all the functions ---
        profileFuncs = [electron_collisionFreqProfile,
                        ion_collisionFreqProfile,
                        mobilityProfile,
                        ionosphericConductivityProfile,
                        makePlotOfModel
                        ]

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

        outputPath = rf'{GenToggles.simFolderPath}\conductivity\conductivity.cdf'
        outputCDFdata(outputPath, data_dict)
