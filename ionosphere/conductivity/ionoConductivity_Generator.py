# --- ionoConductivity_Generator.py ---
# Description: Model the ionospheric conductivity



# --- imports ---
import scipy.interpolate

from ionosphere.simToggles_iono import GenToggles, conductivityToggles,plasmaToggles,neutralsToggles
from spaceToolsLib.variables import u0,m_e,ep0,cm_to_m,IonMasses,q0, m_to_km,Re,kB,m_Hp,m_Op,m_Np,m_Hep,m_NOp,m_O2p,m_N2p,NeutralMasses
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from ionosphere.conductivity.model_conductivity_classes import *
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

def generateIonosphericConductivity(outputData,GenToggles,conductivityToggles, **kwargs):
    data_dict = {}

    # --- choose your dataset ---
    ionKeys = ['Op', 'Hp', 'Hep', 'O2p', 'NOp', 'Np']
    ionMasses = np.array(IonMasses[1:6+1])
    # neutralKeys = ['N2','O2','O','HE','H','AR','N','NO'] # NOTE: does NOT include anomolous O+
    # neutralMasses = NeutralMasses

    # --- get some ubiquitous data variables ---
    n_ions = np.array([data_dict_plasma[f"n_{key}"][0] for key in ionKeys])
    n_neutrals = np.array([data_dict_neutral[f"{key}"][0] for key in neutralsToggles.neutralKeys])

    if conductivityToggles.useIRI_ne_profile:
        ne = data_dict_plasma['ne'][0]

    elif conductivityToggles.useHeight_Ionization_ne_profile:
        raise Exception('Need to code this still!')


    # --- Electron mobility ---
    def electron_collisionFreqProfile(altRange,data_dict,**kwargs):

        # electron-neutral collisions
        model = Nicolet1953()
        nu_en = model.electronNeutral_CollisionFreq(data_dict_neutral, data_dict_plasma)
        for thing in nu_en:
            print(thing)

        # electron-ion collisions
        model = Johnson1961()
        nu_ei = model.electronIon_CollisionFreq(data_dict_neutral,data_dict_plasma)

        # total collision fre
        nu_e_total = nu_ei + nu_en

        data_dict = {**data_dict,
                     **{'nu_e_total': [nu_e_total, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'nu_e'}]},
                     # **{f"nu_e_{key}": [nu_en[idx], {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': f"nu_e_{key}"}] for idx,key in enumerate(neutralsToggles.neutralKeys)}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            ax.plot(nu_en,altRange / xNorm,  linewidth=Plot_LineWidth,label=fr"$\nu$: e-n")
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

        model = Johnson1961()
        nu_in = np.array(model.ionNeutral_CollisionsFreq(data_dict_neutral, data_dict_plasma))

        # total collision freq
        nu_i_total = nu_in


        data_dict = {**data_dict,
                     **{'nu_i_total': [nu_i_total, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'Total Ion Collisions'}]}
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
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
        nu_e = data_dict['nu_e_total'][0]
        nu_i = data_dict['nu_i_total'][0]
        Omega_i_eff = data_dict_plasma['Omega_i_eff'][0]
        Omega_e = data_dict_plasma['Omega_e'][0]
        ionMobility = Omega_i_eff/nu_i
        elecMobility = Omega_e/nu_e


        data_dict = {**data_dict,
                     **{'kappa_i': [ionMobility, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'ionMobility'}]},
                     **{'kappa_e': [elecMobility, {'DEPEND_0': 'simAlt', 'UNITS': '1/s', 'LABLAXIS': 'elecMobility'}]},
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            ax.plot(data_dict['kappa_i'][0], altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\kappa_{i}$")
            ax.plot(data_dict['kappa_e'][0], altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\kappa_{e}$")
            ax.set_title('Ion/Electron Mobility vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('Mobility', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xscale('log')
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
            plt.savefig(f'{GenToggles.simFolderPath}\conductivity\MODEL_mobility.png', dpi=dpi)

        # update the data_dict
        return data_dict

    def ionosphericConductivityProfile(altRange, data_dict, **kwargs):

        kappa_i = data_dict['kappa_i'][0]
        kappa_e = data_dict['kappa_e'][0]
        B_geo = data_dict_Bgeo['Bgeo'][0]
        ne = data_dict_plasma['ne'][0]
        sigmaPedersen = (ne * q0 / B_geo) * ((kappa_i / (1 + kappa_i ** 2)) + (kappa_e) / (1 + kappa_e ** 2))
        sigmaHall = (ne*q0/B_geo)*((kappa_e**2)/(1 + kappa_e**2) - (kappa_i**2)/(1 + kappa_i**2))
        sigmaParallel = (ne*q0/B_geo)*(kappa_i + kappa_e)

        # height-integrated conductivities
        from scipy.integrate import trapz
        SigmaHall = np.array([trapz(y=sigmaHall,x=altRange)])
        SigmaPedersen = np.array([trapz(y=sigmaPedersen,x=altRange)])


        data_dict = {**data_dict,
                     **{'sigma_P': [sigmaPedersen, {'DEPEND_0': 'simAlt', 'UNITS': 'mho/m', 'LABLAXIS': 'Pedersen Conductivity'}]},
                     **{'sigma_H': [sigmaHall, {'DEPEND_0': 'simAlt', 'UNITS': 'mho/m', 'LABLAXIS': 'Hall Conductivity'}]},
                     **{'Sigma_P': [SigmaPedersen, {'DEPEND_0': 'simAlt', 'UNITS': 'mho', 'LABLAXIS': 'Height-Integrated Pedersen Conductivity'}]},
                     **{'Sigma_H': [SigmaHall, {'DEPEND_0': 'simAlt', 'UNITS': 'mho', 'LABLAXIS': 'Height-Integrated Hall Conductivity'}]},
                     **{'sigma_Par': [sigmaParallel, {'DEPEND_0': 'simAlt', 'UNITS': 'mho/m', 'LABLAXIS': 'Parallel Conductivity'}]},
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height * (3 / 2))
            ax.plot(data_dict['sigma_P'][0]*1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\sigma_{P}$",color='red')
            ax.plot(data_dict['sigma_H'][0]*1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\sigma_{H}$",color='blue')
            ax.plot(data_dict['sigma_Par'][0]*1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$\sigma_{0}$                                                                                                                                                                                                                 ", color='green')
            ax.axvline(data_dict['Sigma_P'][0], linewidth=Plot_LineWidth, label=r"$\Sigma_{P}$",color='tab:red')
            ax.axvline(data_dict['Sigma_H'][0], linewidth=Plot_LineWidth, label=r"$\Sigma_{H}$",color='tab:blue')
            ax.set_title('Ionospheric Conductivity vs Altitude', fontsize=Title_FontSize)
            ax.set_xlabel('conductivity [$10^{-6}$ mho/m]', fontsize=Label_FontSize)
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
    if kwargs.get('showPlot', False):

        # --- collect all the functions ---
        profileFuncs = [electron_collisionFreqProfile,
                        ion_collisionFreqProfile,
                        mobilityProfile,
                        ionosphericConductivityProfile
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
