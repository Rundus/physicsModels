# --- verify_IonizationRecombinationModels.py ---
# Description: Use a ionization_recombination Method to create electron density
# altitude profiles via Fang Parameterization or RangeDepth Methods. Offers
# ability to insert real ESA data or made up data

# --- imports ---
import time
start_time = time.time()
from src.physicsModels.ionosphere.simToggles_Ionosphere import *
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.physicsModels.ionosphere.ionization_recombination.ionizationRecomb_classes import *
from src.physicsModels.ionosphere.plasma_environment.plasmaEnvironment_classes import *
import numpy as np
from copy import deepcopy
from spaceToolsLib.tools.CDF_output import outputCDFdata

##################
# --- PLOTTING ---
##################
# --- Plotting Toggles ---
figure_width = 20  # in inches
figure_height = 20  # in inches
Title_FontSize = 25
Label_FontSize = 25
Tick_FontSize = 25
Tick_FontSize_minor = 20
Tick_Length = 10
Tick_Width = 2
Tick_Length_minor = 5
Tick_Width_minor = 1
Text_Fontsize = 20
Plot_LineWidth = 6.5
Legend_fontSize = 32
dpi = 100

xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'

# get the ionospheric neutral data dict
data_dict_neutral = loadDictFromFile(rf'{neutralsToggles.outputFolder}\neutralEnvironment.cdf')

# get the ionospheric plasma data dict
data_dict_plasma = loadDictFromFile(rf'{plasmaToggles.outputFolder}\plasmaEnvironment.cdf')


# TOGGLES
use_real_data = True
if use_real_data:
    data_dict_backScatter = loadDictFromFile(rf'C:\Data\physicsModels\invertedV\backScatter\backScatter.cdf')

def generateHeightIonization(GenToggles, ionizationRecombToggles, **kwargs):
    data_dict = {}

    # --- Electron mobility ---
    def electron_heightIonization(altRange, data_dict, **kwargs):

        if use_real_data:
            tmeIdx = 0
            beam_energyGrid = deepcopy(data_dict_backScatter['beam_energy_Grid'][0][tmeIdx])
            response_energyGrid = deepcopy(data_dict_backScatter['energy_Grid'][0])
            engy_flux_beam = np.multiply(deepcopy(data_dict_backScatter['num_flux_beam'][0][tmeIdx]), beam_energyGrid / 1000)
            engy_flux_sec = np.multiply(deepcopy(data_dict_backScatter['num_flux_sec'][0][tmeIdx]), response_energyGrid / 1000)
            engy_flux_dgdPrim = np.multiply(deepcopy(data_dict_backScatter['num_flux_dgdPrim'][0][tmeIdx]), response_energyGrid / 1000)

            # --- Get the energy/energyFluxes of the incident beam + backscatter electrons ---
            monoEnergyProfile = np.append(response_energyGrid / 1000, beam_energyGrid / 1000)  # IN UNITS OF KEV
            energyFluxProfile = np.append(engy_flux_dgdPrim + engy_flux_sec, engy_flux_beam)
        else:
            # IF COPYING FANG MODEL:
            monoEnergyProfile = np.array([0.01, 0.1, 1, 10, 100, 1000])  # 100eV and 100keV, IN UNITS OF KEV
            energyFluxProfile = (6.242E8) * np.array([1 for i in range(len(monoEnergyProfile))])  # provide in ergs but convert from ergs/cm^-2s^-1 to keV/cm^-2s^-1

        # CHOOSE THE MODEL
        model = fang2010(altRange, data_dict_neutral, data_dict_plasma, monoEnergyProfile, energyFluxProfile)
        H = model.scaleHeight()
        y = model.atmColumnMass(monoEnergyProfile)
        f = model.f(y,model.calcCoefficents(monoEnergyProfile))
        q_profiles, q_total = model.ionizationRate() # in cm^-3 s^-1

        data_dict = {**data_dict,
                     **{'ionizingEnergy': [monoEnergyProfile, {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                        'energyFlux': [energyFluxProfile, {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                         'columnMass': [y, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'y'}],
                        'scaleHeight': [H, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'H'}],
                        'q_total':[q_total, {'DEPEND_0': 'simAlt', 'UNITS': 'm^-3s^-1', 'LABLAXIS': 'qtot'}],
                        },
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2,ncols=2)
            fig.set_size_inches(figure_width, figure_height)

            ax[0, 0].plot(H/(100*1000), altRange / xNorm,  linewidth=Plot_LineWidth, label='H')
            ax[0, 0].set_xlabel('Scale Height [km]', fontsize=Label_FontSize)
            ax[0, 1].set_xlabel('Column Mass (y)', fontsize=Label_FontSize)
            ax[0, 1].set_xlim(1E-6, 1E6)

            for idx, profile in enumerate(q_profiles):
                ax[0, 1].plot(y[idx], altRange / xNorm, linewidth=Plot_LineWidth)
                ax[1, 0].plot(f[idx], y[idx],  linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")
                ax[1, 1].plot(q_profiles[idx], altRange / xNorm,  linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")

            ax[1, 1].plot(q_total, altRange / xNorm,  linewidth=Plot_LineWidth, linestyle='--', label=rf"Total")
            ax[1, 1].set_xlabel('Total ionization Rate [cm$^{-3}$s$^{-1}$]', fontsize=Label_FontSize)
            ax[0, 0].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1, 0].set_ylabel(f'Column Mass (y)', fontsize=Label_FontSize)
            ax[1, 0].set_xlabel(f'Energy Dissipation (f)', fontsize=Label_FontSize)
            ax[1, 1].set_xlim(1E-4, 1E5)
            ax[1, 0].set_xlim(0, 0.8)

            for i in range(2):
                for j in range(2):
                    if [i, j] != [1,0]:
                        ax[i, j].yaxis.set_ticks(np.arange(0, 1000+50, 50))
                        ax[i, j].set_ylim(50, 500)
                        if [i, j] != [0, 0]:
                            ax[i, j].set_xscale('log')
                        else:
                            ax[i, j].set_xlim(0,100)
                    else:
                        ax[i, j].set_ylim(0.1, 10)
                        ax[i, j].set_yscale('log')

                    ax[i, j].grid(True)
                    ax[i, j].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                      length=Tick_Length)
                    ax[i, j].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                                      width=Tick_Width_minor, length=Tick_Length_minor)
                    ax[i, j].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                      length=Tick_Length)
                    ax[i, j].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                                      width=Tick_Width_minor, length=Tick_Length_minor)
                    if not use_real_data:
                        ax[i, j].legend(fontsize=Legend_fontSize, loc='upper right')
            plt.tight_layout()
            if use_real_data:
                file_name = rf'{ionizationRecombToggles.outputFolder}\testScripts\MODEL_heightIonization_real_data.png'
            else:
                file_name = rf'{ionizationRecombToggles.outputFolder}\testScripts\MODEL_heightIonization.png'
            plt.savefig(file_name, dpi=dpi)

        # update the data_dict
        return data_dict

    def recombinationRate(altRange, data_dict, **kwargs):

        model = vickrey1982()
        recombRate_vickrey, profiles_vickrey = model.calcRecombinationRate(altRange, data_dict_plasma) # NOTE: Vickrey is only valid between 0 to 200km.
        model = schunkNagy2009()
        recombRate_schunkNagy, profiles_schunkNagy = model.calcRecombinationRate(altRange, data_dict_plasma)
        data_dict = {**data_dict,
                     **{
                         'recombRate_vickrey': [recombRate_vickrey, {'DEPEND_0': 'simAlt', 'UNITS': 'm^3s^-1', 'LABLAXIS': 'Recombination Rate'}],
                         'recombRate_schunkNaggy': [recombRate_schunkNagy, {'DEPEND_0': 'simAlt', 'UNITS': 'm^3s^-1', 'LABLAXIS': 'Recombination Rate'}]
                        }
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2,sharex=True)
            fig.set_size_inches(figure_width, figure_height)


            # recombination rate - total
            ax[0].plot(recombRate_vickrey/1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"Vickrey 1982")
            ax[0].plot(recombRate_schunkNagy/1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"S&N 2009")
            ax[0].set_title('Ionospheric Recombination Rate', fontsize=Title_FontSize)
            ax[0].set_xlabel('Recombination Rate [cm$^{3}$s$^{-1}$ ]', fontsize=Label_FontSize)
            ax[0].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[0].axhline(y=400000 / xNorm, label='Observation Height', color='red')
            # ax.set_xlim(1E-14, 1E-6)
            ax[0].set_xscale('log')
            ax[0].grid(True)
            ax[0].yaxis.set_ticks(np.arange(0, 1000 + 50, 100))
            ax[0].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax[0].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax[0].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax[0].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax[0].legend(fontsize=Legend_fontSize)

            # recombination rate - profiles
            for idx, ionNam in enumerate(plasmaToggles.wIons):
                ax[1].plot(profiles_schunkNagy[idx], altRange / xNorm, linewidth=Plot_LineWidth, label=ionNam)

            ax[1].set_title('Ionospheric Recombination Rate', fontsize=Title_FontSize)
            ax[1].set_xlabel('Recombination Rate [cm$^{3}$s$^{-1}$ ]', fontsize=Label_FontSize)
            ax[1].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1].axhline(y=400000 / xNorm, label='Observation Height', color='red')
            # ax.set_xlim(1E-14, 1E-6)
            ax[1].set_xscale('log')
            ax[1].grid(True)
            ax[1].yaxis.set_ticks(np.arange(0, 1000 + 50, 100))
            ax[1].tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax[1].tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax[1].tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax[1].tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax[1].legend(fontsize=Legend_fontSize)

            plt.tight_layout()
            plt.savefig(rf'{ionizationRecombToggles.outputFolder}\testScripts\MODEL_recombinationRate.png', dpi=dpi)

        return data_dict

    def electronDensity_fromIonizationRecomb(altRange, data_dict, **kwargs):

        q_total = data_dict['q_total'][0] # sum up the recombination rates from all the incoming electrons

        alpha_SN = data_dict['recombRate_schunkNaggy'][0]
        n_e_SN = np.sqrt(q_total/alpha_SN) # in cm^-3

        alpha_vic = data_dict['recombRate_vickrey'][0]
        n_e_vic = np.sqrt(q_total / alpha_vic)  # in cm^-3

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(n_e_SN, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$n_{e}$ (S&N) ")
            ax.plot(n_e_vic, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$n_{e}$ (Vickrey 1982)")

            ax.set_title('Model Ionospheric Electron Density', fontsize=Title_FontSize)
            ax.set_xlabel('Electron Density [cm$^{3}$]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xlim(1E2, 1E10)
            ax.set_xscale('log')
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000 + 50, 100))
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(rf'{ionizationRecombToggles.outputFolder}\testScripts\MODEL_electronDensityFromIonizationRecomb.png', dpi=dpi)

        return data_dict


    ##################
    # --- PLOTTING ---
    ##################

    if kwargs.get('showPlot', False):

        # --- collect all the functions ---
        profileFuncs = [electron_heightIonization,
                        recombinationRate,
                        electronDensity_fromIonizationRecomb
                        ]

        for i in range(len(profileFuncs)):
            data_dict = profileFuncs[i](altRange=GenToggles.simAlt, data_dict=data_dict, showPlot=True)

    #####################
    # --- OUTPUT DATA ---
    #####################

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

    outputPath = rf'{ionizationRecombToggles.outputFolder}\testScripts\ionizationRecomb_TestScripts.cdf'
    outputCDFdata(outputPath, data_dict)


# --- EXECUTE ---
stl.prgMsg('Regenerating Height Ionization and Recombination')
generateHeightIonization(GenToggles, ionizationRecombToggles, showPlot=True)
stl.Done(start_time)