# --- ionizationRecomb_Generator.py ---
# Description: Use a ionizationRecomb Method to create electron density
# altitude profiles via Fang Parameterization or RangeDepth Methods. Offers
# ability to insert real ESA data or made up data



# --- imports ---

from src.physicsModels.ionosphere.simToggles_Ionosphere import GenToggles
from spaceToolsLib.variables import m_to_km
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from src.physicsModels.ionosphere.ionizationRecomb.ionizationRecomb_classes import *
from src.physicsModels.ionosphere.PlasmaEnvironment.plasmaEnvironment_classes import *
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
Plot_LineWidth = 2.5
Legend_fontSize = 32
dpi = 100

xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'

# get the ionospheric neutral data dict
data_dict_neutral = loadDictFromFile(rf'{GenToggles.simFolderPath}\neutralEnvironment\neutralEnvironment.cdf')

# get the ionospheric plasma data dict
data_dict_plasma = loadDictFromFile(rf'{GenToggles.simFolderPath}\plasmaEnvironment\plasmaEnvironment.cdf')

def generateHeightIonization(GenToggles, heightIonizationToggles, **kwargs):
    data_dict = {}

    # --- Electron mobility ---
    def electron_heightIonization(altRange, data_dict, **kwargs):

        monoEnergyProfile = np.array([0.01, 0.1, 1, 10, 100, 1000])  # 100eV and 100keV, IN UNITS OF KEV
        energyFluxProfile = (6.242E8) * np.array([1 for i in range(len(monoEnergyProfile))])  # provide in ergs but convert from ergs/cm^-2s^-1 to keV/cm^-2s^-1

        # CHOOSE THE MODEL
        model = fang2010(altRange, data_dict_neutral, data_dict_plasma, monoEnergyProfile, energyFluxProfile)
        H = model.scaleHeight()
        y = model.atmColumnMass(monoEnergyProfile)
        f = model.f(y,model.calcCoefficents(monoEnergyProfile))
        qtot = model.ionizationRate() # in m^-3 s^-1

        data_dict = {**data_dict,
                     **{'ionizingEnergy': [monoEnergyProfile, {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                        'energyFlux': [energyFluxProfile, {'DEPEND_0': None, 'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
                         'columnMass': [y, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'y'}],
                        'scaleHeight': [H, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'H'}],
                        'qtot':[qtot, {'DEPEND_0': 'simAlt', 'UNITS': 'm^-3s^-1', 'LABLAXIS': 'qtot'}]
                        },
                     }

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2,ncols=2)
            fig.set_size_inches(figure_width, figure_height)

            # convert q to cm^-3
            qtot = qtot/(np.power(stl.cm_to_m, 3))

            ax[0, 0].plot(H/(100*1000), altRange / xNorm,  linewidth=Plot_LineWidth, label='H')
            ax[0, 0].set_xlabel('Scale Height [km]', fontsize=Label_FontSize)
            ax[0, 1].set_xlabel('Column Mass (y)', fontsize=Label_FontSize)

            for idx, profile in enumerate(qtot):
                ax[0, 1].plot(y[idx], altRange / xNorm, linewidth=Plot_LineWidth)
                ax[1, 0].plot(f[idx], y[idx],  linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")
                ax[1, 1].plot(qtot[idx], altRange / xNorm,  linewidth=Plot_LineWidth, label=rf"{monoEnergyProfile[idx]} keV")

            ax[1, 1].plot(np.sum(qtot, axis=0), altRange / xNorm,  linewidth=Plot_LineWidth, linestyle='--', label=rf"Total")
            ax[1, 1].set_xlabel('Total ionization Rate [cm$^{-3}$s$^{-1}$]', fontsize=Label_FontSize)
            ax[0, 0].set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax[1, 0].set_ylabel(f'Column Mass (y)', fontsize=Label_FontSize)
            ax[1, 0].set_xlabel(f'Energy Dissipation (f)', fontsize=Label_FontSize)
            ax[1, 1].set_xlim(1E1, 1E5)
            ax[1, 0].set_xlim(0, 0.8)

            for i in range(2):
                for j in range(2):
                    if [i, j] != [1,0]:
                        ax[i, j].yaxis.set_ticks(np.arange(0, 1000+50, 50))
                        ax[i, j].set_ylim(50, 500)
                        if [i, j] != [0,0]:
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

                    ax[i, j].legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\ionizationRecomb\MODEL_heightIonization.png', dpi=dpi)

        # update the data_dict
        return data_dict

    def recombinationRate(altRange, data_dict, **kwargs):

        model = vickrey1982()
        recombRate_vickrey = model.calcRecombinationRate(altRange, data_dict_plasma) # NOTE: Vickrey is only valid between 0 to 200km.
        model = schunkNagy2009()
        recombRate_schunkNagy = model.calcRecombinationRate(altRange, data_dict_plasma)
        data_dict = {**data_dict, **{'recombRate': [recombRate_schunkNagy, {'DEPEND_0': 'simAlt', 'UNITS': 'm^3s^-1', 'LABLAXIS': 'Recombination Rate'}]}}

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(recombRate_vickrey/1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"Vickrey 1982")
            ax.plot(recombRate_schunkNagy/1E6, altRange / xNorm, linewidth=Plot_LineWidth, label=r"S&N 2009")
            ax.set_title('Ionospheric Recombination Rate', fontsize=Title_FontSize)
            ax.set_xlabel('Recombination Rate [cm$^{3}$s$^{-1}$ ]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xlim(1E-14, 1E-6)
            ax.set_xscale('log')
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000 + 50, 100))
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\ionizationRecomb\MODEL_recombinationRate.png', dpi=dpi)

        return data_dict

    def electronDensity_fromIonizationRecomb(altRange, data_dict, **kwargs):

        q = np.sum(data_dict['qtot'][0],axis=0) # sum up the recombination rates from all the incoming electrons
        alpha = data_dict['recombRate'][0]
        n_e = np.sqrt(q/alpha) # in m^-3
        n_e_cm3 = n_e/(np.power(stl.cm_to_m, 3))

        if kwargs.get('showPlot', False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(sharex=True)
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(n_e_cm3, altRange / xNorm, linewidth=Plot_LineWidth, label=r"$n_{e} = \sqrt{\frac{q(z)}{\alpha}}$")
            ax.set_title('Model Ionospheric Electron Density', fontsize=Title_FontSize)
            ax.set_xlabel('Electron Density [cm$^{3}$]', fontsize=Label_FontSize)
            ax.set_ylabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.axhline(y=400000 / xNorm, label='Observation Height', color='red')
            ax.set_xlim(1E2, 1E7)
            ax.set_xscale('log')
            ax.grid(True)
            ax.yaxis.set_ticks(np.arange(0, 1000 + 50, 100))
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor, length=Tick_Length_minor)

            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(f'{GenToggles.simFolderPath}\ionizationRecomb\MODEL_electronDensityFromIonizationRecomb.png', dpi=dpi)

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

    outputPath = rf'{GenToggles.simFolderPath}\ionizationRecomb\ionizationRecomb.cdf'
    outputCDFdata(outputPath, data_dict)
