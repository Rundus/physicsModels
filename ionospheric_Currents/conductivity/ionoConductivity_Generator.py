# --- imports ---
from ionosphere.simToggles_iono import GenToggles
from spaceToolsLib.variables import u0,m_e,ep0,cm_to_m,IonMasses,q0, m_to_km,Re
from spaceToolsLib.tools.CDF_load import loadDictFromFile
from spaceToolsLib.tools.CDF_output import outputCDFdata
from copy import deepcopy



##################
# --- PLOTTING ---
##################
xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'


# get the geomagnetic field data dict
data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simFolderPath}\geomagneticField\geomagneticField.cdf')
data_dict_plasEvrn = loadDictFromFile(f'{GenToggles.simOutputPath}\plasmaEnvironment\plasmaEnvironment.cdf')

def generateIonosphericConductivity(outputData,GenToggles, **kwargs):
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

    def ionMassProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        plasmaDensity_total = plasmaDensityProfile(altRange)
        z_i = 2370 * m_to_km  #
        h_i = 1800 * m_to_km  # height of plasma sheet (in meters)
        n_Op = array([plasmaDensity_total[i] * 0.5 * (1 - tanh((altRange[i] - z_i) / h_i)) for i in range(len(altRange))])
        n_Hp = plasmaDensity_total - n_Op
        m_Op = IonMasses[1]
        m_Hp = IonMasses[2]
        m_eff_i = array([m_Hp * 0.5 * (1 + tanh((altRange[i] - z_i) / h_i)) + m_Op * 0.5 * (1 - tanh((altRange[i] - z_i) / h_i)) for i in range(len(altRange))])

        if plotBool:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, sharex=True)
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

        return n_Op, n_Hp, m_eff_i


    def ionNeutralCollisions(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)



    ##################
    # --- PLOTTING ---
    ##################
    if plotting:
        plottingDict = {'Temperature': True,
                        'lambdaPerp': True,
                        'Density': True,
                        'ionMass': True,
                        'Beta': True,
                        'plasmaFreq': True,
                        'ionCyclotron': True,
                        'ionLarmorRadius': True}

        # --- collect all the functions ---
        profileFuncs = [temperatureProfile,
                        plasmaDensityProfile,
                        ionMassProfile,
                        plasmaBetaProfile,
                        plasmaFreqProfile,
                        ionCyclotronProfile,
                        ionLarmorRadiusProfile,
                        MHD_alfvenSpeedProfile]

        counter = 0
        for key, val in plottingDict.items():
            profileFuncs[counter](altRange=GenToggles.simAlt, showPlot=plotting)
            counter+= 1


    #####################
    # --- OUTPUT DATA ---
    #####################
    if outputData:

        # get all the variables
        Temp = temperatureProfile(GenToggles.simAlt)
        plasmaDensity = plasmaDensityProfile(GenToggles.simAlt)
        n_Op, n_Hp, m_eff_i = ionMassProfile(GenToggles.simAlt)
        beta = plasmaBetaProfile(GenToggles.simAlt)
        plasmaFreq = plasmaFreqProfile(GenToggles.simAlt)
        ionCyclotron, ionCyclotron_Op, ionCyclotron_Hp = ionCyclotronProfile(GenToggles.simAlt)
        ionLarmorRadius, ionLarmorRadius_Op, ionLarmorRadius_Hp = ionLarmorRadiusProfile(GenToggles.simAlt)
        alfSpdMHD = MHD_alfvenSpeedProfile(GenToggles.simAlt)

        # --- Construct the Data Dict ---
        exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                      'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                      'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}

        data_dict = {'Temp': [Temp, {'DEPEND_0': 'simAlt', 'UNITS': 'eV', 'LABLAXIS': 'Temperature'}],
                     'plasmaDensity': [plasmaDensity, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'plasmaDensity'}],
                     'n_Op': [n_Op, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'n_Op'}],
                     'n_Hp': [n_Hp, {'DEPEND_0': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'n_Hp'}],
                     'm_eff_i': [m_eff_i, {'DEPEND_0': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i'}],
                     'beta': [beta, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta'}],
                     'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}],
                     'ionCyclotron': [ionCyclotron, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron'}],
                     'ionCyclotron_Op': [ionCyclotron_Op, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron_Op'}],
                     'ionLarmorRadius_Hp': [ionLarmorRadius_Hp, {'DEPEND_0': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionLarmorRadius_Hp'}],
                     'alfSpdMHD': [alfSpdMHD, {'DEPEND_0': 'simAlt', 'UNITS': 'm/s', 'LABLAXIS': 'alfSpdMHD'}],
                     'simAlt': [GenToggles.simAlt, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}]}

        # update the data dict attrs
        for key, val in data_dict.items():
            newAttrs = deepcopy(exampleVar)

            for subKey, subVal in data_dict[key][1].items():
                newAttrs[subKey] = subVal

            data_dict[key][1] = newAttrs

        outputPath = rf'{GenToggles.simFolderPath}\plasmaEnvironment\plasmaEnvironment.cdf'
        outputCDFdata(outputPath, data_dict)
