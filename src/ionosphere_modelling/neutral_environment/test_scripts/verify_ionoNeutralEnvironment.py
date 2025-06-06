# --- ionoNeutralEnvironment_Generator ---
# get the NRLMSIS data and export the neutral temperature vs altitude at ACES-II times.
# Also, interpolate each variable in order to sample the data at my model cadence


# --- imports ---
from spaceToolsLib.tools.CDF_output import outputCDFdata
from spaceToolsLib.variables import m_to_km,Re,netural_dict
from numpy import datetime64,squeeze
import pymsis
import numpy as np
from src.physicsModels.ionosphere.simToggles_Ionosphere import GenToggles,neutralsToggles


##################
# --- PLOTTING ---
##################
xNorm = m_to_km # use m_to_km otherwise
xLabel = '$R_{E}$' if xNorm == Re else 'km'
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


def generateNeutralEnvironment(**kwargs):
    showPlot = kwargs.get('showPlot', False)

    # get the NRLMSIS data
    lon = GenToggles.target_Longitude
    lat = GenToggles.target_Latitude
    alts = GenToggles.simAlt/m_to_km
    f107 = 150  # the F10.7 (DON'T CHANGE)
    f107a = 150  # ap data (DON't CHANGE)
    ap = 7
    aps = [[ap] * 7]
    dt_targetTime = GenToggles.target_time
    date = datetime64(f"{dt_targetTime.year}-{dt_targetTime.month}-{dt_targetTime.day}T{dt_targetTime.hour}:{dt_targetTime.minute}")

    #  output is of the shape (1, 1, 1, 1000, 11), use squeeze to Get rid of the single dimensions
    NRLMSIS_data = squeeze(pymsis.calculate(date, lon, lat, alts, f107, f107a, aps))

    # create new data_dict with data from NRLMSIS at specific Lat/Long/Time Values
    data_dict = {}
    for var in pymsis.Variable:

        varData = np.array([val if val> 1E-25 else 0 for val in NRLMSIS_data[:, var]])


        if var.name =='MASS_DENSITY':
            varunits = 'kg m!A-3!N'
            varname = 'rho_n'
        elif var.name =='TEMPERATURE':
            varunits = 'K'
            varname = 'Tn'
        else:
            varunits = 'm!A-3!N'
            varname = var.name

        data_dict = {**data_dict, **{varname: [varData, {'DEPEND_0': 'simAlt', 'UNITS':varunits , 'LABLAXIS': var.name,'VAR_TYPE': 'data'} ]}}

        if showPlot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(GenToggles.simAlt / xNorm, NRLMSIS_data[:,var], label=rf'{var.name}', linewidth=Plot_LineWidth)
            ax.set_title(f'Time: {GenToggles.target_time}, Lat: {GenToggles.target_Latitude}, Long: {GenToggles.target_Longitude}\n'+rf'{var.name} vs Altitude', fontsize=Title_FontSize)
            ax.set_ylabel(f'{var.name} [{varunits}]', fontsize=Label_FontSize)
            ax.set_xlabel(f'Altitude [{xLabel}]', fontsize=Label_FontSize)
            ax.grid(True)
            ax.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,
                           length=Tick_Length_minor)
            ax.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width, length=Tick_Length)
            ax.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor, width=Tick_Width_minor,
                           length=Tick_Length_minor)
            plt.legend(fontsize=Legend_fontSize)
            plt.tight_layout()
            plt.savefig(rf'{neutralsToggles.outputFolder}\MODEL_{var.name}.png', dpi=dpi)


    # add the ionosphere_models simulation altitude
    data_dict = {**data_dict, **{'simAlt': [GenToggles.simAlt, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt','VAR_TYPE': 'data'}]}}

    # add the total neutral density
    n_n = np.array([data_dict[f"{key}"][0] for key in neutralsToggles.wNeutrals])
    data_dict = {**data_dict, **{'nn': [np.sum(n_n,axis=0), {'DEPEND_0': 'simAlt', 'UNITS': 'm^-3', 'LABLAXIS': 'simAlt', 'VAR_TYPE': 'data'}]}}

    # add the effective neutral mass
    m_eff_n = np.sum(np.array( [netural_dict[key]*data_dict[f"{key}"][0] for key in neutralsToggles.wNeutrals]), axis=0)/data_dict['nn'][0]
    data_dict = {**data_dict, **{'m_eff_n': [m_eff_n, {'DEPEND_0': 'simAlt', 'UNITS': 'kg', 'LABLAXIS': 'simAlt', 'VAR_TYPE': 'data'}]}}

    #####################
    # --- OUTPUT DATA ---
    #####################

    outputPath = rf'{neutralsToggles.outputFolder}\neutralEnvironment.cdf'
    outputCDFdata(outputPath, data_dict)
