# --- imports ---

from spaceToolsLib.variables import m_to_km,Re
from spaceToolsLib.tools import CHAOS
from spaceToolsLib.tools.CDF_output import outputCDFdata
from copy import deepcopy
from numpy import array,degrees,arccos
from numpy.linalg import norm
from datetime import datetime
from spacepy import coordinates as coord
from spacepy.time import Ticktock
from src.physicsModels.ionosphere.simToggles_Ionosphere import BgeoToggles,GenToggles



########################################
# --- GENERATE THE B-FIELD & TOGGLES ---
########################################

def generateGeomagneticField(**kwargs):
    plotting = kwargs.get('showPlot', False)

    def geomagneticFieldProfile(altRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        if BgeoToggles.useConstantBval:
            # MAKE IT A CONSTANT
            Bgeo = array([BgeoToggles.ConstantBval for alt in altRange])
            Bgrad = array([0 for alt in altRange])
        else:

            geomagAlts = [((alt + (Re*m_to_km)) / (Re*m_to_km)) for alt in altRange]
            geomagLats = array([degrees(arccos(radi / BgeoToggles.Lshell)) for radi in geomagAlts])
            geomagLongs = array([111.83 for i in range(len(altRange))])
            times = [datetime(2022, 11, 20, 17, 20, 00, 000) for i in range(len(altRange))]
            Pos = array([geomagAlts, geomagLats, geomagLongs]).transpose()
            ISOtime = [times[i].isoformat() for i in range(len(times))]
            cvals_MAG = coord.Coords(Pos, 'MAG', 'sph')
            cvals_MAG.ticks = Ticktock(ISOtime, 'ISO')
            cvals_GDZ = cvals_MAG.convert('GEO', 'sph')
            Lat_geo = cvals_GDZ.lati

            # Get the Chaos model
            B = CHAOS(Lat_geo, [15.25 for i in range(len(altRange))], array(altRange) / m_to_km, times)
            Bgeo = (1E-9) * array([norm(Bvec) for Bvec in B])
            Bgrad = [(Bgeo[i + 1] - Bgeo[i]) / (altRange[i + 1] - altRange[i]) for i in range(len(Bgeo) - 1)]
            Bgrad = array(Bgrad + [Bgrad[-1]]) # add the high altitude value Bgrad again to model the starting point (MAYBE it SHOULD BE 0?)

        if plotBool:
            import matplotlib.pyplot as plt
            figure_width = 20  # in inches
            figure_height = 8  # in inches
            Title_FontSize = 35
            Label_FontSize = 30
            Tick_FontSize = 25
            Tick_FontSize_minor = 20
            Tick_Length = 10
            Tick_Width = 2
            Tick_Length_minor = 5
            Tick_Width_minor = 1
            Plot_LineWidth = 4
            Legend_fontSize = 30
            dpi = 100

            fig, ax = plt.subplots()
            fig.set_size_inches(figure_width, figure_height)
            ax.plot(altRange/(Re*m_to_km), Bgeo/(1E-9),linewidth=Plot_LineWidth, color='black',label=r'$\vec{B}_{geo}$')
            ax.set_title(r'|$\vec{B}_{geo}$|,$\nabla B$ vs Altitude', fontsize=Title_FontSize)
            ax.set_ylabel('$B_{geo}$ [nT]', fontsize=Label_FontSize)
            ax.set_xlabel('Altitude [$R_{E}$]', fontsize=Label_FontSize)
            ax.set_yscale('log')
            ax.axvline(x=400000/(Re*m_to_km),label='Observation Height',color='red',linewidth=Plot_LineWidth)
            ax.plot(altRange / (Re*m_to_km), Bgrad / (1E-9), color='black', linewidth=Plot_LineWidth, linestyle='--', label=r'$\nabla B$', alpha=1)
            ax.set_ylim(1E3,1E5)
            ax.legend(fontsize=Legend_fontSize)

            axBGrad = ax.twinx()
            axBGrad.plot(altRange / (Re*m_to_km), Bgrad/(1E-9), color='black',linewidth=Plot_LineWidth, linestyle='--',label=r'$\nabla B$')
            axBGrad.set_ylabel(r'$\nabla B$ [nT/m]', fontsize=Label_FontSize)
            axBGrad.axvline(x=400000 / (Re*m_to_km), label='Observation Height', color='red',linewidth=Plot_LineWidth)

            axesE = [ax,axBGrad]
            for axes in axesE:
                axes.tick_params(axis='y', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                axes.tick_params(axis='y', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)
                axes.tick_params(axis='x', which='major', labelsize=Tick_FontSize, width=Tick_Width,
                                           length=Tick_Length)
                axes.tick_params(axis='x', which='minor', labelsize=Tick_FontSize_minor,
                                           width=Tick_Width_minor, length=Tick_Length_minor)

            plt.subplots_adjust(left=0.08, bottom=0.13, right=0.89, top=0.9, wspace=None, hspace=None)
            plt.savefig(f'{BgeoToggles.outputFolder}\MODEL_Bgeo.png',dpi=dpi)
            # plt.show()

        return Bgeo, Bgrad

    if plotting:
        # get all the variables and plot them if required
        geomagneticFieldProfile(altRange=GenToggles.simAlt, showPlot=plotting)


    # get all the variables and plot them if required
    Bgeo, Bgrad = geomagneticFieldProfile(altRange=GenToggles.simAlt)

    # --- Construct the Data Dict ---
    exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                  'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                  'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}

    data_dict = {'Bgeo': [Bgeo, {'DEPEND_0':'simAlt', 'UNITS':'T', 'LABLAXIS': 'Bgeo'}],
                  'Bgrad': [Bgrad, {'DEPEND_0':'simAlt', 'UNITS':'T', 'LABLAXIS': 'Bgrad'}],
                  'simAlt': [GenToggles.simAlt, {'DEPEND_0':'simAlt', 'UNITS':'m', 'LABLAXIS': 'simAlt'}]}

    # update the data dict attrs
    for key, val in data_dict.items():
        newAttrs = deepcopy(exampleVar)

        for subKey, subVal in data_dict[key][1].items():
            newAttrs[subKey] = subVal

        data_dict[key][1] = newAttrs

    outputPath = rf'{BgeoToggles.outputFolder}\geomagneticfield.cdf'
    outputCDFdata(outputPath, data_dict)
