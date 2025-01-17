# --- alfven_Eperp_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION:
# Generates the entire E-Field for all time in the simulation and returns a variable that looks like:
# [
#   [Ez(x=0,t=0),Ez(x=1,t=0),Ez(x=2,t=0)...., Ez(x=len(Alt),t=0)],
#   [Ez(x=0,t=1),Ez(x=1,t=1),Ez(x=2,t=1)...., Ez(x=len(Alt),t=1)]
#   ,...]

# --- imports ---
from ACESII_code.Science.Simulations.TestParticle.simToggles import R_REF, GenToggles,EToggles,runFullSimulation
from time import  time
import numpy as np
from itertools import product
from copy import deepcopy
from tqdm import tqdm
from ACESII_code.class_var_func import prgMsg, Done
from myspaceToolsLib.CDF_load import outputCDFdata, loadDictFromFile
start_time = time()

##################
# --- PLOTTING ---
##################
plot_Epara = False

################
# --- OUTPUT ---
################
outputData = True if not runFullSimulation else True

# get the Eperp and plasma environment Profiles
data_dict_plasEvrn = loadDictFromFile(f'{GenToggles.simOutputPath}\plasmaEnvironment\plasmaEnvironment.cdf')
data_dict_Eperp = loadDictFromFile(rf'{GenToggles.simOutputPath}\Eperp\Eperp.cdf')


def alfvenEparaGenerator(outputData, **kwargs):


    def EparaProfile(altRange, timeRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        # get the profiles and flip them so we begin with high altitudes
        altRange = altRange
        lambdaPerp, kperp, skinDepth = data_dict_plasEvrn['lambdaPerp'][0],data_dict_plasEvrn['kperp'][0], data_dict_plasEvrn['skinDepth'][0]
        Eperp = data_dict_Eperp['Eperp'][0]

        # create the X dimension
        simXRange = np.linspace(0, EToggles.lambdaPerp0, EToggles.lambdaPerp_Rez)
        lambdaPerpRange = np.linspace(0, 1, EToggles.lambdaPerp_Rez)  # 0 to 2 because I've cut the pulse in half

        # create a meshgrid for determining dE_perp /dz
        PhiPara = np.zeros(shape=(len(timeRange), len(altRange), len(simXRange)))
        Epara = np.zeros(shape=(len(timeRange), len(altRange), len(simXRange)))

        print('\nNum of iterations (Epara):')
        print(f'{len(timeRange) * len(altRange) * len(simXRange)}          {len(timeRange)}   {len(altRange)}   {len(simXRange)}\n')

        for t, z, x in tqdm(product(*[range(len(timeRange)), range(len(altRange)-1), range(len(simXRange))])):
            Eperp_n = Eperp[t][z][x]
            Eperp_n1 = Eperp[t][z + 1][x]
            EperpGradVal = (Eperp_n1 - Eperp_n) / (altRange[z + 1] - altRange[z])
            Epara[t][z][x] = kperp[z] * (skinDepth[z] ** 2) * EperpGradVal / (1 + (kperp[z] * skinDepth[z])**2)

        print('\nNum of iterations (PhiPara):')
        print(f'{len(timeRange) * len(altRange) * len(simXRange)}          {len(timeRange)}   {len(altRange)}   {len(simXRange)}\n')
        for t, z, x in tqdm(product(*[range(len(timeRange)), range(len(altRange)-1), range(len(simXRange))])):
            Epara_n = Epara[t][z][x]
            Epara_n1 = Epara[t][z + 1][x]
            PhiPara[t][z][x] = 0.5*(Epara_n1 - Epara_n) * (altRange[z + 1] - altRange[z])

        # plotting
        if plotBool:
            from ACESII_code.class_var_func import prgMsg,Done
            import time
            start_time = time.time()
            prgMsg('Creating Epara Plots')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 22})
            for timeIndexChoice in range(len(timeRange)):
                fig,ax = plt.subplots()
                fig.set_figwidth(15)
                fig.set_figheight(15)
                ax.set_title('$E_{\parallel}$ Propogation vs Altitude \n' + f't = {timeRange[timeIndexChoice]}'+r'  $\tau_{0}$=' +f'{EToggles.tau0} s' +'\n' + '$\lambda_{\perp}$ =' + f'{EToggles.lambdaPerp0} [m],  ' + '$\omega_{wave}$ =' + f'{EToggles.waveFreq_Hz} [Hz]')
                cmap = ax.pcolormesh(simXRange/EToggles.lambdaPerp0,altRange/R_REF, Epara[timeIndexChoice]*1E6, cmap='bwr', vmin=-1*1E6,vmax=1E6)
                ax.set_xlabel('X Distance [$\lambda_{\perp}$]')
                ax.set_ylabel('Z Distance [$R_{E}$]')
                ax.grid(True)
                cbar = plt.colorbar(cmap)
                cbar.set_label('$|E_{\parallel}$| [uV/m]')
                plt.savefig(rf'{GenToggles.simFolderPath}\alfvenWave\Epara\plots\Epara_t{timeIndexChoice}.png')
                plt.close()
            Done(start_time)

        # create the output variable
        return Epara,PhiPara, lambdaPerpRange



    ################
    # --- OUTPUT ---
    ################
    if outputData:
        prgMsg('Writing out Epara Data')

        # get all the variables
        Epara,PhiPara, simXRange = EparaProfile(altRange=GenToggles.simAlt, timeRange=GenToggles.simTime, showPlot=plot_Epara)

        # --- Construct the Data Dict ---
        exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                      'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                      'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}

        data_dict = {'Epara': [Epara, {'DEPEND_0': 'simTime', 'DEPEND_1': 'simAlt', 'DEPEND_2': 'simXRange', 'UNITS': 'V/m', 'LABLAXIS': 'Epara'}],
                     'PhiPara': [PhiPara, {'DEPEND_0': 'simTime', 'DEPEND_1': 'simAlt', 'DEPEND_2': 'simXRange', 'UNITS': 'V', 'LABLAXIS': 'PhiPara'}],
                     'simXRange': [simXRange, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': '&lambda;!B&perp;!N'}],
                     'simTime': [GenToggles.simTime, {'DEPEND_0': 'simAlt', 'UNITS': 'seconds', 'LABLAXIS': 'simTime'}],
                     'simAlt': [GenToggles.simAlt, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}]}

        # update the data dict attrs
        for key, val in data_dict.items():
            newAttrs = deepcopy(exampleVar)

            for subKey, subVal in data_dict[key][1].items():
                newAttrs[subKey] = subVal

            data_dict[key][1] = newAttrs

            # --- output the data ---
        outputPath = rf'{GenToggles.simOutputPath}\Epara\Epara.cdf'
        outputCDFdata(outputPath, data_dict)
        Done(start_time)



#################
# --- EXECUTE ---
#################
if outputData:
    alfvenEparaGenerator(outputData=outputData,showPlot=plot_Epara)