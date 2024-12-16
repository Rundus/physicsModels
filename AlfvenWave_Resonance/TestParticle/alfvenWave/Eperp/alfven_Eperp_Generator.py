# --- alfven_Eperp_Generator.py ---
# --- Author: C. Feltman ---
# DESCRIPTION:
# Generates the entire E-Field for all time in the simulation and returns a variable that looks like:
# [
#   [Ez(x=0,t=0),Ez(x=1,t=0),Ez(x=2,t=0)...., Ez(x=len(Alt),t=0)],
#   [Ez(x=0,t=1),Ez(x=1,t=1),Ez(x=2,t=1)...., Ez(x=len(Alt),t=1)]
#   ,...]

# --- imports ---
from ACESII_code.Science.Simulations.TestParticle.simToggles import m_to_km, R_REF, GenToggles,EToggles,runFullSimulation
from ACESII_code.Science.Simulations.TestParticle.plasmaEnvironment.plasmaEnvironment_Generator import generatePlasmaEnvironment
from ACESII_code.Science.Simulations.TestParticle.geomagneticField.geomagneticField_Generator import generateGeomagneticField
import time
import numpy as np
from itertools import product
from copy import deepcopy
from tqdm import tqdm
from ACESII_code.class_var_func import prgMsg,Done
from myspaceToolsLib.CDF_load import outputCDFdata,loadDictFromFile
start_time = time.time()

#TODO: CHANGE SPEED = INERTIAL ALF SPEED

##################
# --- PLOTTING ---
##################
plot_Eperp = False

################
# --- OUTPUT ---
################
outputData = True if not runFullSimulation else True


# --- Re-run the plasma environment and load the data ---
regenerateEnvironment = False
if regenerateEnvironment:
    prgMsg('Regenerating Plasma Environment')
    generateGeomagneticField(outputData=True)
    generatePlasmaEnvironment(outputData=True)
    Done(start_time)

data_dict_Bgeo = loadDictFromFile(rf'{GenToggles.simOutputPath}\geomagneticField\geomagneticField.cdf')
data_dict_plasEvrn = loadDictFromFile(f'{GenToggles.simOutputPath}\plasmaEnvironment\plasmaEnvironment.cdf')


def alfvenEperpGenerator(outputData, **kwargs):

    def Phiperp_generator(x, z, t, Vel, kperp ):
        # the middle of wave
        if EToggles.Z0_wave - Vel * (t - EToggles.tau0) > z > EToggles.Z0_wave - Vel * t:
            altTimeTerm = (1 - np.cos(((z - EToggles.Z0_wave) + Vel * t) * (2 * np.pi / (Vel * EToggles.tau0))))
            Phi_perp_Val = EToggles.waveFraction*(EToggles.Eperp0/EToggles.kperp0) *altTimeTerm * np.cos(2*np.pi*x/EToggles.waveFraction)
        else:
            Phi_perp_Val = 0

        return Phi_perp_Val

    # --- Eperp Maker ---
    def Eperp_generator(x, z, t, Vel, Bgeo_init,Bgeo):
        # the middle of wave
        if EToggles.Z0_wave - Vel * (t - EToggles.tau0) > z > EToggles.Z0_wave - Vel * t:
            amplitude = np.sqrt(Bgeo/Bgeo_init)*EToggles.Eperp0 * np.sin(2*np.pi * x / EToggles.waveFraction)
            EperpVal = amplitude * (1 - np.cos(((z - EToggles.Z0_wave) + Vel * t) * (2 * np.pi / (Vel * EToggles.tau0))))
        else:
            EperpVal = 0

        return EperpVal

    def EperpProfile(altRange, timeRange, **kwargs):
        plotBool = kwargs.get('showPlot', False)

        # get the profiles and flip them so we begin with high altitudes
        altRange = altRange
        lambdaPerp, kperp = data_dict_plasEvrn['lambdaPerp'][0],data_dict_plasEvrn['kperp'][0]
        alfSpdMHD = data_dict_plasEvrn['alfSpdMHD'][0]
        alfSpdInertial = data_dict_plasEvrn['alfSpdInertial'][0]
        Bgeo, Bgrad = data_dict_Bgeo['Bgeo'][0], data_dict_Bgeo['Bgrad'][0]
        initindex = np.abs(altRange - EToggles.Z0_wave).argmin()  # the index of the startpoint of the Wave
        initBgeo = Bgeo[initindex]  # <--- This determines where the scaling begins
        # speed = [R_REF for i in range(len(altRange))]
        speed = alfSpdMHD

        # create the X dimension
        if EToggles.lambdaPerp_Rez%2 == 0:
            raise Exception('lambdaPerp_Rez must be even')

        simXRange = np.linspace(0, EToggles.lambdaPerp0, EToggles.lambdaPerp_Rez)
        lambdaPerpRange = np.linspace(0, 1, EToggles.lambdaPerp_Rez)# 0 to 2 because I've cut the pulse in half

        #####################
        # --- Phi & Eperp ---
        #####################
        PhiPerp = np.zeros(shape=(len(timeRange), len(altRange), len(simXRange)))
        Eperp = np.zeros(shape=(len(timeRange), len(altRange), len(simXRange)))

        print('\nNum of iterations:')
        print(f'{len(timeRange) * len(altRange)* len(simXRange)}\n')

        # create a meshgrid
        for t, z, x in tqdm(product(*[range(len(timeRange)),range(len(altRange)),range(len(simXRange))])):
            timeVal = timeRange[t]
            zVal = altRange[z]
            xVal = lambdaPerpRange[x]
            BgeoVal = Bgeo[np.abs(altRange - zVal).argmin()]  # the index of the startpoint of the Wave

            # PhiPerp
            PhiPerp[t][z][x] = Phiperp_generator(x=xVal,
                                                 z=zVal,
                                                 t=timeVal,
                                                 Vel=speed[z],
                                                 kperp=kperp[z])

            # Eperp
            Eperp[t][z][x] = Eperp_generator(x=xVal,
                                              z=zVal,
                                              t=timeVal,
                                              Vel=speed[z],
                                              Bgeo_init=initBgeo,
                                              Bgeo=BgeoVal)


        # plotting
        if plotBool:
            from ACESII_code.class_var_func import prgMsg,Done
            import time
            start_time = time.time()
            prgMsg('Creating Eperp Plots')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 22})
            for timeIndexChoice in range(len(timeRange)):
                fig,ax = plt.subplots()
                fig.set_figwidth(15)
                fig.set_figheight(15)
                ax.set_title('$E_{\perp}$ Propogation vs Altitude \n' + f't = {timeRange[timeIndexChoice]}'+r'  $\tau_{0}$=' +f'{EToggles.tau0} s' +'\n' + '$\lambda_{\perp}$ =' + f'{EToggles.lambdaPerp0} [m],  ' + '$\omega_{wave}$ =' + f'{EToggles.waveFreq_Hz} [Hz]')
                cmap = ax.pcolormesh(simXRange/EToggles.lambdaPerp0,altRange/R_REF,Eperp[timeIndexChoice]*m_to_km, cmap='turbo', vmin=0*-1*EToggles.Eperp0*m_to_km,vmax=EToggles.Eperp0*m_to_km)
                ax.set_xlabel('X Distance [$\lambda_{\perp}$]')
                ax.set_ylabel('Z Distance [$R_{E}$]')
                ax.grid(True)
                cbar = plt.colorbar(cmap)
                cbar.set_label('$|E_{\perp}$| [mV/m]')
                plt.savefig(rf'{GenToggles.simOutputPath}\Eperp\Eperp_t{timeIndexChoice}.png')
                plt.close()
            Done(start_time)

        # create the output variable
        return Eperp, PhiPerp, lambdaPerpRange

    ################
    # --- OUTPUT ---
    ################
    if outputData:
        prgMsg('Writing out Potential/Eperp Data')

        # get all the variables
        Eperp, PhiPerp, lambdaPerpRange = EperpProfile(altRange=GenToggles.simAlt, timeRange=GenToggles.simTime, showPlot=plot_Eperp)

        # --- Construct the Data Dict ---
        exampleVar = {'DEPEND_0': None, 'DEPEND_1': None, 'DEPEND_2': None, 'FILLVAL': -9223372036854775808,
                      'FORMAT': 'I5', 'UNITS': 'm', 'VALIDMIN': None, 'VALIDMAX': None, 'VAR_TYPE': 'data',
                      'SCALETYP': 'linear', 'LABLAXIS': 'simAlt'}


        data_dict = {'Eperp': [Eperp, {'DEPEND_0': 'simTime', 'DEPEND_1': 'simAlt', 'DEPEND_2': 'simXRange', 'UNITS': 'V/m', 'LABLAXIS': 'Eperp'}],
                     'PhiPerp': [PhiPerp, {'DEPEND_0': 'simTime', 'DEPEND_1': 'simAlt', 'DEPEND_2': 'simXRange', 'UNITS': 'V', 'LABLAXIS': 'PhiPerp'}],
                     'simXRange': [lambdaPerpRange, {'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': '&lambda;!B&perp;!N'}],
                     'simTime': [GenToggles.simTime, {'DEPEND_0': 'simAlt', 'UNITS': 'seconds', 'LABLAXIS': 'simTime'}],
                     'simAlt': [GenToggles.simAlt, {'DEPEND_0': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'simAlt'}]
                     }

        # update the data dict attrs
        for key, val in data_dict.items():
            newAttrs = deepcopy(exampleVar)

            for subKey, subVal in data_dict[key][1].items():
                newAttrs[subKey] = subVal

            data_dict[key][1] = newAttrs

        # --- output the data ---
        outputPath = rf'{GenToggles.simOutputPath}\Eperp\Eperp.cdf'
        outputCDFdata(outputPath, data_dict)
        Done(start_time)



#################
# --- EXECUTE ---
#################
if outputData:
    alfvenEperpGenerator(outputData=outputData,showPlot=plot_Eperp)