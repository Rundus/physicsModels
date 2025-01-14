# --- model_primaryBeam_classes --
import spaceToolsLib as stl
import numpy as np
from scipy.integrate import simpson

class helperFitFuncs:
    def distFunc_to_diffNFlux(self, Vperp, Vpara, dist, mass, charge):
        # Input: Velocities [m/s], distribution function [s^3m^-6]
        # output: diffNFlux [cm^-2 s^-1 eV^-1 str^-1]
        Emag = 0.5 * mass * (Vperp ** 2 + Vpara ** 2) / charge
        return (2 * Emag) * np.power(charge / (100 * mass), 2) * dist
    def diffNFlux_to_distFunc(self, Vperp, Vpara, diffNFlux, mass, charge):
        # Input: Vperp,Vpara in [m/s]. DiffNFlux in [cm^-2 s^-1 str^-1 eV^-1]
        # output: distribution function [s^3m^-6]
        Energy = 0.5 * mass * (Vperp ** 2 + Vpara ** 2) / charge
        return 0.5 * np.power((100 * mass / charge), 2) * diffNFlux / Energy
    def generateNoiseLevel(self, energyData, primaryBeamToggles):
        count_interval = 0.8992E-3
        geo_factor = 8.63E-5
        deadtime = 324E-9

        # --- DEFINE THE NOISE LEVEL ---
        diffNFlux_NoiseCount = np.zeros(shape=(len(energyData)))

        for idx,engy in enumerate(energyData):
            deltaT = (count_interval) - (primaryBeamToggles.countNoiseLevel * deadtime)
            diffNFlux_NoiseCount[idx] = (primaryBeamToggles.countNoiseLevel) / (geo_factor * deltaT * engy)

        return diffNFlux_NoiseCount
    def groupAverageData(self, data_dict_diffFlux, pitchIdxs, GenToggles, primaryBeamToggles):
        '''
        # Input:
        # data_dict_diffFlux - data_dict with "Differential_Number_Flux", "Pitch_Angle" and "Energy"  variables
        # pitchIdxs - array of indicies corresponding to specifc pitch values in the "Pitch_Angle" variable
        # Output: Epoch, Differential_Number_Flux and stdDevs averaged over N points specified in the primaryBeamToggles
        '''

        ##############################
        # --- COLLECT THE FIT DATA ---
        ##############################
        # ensure the data is divided into chunks that can be sub-divided. If not, keep drop points from the end until it can be
        low_idx, high_idx = np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][0]).argmin(), np.abs(data_dict_diffFlux['Epoch'][0] - GenToggles.invertedV_times[GenToggles.wRegion][1]).argmin()

        if (high_idx - low_idx) % primaryBeamToggles.numToAverageOver != 0:
            high_idx -= (high_idx - low_idx) % primaryBeamToggles.numToAverageOver


        # Handle the Epoch
        chunkedEpoch = np.split(data_dict_diffFlux['Epoch'][0][low_idx:high_idx], round(len(data_dict_diffFlux['Epoch'][0][low_idx:high_idx]) / primaryBeamToggles.numToAverageOver))
        EpochFitData = np.array([chunkedEpoch[i][int((primaryBeamToggles.numToAverageOver - 1) / 2)] for i in range(len(chunkedEpoch))])


        # --- handle the multi-dimenional data ---

        # create the storage variable
        diffNFlux_avg = np.zeros(shape=(len(EpochFitData), len(pitchIdxs), len(data_dict_diffFlux['Energy'][0])))
        stdDevs_avg = np.zeros(shape=(len(EpochFitData), len(pitchIdxs), len(data_dict_diffFlux['Energy'][0])))

        for loopIdx, pitchIdx in enumerate(pitchIdxs):

            chunkedyData = np.split(data_dict_diffFlux['Differential_Number_Flux'][0][low_idx:high_idx, pitchIdx, :], round(len(data_dict_diffFlux['Differential_Number_Flux'][0][low_idx:high_idx, pitchIdx,:]) / primaryBeamToggles.numToAverageOver))
            chunkedStdDevs = np.split(data_dict_diffFlux['Differential_Number_Flux_stdDev'][0][low_idx:high_idx, pitchIdx, :], round(len(data_dict_diffFlux['Differential_Number_Flux_stdDev'][0][low_idx:high_idx, pitchIdx,:]) / primaryBeamToggles.numToAverageOver))

            # --- Average the chunked data ---
            fitData = np.zeros(shape=(len(chunkedyData), len(data_dict_diffFlux['Energy'][0])))
            fitData_stdDev = np.zeros(shape=(len(chunkedStdDevs), len(data_dict_diffFlux['Energy'][0])))

            for i in range(len(chunkedEpoch)):
                # average the diffFlux data by only choosing data which is valid
                chunkedyData[i][chunkedyData[i] < 0] = np.NaN
                fitData[i] = np.nanmean(chunkedyData[i], axis=0)

                # average the diffFlux data by only choosing data which is valid
                chunkedStdDevs[i][chunkedStdDevs[i] < 0] = np.NaN
                fitData_stdDev[i] = np.nanmean(chunkedStdDevs[i], axis=0)

            diffNFlux_avg[:, loopIdx, :] = fitData
            stdDevs_avg[:, loopIdx, :] = fitData_stdDev

        return EpochFitData, diffNFlux_avg, stdDevs_avg
    def calcTotal_NFlux(self, diffNFlux, pitchValues, energyValues):

        # Inputs:
        # diffNFlux - multidimensional array with shape= (len(pitchRange), len(EnergyRange)) that contains diffNFlux values
        # pitchValues - 1D array with the pitch angles
        # energyValues - 1D array with the energy values in eV

        # output:
        # Phi

        # --- integrate over energies first ---
        diffNflux_PerPitch = np.array([simpson(y=diffNFlux[ptchIdx], x=energyValues) for ptchIdx in range(len(pitchValues))])

        # --- integrate over pitch angle ---
        diffNflux_PerPitch = np.nan_to_num(diffNflux_PerPitch.T)
        omniNFlux = 2 * np.pi * simpson(y=np.sin(np.radians(pitchValues)) * diffNflux_PerPitch, x=pitchValues)
        return omniNFlux
    def calcOmni_diffNFlux(self, diffNFlux, pitchValues, energyValues):
        # Inputs:
        # diffNFlux - multidimensional array with shape= (len(pitchRange), len(EnergyRange)) that contains diffNFlux values
        # pitchValues - 1D array with the pitch angles (in deg)
        # energyValues - 1D array with the energy values in eV

        # output:
        # Phi(E)

        # --- integrate over energies first ---
        diffNFlux = np.nan_to_num(diffNFlux.T)
        diffNflux_Integrand = np.array([np.sin(np.radians(pitchValues))*diffNFlux[idx] for idx in range(len(energyValues))])

        # --- integrate over pitch angle ---
        omniDiffNFlux = 2 * np.pi * np.array([simpson(y=diffNflux_Integrand[idx], x=pitchValues) for idx in range(len(energyValues)) ])

        return omniDiffNFlux
    def calcPara_diffNFlux(self, diffNFlux, pitchValues):
        # Inputs:
        # diffNFlux - multidimensional array with shape= (len(pitchRange), len(EnergyRange)) that contains diffNFlux values
        # pitchValues - 1D array with the pitch angles
        # energyValues - 1D array with the energy values in eV

        # output:
        # Phi(E)

        # --- integrate over energies first ---
        diffNFlux = np.nan_to_num(diffNFlux.T)
        diffNflux_Integrand = np.array([np.cos(np.radians(pitchValues)) * np.sin(np.radians(pitchValues)) * diffNFlux[idx] for idx in range(pitchValues)])

        # --- integrate over pitch angle ---
        paraDiffNFlux = 2 * np.pi * np.array([simpson(y=diffNflux_Integrand[idx].T, x=pitchValues) for idx in range(len(pitchValues))])
        return paraDiffNFlux

    def removeDuplicates(self, a, b):
        from collections import defaultdict
        D = defaultdict(list)
        for i, item in enumerate(a):
            D[item].append(i)
        D = {k: v for k, v in D.items() if len(v) > 1}
        badIndicies = [D[key][1:] for key in D.keys()]
        badIndicies = [item for sublist in badIndicies for item in sublist]
        newA = np.delete(a, badIndicies, axis=0)
        newB = np.delete(b, badIndicies, axis=0)
        return newA, newB

class velocitySpace:

    # --- Generate Distributions from Velocity Space ---
    def generate_Maxwellian(self, mass, charge, n, T, Vperp, Vpara):
        # Input: density [cm^-3], Temperature [eV], Velocities [m/s]
        # output: the distribution function in SI units [s^3 m^-6]
        Emag = (0.5 * mass * (Vperp ** 2 + Vpara ** 2)) / charge
        return (1E6 * n) * np.power(mass / (2 * np.pi * charge * T), 3 / 2) * np.exp(-1 * Emag / T)

    def generate_kappa(self, mass, charge, n, T, kappa, Vperp, Vpara):
        # Input: density [cm^-3], Temperature [eV], Velocities [m/s]
        # output: the distribution function in SI units [s^3 m^-6]
        Emag = (0.5 * mass * (Vperp ** 2 + Vpara ** 2)) / charge
        Ek = T*(1 - 3/(2*kappa))
        return (1E6)*n * np.power(mass/(2*np.pi*kappa*stl.q0*Ek),3/2) * (np.special.gamma(kappa+1)/np.special.gamma(kappa-0.5)) * np.power(1 + Emag/(kappa*Ek),-(kappa +1))

    # def calc_BackScatter_onto_Velspace(self, mass, charge, VperpGrid, VparaGrid, BackScatterSpline, EngyLimit):
    #     Energy = 0.5 * mass * (VperpGrid ** 2 + VparaGrid ** 2) / charge
    #     diffNFluxInterp = BackScatterSpline(Energy)
    #     diffNFluxInterp[np.where(EngyLimit[0] >= Energy)] = 0
    #     diffNFluxInterp[np.where(Energy > EngyLimit[1])] = 0
    #     return diffNFluxInterp
    #
    # def calc_velSpace_DistFuncDiffNFluxGrid(self, Vperp_gridVals, Vpara_gridVals, model_Params, **kwargs):
    #
    #     # --- Define a grid a velocities (static) ---
    #     VperpGrid, VparaGrid = np.meshgrid(Vperp_gridVals, Vpara_gridVals)
    #     distGrid = dist_Maxwellian(VperpGrid, VparaGrid, model_Params)
    #
    #     # --- modify the initial beam ---
    #     initalBeamParams = kwargs.get('initalBeamParams', [])
    #
    #     if initalBeamParams != []:
    #         initialBeamAngle, initialBeamEnergyThresh = initalBeamParams[0], initalBeamParams[1]
    #         for i in range(len(VperpGrid)):
    #             for j in range(len(VperpGrid[0])):
    #
    #                 pitchVal = np.degrees(np.arctan2(VperpGrid[i][j], VparaGrid[i][j]))
    #                 EnergyVal = 0.5 * m_e * (VperpGrid[i][j] ** 2 + VparaGrid[i][j] ** 2) / q0
    #
    #                 if np.abs(pitchVal) >= initialBeamAngle:
    #                     distGrid[i][j] = 0
    #                 if EnergyVal >= initialBeamEnergyThresh:
    #                     distGrid[i][j] = 0
    #
    #     # Accelerate the Beam based off of model_Params[-1]
    #     Vperp_gridVals_Accel = Vperp_gridVals
    #     Vpar_gridVals_Accel = np.array([np.sqrt(val ** 2 + 2 * model_Params[-1] * q0 / m_e) for val in Vpara_gridVals])
    #     VperpGrid_Accel, VparaGrid_Accel = np.meshgrid(Vperp_gridVals_Accel, Vpar_gridVals_Accel)
    #     diffNFluxGrid_Accel = calc_diffNFlux(VperpGrid_Accel, VparaGrid_Accel, distGrid)
    #
    #     return VperpGrid_Accel, VparaGrid_Accel, distGrid, diffNFluxGrid_Accel
    #
    # def mapping_VelSpace_magMirror(self, VperpGrid, VparaGrid, distFuncGrid, targetAlt, startingAlt, mapToMagSph):
    #     # INPUT:
    #     # velocity Space Grids and distribution function to map them either:
    #     # (a) FROM the ionosphere to Magnetosphere
    #     # (b) FROM the magnetosphere to Ionosphere
    #
    #     # OUTPUT:
    #     # VperpGrid_newBeta, VparaGrid_newBeta, diffNFlux_newBeta
    #
    #     # --- Determine the beta value ---
    #     betaVal = ((6378 + startingAlt) / (6378 + targetAlt)) ** 3
    #
    #     # --- Determine the velocity values from the Grids ---
    #     Vperp_gridVals = VperpGrid.flatten()
    #     Vpara_gridVals = VparaGrid.flatten()
    #
    #     if mapToMagSph:
    #         # from Ionosphere to Magnetosphere
    #         Vperp_gridVals_mapped = np.array([val / np.sqrt(betaVal) for val in Vperp_gridVals])
    #         Vpara_gridVals_mapped_sqrd = np.array(
    #             [Vpara_iono ** 2 + (1 - 1 / betaVal) * (Vperp_iono ** 2) for Vperp_iono, Vpara_iono in
    #              zip(Vperp_gridVals, Vpara_gridVals)])
    #         Vpara_gridVals_mapped = np.array(
    #             [np.sqrt(val) if val >= 0 else -1 * np.sqrt(np.abs(val)) for val in Vpara_gridVals_mapped_sqrd])
    #         VperpGrid_mapped, VparaGrid_mapped = Vperp_gridVals_mapped.reshape(len(VperpGrid), len(
    #             VperpGrid[0])), Vpara_gridVals_mapped.reshape(len(VparaGrid), len(VparaGrid[0]))
    #         diffNFlux_mapped = calc_diffNFlux(VperpGrid_mapped, VparaGrid_mapped, distFuncGrid)
    #
    #     else:
    #         # from Magnetosphere to Ionosphere
    #         Vperp_gridVals_mapped = np.array([np.sqrt(betaVal) * val for val in Vperp_gridVals])
    #         Vpara_gridVals_mapped_sqrd = np.array(
    #             [Vpar_magsph ** 2 + (1 - betaVal) * (Vper_magsph ** 2) for Vper_magsph, Vpar_magsph in
    #              zip(Vperp_gridVals, Vpara_gridVals)])
    #         Vpara_gridVals_mapped = np.array(
    #             [np.sqrt(val) if val >= 0 else -1 * np.sqrt(np.abs(val)) for val in Vpara_gridVals_mapped_sqrd])
    #         VperpGrid_mapped, VparaGrid_mapped = Vperp_gridVals_mapped.reshape(len(VperpGrid), len(
    #             VperpGrid[0])), Vpara_gridVals_mapped.reshape(len(VparaGrid), len(VparaGrid[0]))
    #         diffNFlux_mapped = calc_diffNFlux(VperpGrid_mapped, VparaGrid_mapped, distFuncGrid)
    #
    #     return VperpGrid_mapped, VparaGrid_mapped, diffNFlux_mapped
    #
    # def velocitySpace_to_PitchEnergySpace(self, EnergyBins, PitchBins, VperpGrid, VparaGrid, ZGrid, method):
    #
    #     # description:
    #     # INPUT: Two velocity space grids  + Z-value grid
    #     # OUTPUT: Energy and Pitch grids + new Z-value grid
    #
    #     # determine the type of input data
    #     VperpValues = VperpGrid.flatten()
    #     VparaValues = VparaGrid.flatten()
    #     ZgridValues = ZGrid.flatten()
    #
    #     ZGrid_New = [[[] for engy in range(len(EnergyBins))] for ptch in range(len(PitchBins))]
    #     calcEnergies = [0.5 * m_e * (perp ** 2 + par ** 2) / q0 for perp, par in zip(VperpValues, VparaValues)]
    #     calcPitch = [(180 / pi) * arctan2(perp, par) for perp, par in zip(VperpValues, VparaValues)]
    #
    #     # assign the values to ZGrid_new
    #     for i in range(len(ZgridValues)):
    #         engyIdx = abs(EnergyBins - calcEnergies[i]).argmin()
    #         ptchIdx = abs(PitchBins - calcPitch[i]).argmin()
    #         if method == 'convolve':
    #             energyResolution = 0.18
    #             mean = EnergyBins[engyIdx]
    #             sigma = energyResolution * mean / (2 * np.sqrt(2 * np.log(2)))
    #             ZGrid_New[ptchIdx][engyIdx].append(
    #                 ZgridValues[i] * normalized_normalDistribution(calcEnergies[i], mean=mean, sigma=sigma))
    #         else:
    #             ZGrid_New[ptchIdx][engyIdx].append(ZgridValues[i])
    #
    #     # flatten the values in the diffnFlux new array
    #     for ptch in range(len(PitchBins)):
    #         for engy in range(len(EnergyBins)):
    #             if method == 'average' or method == 'convolve':
    #                 try:
    #                     ZGrid_New[ptch][engy] = sum(ZGrid_New[ptch][engy]) / len(ZGrid_New[ptch][engy])
    #                 except:
    #                     ZGrid_New[ptch][engy] = sum(ZGrid_New[ptch][engy])
    #             elif method == 'sum':
    #                 ZGrid_New[ptch][engy] = sum(ZGrid_New[ptch][engy])
    #
    #     EnergyGrid, PitchGrid = np.meshgrid(EnergyBins, PitchBins)
    #     return np.array(ZGrid_New), EnergyGrid, PitchGrid

class fittingDistributions:

    # --- FUNCTION for fitting ---
    def diffNFlux_fitFunc_Maxwellian(self, x, n, T, V, mass, charge):  # Used in primaryBeam_fitting
        Energy = (2 * x/ mass) - 2 * V / mass + (2 * x / mass)
        return (2 * x) * ((charge / mass) ** 2) * (1E2 * n) * np.power(mass / (2 * np.pi * charge * T), 3 / 2) * np.exp((-mass * Energy / (2 * T)))

    def diffNFlux_fitFunc_Kappa(self, x, n, T, V,kappa, mass, charge):  # Used in primaryBeam_fitting
        Energy = (2 * x/ mass) - 2 * V / mass + (2 * x / mass)
        return (2 * x) * ((charge / mass) ** 2) * (1E2 * n) * np.power(mass / (2 * np.pi * charge * T), 3 / 2) * np.exp((-mass * Energy / (2 * T)))


