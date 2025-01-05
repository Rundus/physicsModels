# --- model_primaryBeam_classes --
import spaceToolsLib as stl
import numpy as np

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


