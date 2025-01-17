# --- model_primaryBeam_classes --
import spaceToolsLib as stl
import numpy as np
from scipy.integrate import simpson
from invertedV_fitting.BackScatter.Evans_Model.parameterizationCurves_Evans1974_classes import *

class backScatter_class:

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
        '''
        # Inputs:
        # diffNFlux - multidimensional array with shape= (len(pitchRange), len(EnergyRange)) that contains diffNFlux values
        # pitchValues - 1D array with the pitch angles (in deg)
        # energyValues - 1D array with the energy values in eV

        # output:
        # Phi(E)
        '''

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

    #############################
    # --- CALCULATE RESPONSES ---
    #############################

    def calcBackscatter(self, energy_Grid, beam_Energies, beam_OmniDiffFlux):
        '''

        :param energy_Grid: 1D grid of energies for the output curves. Arbitrary Length
        :param beam_Energies: 1D array of energy values the for the input Beam.
        :param beam_OmniDiffFlux: 1D array of omniDiffFlux values of the beam [cm^-2s^-1eV^-1]. Length = Len(beam_Energies)
        :return:
        upWard omniDiffFlux (Degraded Primaries) - 1D array of ionospheric degraded primaries flux in units of [cm^-2 s^-1 eV^-1]
        upWard omniDiffFlux (Secondaries) - 1D array of ionospheric secondaries flux in units of [cm^-2 s^-1 eV^-1]
        '''


        model = Evans1974()
        V0 = min(beam_Energies)

        # --- define the outputs ---
        secondariesFlux = np.zeros(shape=(len(energy_Grid)))
        degradedPrimFlux = np.zeros(shape=(len(energy_Grid)))

        # --- loop over beam energies ---
        # print(len(beam_Energies),len(beam_OmniDiffFlux))
        for engyIdx, E_Incident in enumerate(beam_Energies):

            # --- Secondaries ---
            spline = model.generate_SecondariesCurve() # get the secondaries spline
            curve_secondaries = spline(energy_Grid)
            curve_secondaries[np.where(energy_Grid > E_Incident)[0]] = 0
            curve_secondaries[np.where(energy_Grid > 1000)[0]] = 0
            curve_secondaries[np.where(energy_Grid > V0)[0]] = 0
            secondariesFlux += curve_secondaries*beam_OmniDiffFlux[engyIdx]

            # --- DegradedPrimaries ---
            spline = model.generate_BackScatterCurve(E_Incident) # get the degradedPrimaries
            curve_degradedPrimaries = spline(energy_Grid)
            curve_degradedPrimaries[np.where(energy_Grid < E_Incident * 1E-2)[0]] = 0
            curve_degradedPrimaries[np.where(energy_Grid > E_Incident)[0]] = 0
            curve_degradedPrimaries[np.where(energy_Grid > V0)[0]] = 0
            degradedPrimFlux += curve_degradedPrimaries*beam_OmniDiffFlux[engyIdx]

        return degradedPrimFlux, secondariesFlux




    #
    # def calcSecondaries(self, energyRange, InputOmniFlux, Niterations, V0):
    #
    #     '''
    #     # INPUTS:
    #     # detectorEnergies - ALL Energy values the detector energy range.
    #     # InputOmniFlux - scalar value of the omni-directional flux
    #     # V0 - returns only 0's in the output array for energies above this limit i.e. the parallel potential
    #
    #     # OUTPUT
    #     # up-ward differentialFlux (cm^-2s^-1str^-1eV^-1)
    #     '''
    #
    #     # --- get the spline curve ---
    #     secondariesSpline = self.generate_SecondariesCurve()
    #     secondaryFlux = np.zeros(shape=(len(energyRange)))
    #     for i in range(Niterations):
    #         S_n = InputOmniFlux * np.power(np.array(secondariesSpline(energyRange)), i + 1)
    #         S_n[np.where(energyRange > 1000)[0]] = 0  # remove any spline above the Evans1974 limit
    #         S_n[np.where(energyRange > V0)[0]] = 0
    #         secondaryFlux += S_n
    #
    #     return secondaryFlux / (2 * np.pi)  # divide by 2pi to get str^-1
    #
    # def calcDegradedPrimaries(self, IncidentBeamEnergies, Incident_OmniDiffFlux, Niterations, detectorEnergies, V0):
    #     '''
    #     # INPUTS:
    #     # IncidentEnergies - Energy values the detector energy range.
    #     # InputOmniDiffFlux - array of the omni-directional differential flux with len = len(IncidentEnergy)
    #     # V0 (kwarg) - if given, returns only 0's in the output array for energies above this limit
    #
    #     # OUTPUT
    #     # up-ward differentialFlux (cm^-2s^-1str^-1eV^-1) for Secondaries and Backscatter, iterated "Niterations" number of times
    #     '''
    #
    #     # steps:
    #     # 1: Calculate JUST the backscatter from the incoming beam, NOT the secondaries since this is already accounted for
    #     # 2: Use the generated backscatter to generate MORE backscatter as well as secondaries
    #     # 3. repeat (2) for the N+1 step as many times as needed until convergence
    #
    #     #####################
    #     # --- BACKSCATTER ---
    #     #####################
    #
    #     # --- First step (N=1) ---
    #     # calculate the Backscatter for each energy
    #     B_1 = np.zeros(shape=(len(detectorEnergies)))
    #
    #     for idx, E_Incident in enumerate(IncidentBeamEnergies):
    #
    #         # --- get the spline curve ---
    #         backscatterSpline = self.generate_BackScatterCurve(E_Incident)
    #
    #         # --- Calculate the spline at the relevant energies ---
    #         B_iter = Incident_OmniDiffFlux[idx] * backscatterSpline(detectorEnergies)
    #         B_iter[np.where(detectorEnergies >= E_Incident)[
    #             0]] = 0  # limit the fluxes to below the incident energy of the primary beam
    #
    #         # Note:
    #         # the Evans curve does NOT consider backscatter from energies 10E-2 less than E_incident e.g. 1keV only generates 10eV to 1keV, thus
    #         # the spline with fail outside this region. Implement a fix to only evaluate energies within this range:
    #         B_iter[np.where(detectorEnergies < 1E-2 * E_Incident)] = 0
    #
    #         B_1 += B_iter
    #
    #         fig, ax = plt.subplots(ncols=2)
    #         ax[0].set_title(f'{E_Incident} eV')
    #         ax[0].plot(detectorEnergies, B_iter)
    #         ax[0].set_ylabel('B_iter')
    #         ax[1].plot(detectorEnergies, B_1)
    #         ax[1].set_ylabel('B_1')
    #
    #         for i in range(2):
    #             ax[i].set_yscale('log')
    #             ax[i].set_xscale('log')
    #             ax[i].set_xlim(1E1, 1E4)
    #
    #         plt.show()
    #
    #     # using the iterative "bounces" trick again, we can calcualte ALL the secondary responses from the first backscatter "beam"
    #     # omniFlux = -1 * helperFitFuncs().calcTotal_NFlux(
    #     #     diffNFlux=np.array([B_1 for ptchIdx in range(19)]),
    #     #     pitchValues=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
    #     #     energyValues=incidentEnergies)  # -1 is added since I do High-to-low energy
    #
    #     # use the total backscatter flux to get a new secondaries curve
    #     # secondaryFlux_backscatter = self.calcSecondaries(detectorEnergies=detectorEnergies,
    #     #                                         Niterations=secondaryBackScatterToggles.Niterations_secondaries,
    #     #                                         InputOmniFlux=omniFlux,
    #     #                                         V0=V0)
    #
    #     ###################################
    #     # --- Iterate until convergence ---
    #     ###################################
    #     # create the Nth iteration variables and append the first step
    #     B_n = np.append([B_1], np.zeros(shape=(Niterations - 1, len(detectorEnergies))), axis=0)
    #
    #     # Generate the (Niterations - 1) number of steps for Backscatters + Secondaries
    #     for iterIdx in range(Niterations - 1):
    #
    #         ### Nth BACKSCATTER ###
    #
    #         # incident flux - Only at the non-zero flux points
    #         previous_backScatterFlux = B_n[iterIdx]
    #         incidentFlux = B_n[iterIdx][np.where(previous_backScatterFlux > 0)[0]]
    #
    #         # determine the incident energies - only at the non-zero flux points
    #         incidentEnergies = detectorEnergies[np.where(previous_backScatterFlux > 0)[0]]
    #
    #         # temporary storage variable
    #         B_temp = np.zeros(shape=(len(detectorEnergies)))
    #
    #         # Calc backscatter
    #         for engyIdx, E_Incident in enumerate(incidentEnergies):
    #             # --- get the spline curve ---
    #             backscatterSpline = self.generate_BackScatterCurve(E_Incident)
    #
    #             # --- Calculate the spline at the relevant energies ---
    #             B_iter = incidentFlux[engyIdx] * backscatterSpline(detectorEnergies)
    #
    #             # limit the fluxes to below the incident energy of the primary beam
    #             B_iter[np.where(detectorEnergies >= E_Incident)[0]] = 0
    #
    #             # Note:
    #             # the Evans curve does NOT consider backscatter from energies 10E-2 less than E_incident e.g. 1keV only generates 10eV to 1keV, thus
    #             # the spline with fail outside this region. Implement a fix to only evaluate energies within this range:
    #             B_iter[np.where(detectorEnergies < 1E-2 * E_Incident)] = 0
    #
    #             B_temp += B_iter
    #
    #             # fig, ax = plt.subplots(ncols=2)
    #             # ax[0].set_title(f'{E_Incident} eV')
    #             # ax[0].plot(detectorEnergies, B_iter)
    #             # ax[0].set_ylabel('B_iter')
    #             # ax[1].plot(detectorEnergies, B_temp)
    #             # ax[1].set_ylabel('B_temp')
    #             #
    #             # for i in range(2):
    #             #     ax[i].set_yscale('log')
    #             #     ax[i].set_xscale('log')
    #             #     ax[i].set_xlim(1E1,1E4)
    #             #     ax[i].set_ylim(1E-1,1E8)
    #             #
    #             # plt.show()
    #
    #         B_n[iterIdx + 1] = B_temp  # store this as the "next" backscatter profile
    #
    #     # sum all iterations into one variable
    #     backscatterFlux = np.sum(B_n, axis=0)
    #
    #     # ensure no values above V0 exists - ANY reflected flux that has parallel energy >V0 at the parallel potential will be able to
    #     # overcome the potential, and thus should NOT be included in the final product
    #     backscatterFlux[np.where(detectorEnergies > V0)[0]] = 0
    #     secondaryFlux_backscatter[np.where(detectorEnergies > V0)[0]] = 0
    #
    #     return backscatterFlux, secondaryFlux_backscatter
    #
    #     #
    #     #
    #     # for iterIdx in range(Niterations):
    #     #
    #     #     B_n = np.zeros(shape=(len(detectorEnergies)))
    #     #     S_n = np.zeros(shape=(len(detectorEnergies)))
    #     #
    #     #     # Determine the incoming Incident_omniDiffFlux is. Make special case if it's the first iteration
    #     #
    #     #
    #     #     for idx, E_Incident in enumerate(IncidentBeamEnergies):
    #     #
    #     #         # --- get the spline curve ---
    #     #         backscatterSpline = self.generate_BackScatterCurve(E_Incident)
    #     #
    #     #         # --- Calculate the spline at the relevant energies ---
    #     #
    #     #         B_iter = Incident_OmniDiffFlux[idx]*backscatterSpline(detectorEnergies)
    #     #         B_iter[np.where(detectorEnergies >= E_Incident)[0]] = 0 # limit the fluxes to below the incident energy of the primary beam
    #     #         backscatterFlux += B_n