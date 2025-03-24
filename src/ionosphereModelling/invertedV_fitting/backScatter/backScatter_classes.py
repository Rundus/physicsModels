# --- model_primaryBeam_classes --
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np

from src.physicsModels.invertedV_fitting.backScatter.Evans_Model.parameterizationCurves_Evans1974_classes import *
from src.physicsModels.invertedV_fitting.simToggles_invertedVFitting import backScatterToggles

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

    def calcBackscatter(self, response_energy_Grid, incident_energy_grid, incident_number_flux):
        '''
        :param response_energy_Grid:
            1D grid of energies for the output curves. Arbitrary Length
        :param incident_energy_grid:
            1D array of energy values the for the input Beam.
        :param beam_IncidentElectronFlux:
            1D array of incident electron values of the beam [cm^-2s^-1]. Length = Len(beam_Energies). Calculated from integrating varPhi(E) over a deltaE for each energy to perserve the total number of electrons regardless of energy grid resolution.
        :return:
        upWard omniDiffFlux (Degraded Primaries)
            - 1D array of ionospheric degraded primaries flux in units of [cm^-2 s^-1 eV^-1]
        upWard omniDiffFlux (Secondaries)
            - 1D array of ionospheric secondaries flux in units of [cm^-2 s^-1 eV^-1]
        V0 (kwarg)
            - Scalar value of the parallel potential. Used to limit the secondary/backscatter flux. If unspecified, the minimum value of beam_Energies is taken
        onlySecondaries (kwarg)
            - boolean. returns only zeros for the backscatter flux on the energy_Grid variable.
        '''

        model = Evans1974()

        # --- define the outputs ---
        secondariesFlux = np.zeros(shape=(len(response_energy_Grid)))
        degradedPrimFlux = np.zeros(shape=(len(response_energy_Grid)))

        # --- loop over beam energies ---
        # print(len(beam_Energies),len(beam_OmniDiffFlux))
        for engyIdx, E_Incident in enumerate(incident_energy_grid):

            # --- Secondaries ---
            spline = model.generate_SecondariesCurve() # get the secondaries spline
            curve_secondaries = spline(response_energy_Grid)
            curve_secondaries[np.where(response_energy_Grid > 1E3)[0]] = 0 # no secondaries above the incident energy
            curve_secondaries[np.where(response_energy_Grid > E_Incident)[0]] = 0  # no secondaries above the incident energy
            curve_secondaries[np.where(curve_secondaries < 0)[0]] = 0 # no negative values
            secondariesFlux += curve_secondaries*incident_number_flux[engyIdx]

            # --- DegradedPrimaries ---
            spline = model.generate_BackScatterCurve(E_Incident) # get the degradedPrimaries
            curve_degradedPrimaries = spline(response_energy_Grid)
            curve_degradedPrimaries[np.where(response_energy_Grid < E_Incident * 1E-2)[0]] = 0 # only energies between 1E-2 E_energy and 1 E_energy
            curve_degradedPrimaries[np.where(response_energy_Grid > E_Incident)[0]] = 0
            curve_degradedPrimaries[np.where(curve_degradedPrimaries < 0)[0]] = 0 # no negative values
            degradedPrimFlux += curve_degradedPrimaries*incident_number_flux[engyIdx]

        return degradedPrimFlux, secondariesFlux

    def calcIonosphericResponse(self, beta, V0, response_energy_grid, beam_energy_grid, beam_jN):
        '''
        :param beta: - Scalar.
            Value of B_max/B_min indicating the height of the lower boundary of the Inverted-V
        :param response_energy_grid: -
            1D array of energy values with arbitrary number of points. Must be between 0 to 1keV.
        :param beam_energy_grid:
            - 1D array of energy grid values for the Primary Inverted-V beam of electrons. Arbitrary number of points allowed i.e. raw data or model data accepted.
        :param beam_diffNFlux:
            - 1D array of differential number flux values (j_N) for the beam.
        :param V0: Scalar.
            parallel potential value of inverted-V
        :return:
        degradedPrimaries_Flux
            - 1D array of total electron differential Flux [cm^-2 s^-1 eV^-1] values for degraded primaries which has been iterated 6 times
        secondaries_Flux
            - 1D array of total electron differential Flux [cm^-2 s^-1 eV^-1] values for secondary electrons which has been iterated 6 times
        beam_Flux
            - 1D array of beam flux values at the specific pitch angle
        '''


        ######################
        # --- PRIMARY BEAM ---
        ######################

        # Description: Evans 1974 pointed out two parts to the beam pitch angle:
        # (1) The beam exiting the inverted-V will be collimated by alpha = arcsin(sqrt(E/(E + V0)))
        # (2) Magnetic mirroring effects will also widen the beam
        # At an arbitrary altitude the beam will widen due to (2), thus the beam itself may not be visible at certain eneriges for a given pitch angle
        # e.g. at low energies, the beam is really collimated, so low energies may not show up at ~60deg for a given altitude

        alpha_m = deepcopy((180 / np.pi) * np.arcsin(np.sqrt(response_energy_grid / (response_energy_grid + V0))))

        # atmosphere boundary - pitch angle of beam electrons which had the highest pitch for a given energy
        alpha_I = np.degrees(np.arcsin(np.sqrt(beta) * np.sin(np.radians(alpha_m))))
        Gamma = np.nan_to_num(alpha_I, nan=90)

        # --- get varPhi(E) of the beam [cm^-2 s^-1 eV^-1]---
        # Description: Integrate the beam over pitch angle by implementing a Gamma angle
        varPhi_E_beam = np.pi * beam_jN * np.power(np.sin(np.radians(Gamma)), 2)

        # --- incident Electron Flux ---
        # desciption: We integrate each varPhi(E) point around a deltaE to get the total number of electrons at that energy.
        # This integration preserves the total number of electrons in the beam
        beamSpline = CubicSpline(y=varPhi_E_beam, x=beam_energy_grid)
        deltaE = (beam_energy_grid[1] - beam_energy_grid[0]) / 2
        para_num_flux_beam = np.array([ # Phi_parallel - Beam
            simpson(
                x=[beam_energy_grid[idx] - deltaE, beam_energy_grid[idx] + deltaE],
                y=[beamSpline(beam_energy_grid[idx] - deltaE), beamSpline(beam_energy_grid[idx] + deltaE)]
            )
            for idx in range(len(varPhi_E_beam))])

        ######################
        # --- BACKSCATTER ---
        ######################
        nIterations = backScatterToggles.niterations_backscatter
        R = 0.1

        # --- define the outputs ---
        sec_Flux = deepcopy(np.zeros(shape=(nIterations + 1, len(response_energy_grid))))
        dgdPrim_Flux = deepcopy(np.zeros(shape=(nIterations + 1, len(response_energy_grid))))

        # ----------------------
        # --- initial Impact ---
        # ----------------------
        degradedPrimaries, secondaries = self.calcBackscatter(
            response_energy_Grid=response_energy_grid,
            incident_energy_grid=beam_energy_grid,
            incident_number_flux=para_num_flux_beam
        )

        # store the first beam fluxes
        sec_Flux[0] = secondaries / (1 - R)  # account for the ENTIRE secondaries cascade by multiplying by 1/(1-R)
        dgdPrim_Flux[0] = degradedPrimaries

        ####################
        # --- ITERATIONS ---
        ####################

        # --- determine the G(E) factor ---
        # this accounts for electrons with enough energy to escape the parallel potential (see my notebook about this)
        # ONLY applies to backscatter that reaches/escapes from the inverted-V, NOT the primary beam
        G_E = []
        for idx, eVal in enumerate(response_energy_grid):
            if eVal > V0 * (beta / (beta - 1)):
                G_E.append(0)
            elif V0 <= eVal <= V0 * (beta / (beta - 1)):
                G_E.append(1 - beta * (1 - V0 / eVal))
            else:
                G_E.append(1)
        G_E = deepcopy(np.array(G_E))

        # Perform same operation nIteration more times
        for loopIdx in range(1, nIterations + 1):

            # --- second impact ---
            varPhi_E_previous = G_E*dgdPrim_Flux[loopIdx - 1] # apply G(E) factor to the backscatter varPhi FLux only
            beamSpline = CubicSpline(y=varPhi_E_previous, x=response_energy_grid)
            deltaE = (response_energy_grid[1] - response_energy_grid[0]) / 2
            incident_number_flux = np.array([
                simpson(
                    x=[response_energy_grid[idx] - deltaE, response_energy_grid[idx] + deltaE],
                    y=[beamSpline(response_energy_grid[idx] - deltaE), beamSpline(response_energy_grid[idx] + deltaE)]
                )
                for idx in range(len(varPhi_E_previous))])
            incident_energies = response_energy_grid

            # Degraded primaries
            degradedPrimaries, secondaries = self.calcBackscatter(
                response_energy_Grid=response_energy_grid,
                incident_energy_grid=incident_energies,
                incident_number_flux=incident_number_flux
            )

            sec_Flux[loopIdx] = secondaries / (1 - R)
            dgdPrim_Flux[loopIdx] = degradedPrimaries

        # Get the total of the number flux [cm^-2 s^-1]
        para_num_flux_sec = np.sum(sec_Flux, axis=0)
        para_num_flux_dgdPrim = np.sum(dgdPrim_Flux, axis=0)

        return para_num_flux_beam, para_num_flux_dgdPrim, para_num_flux_sec

    def calc_response_at_target_pitch(self, V0, beta , beam_jN, beam_energy_grid, sec_num_flux, dgdPrim_num_flux, energy_grid, target_pitch):
        '''
        :param V0:
        :param beta:
        :param beam_num_flux:
        :param beam_energy_grid:
        :param sec_num_flux:
        :param dgdPrim_num_flux:
        :param energy_grid:
        :param target_pitch:
        :return:
        '''

        alpha_m = deepcopy((180 / np.pi) * np.arcsin(np.sqrt(energy_grid / (energy_grid + V0))))

        # atmosphere boundary - pitch angle of beam electrons which had the highest pitch for a given energy
        alpha_I = np.degrees(np.arcsin(np.sqrt(beta) * np.sin(np.radians(alpha_m))))
        Gamma = np.nan_to_num(alpha_I, nan=90)

        # determine the lost condition for this specific target_pitch
        lostCondition = V0 * (beta / (beta - np.power(np.sin(np.radians(target_pitch)), 2)))

        # modify the degraded primaries
        dgdPrim_targetPitch = deepcopy(dgdPrim_num_flux)/np.pi
        dgdPrim_targetPitch[np.where(energy_grid > lostCondition)[0]] = 0

        # modify the secondaries
        sec_targetPitch = deepcopy(sec_num_flux)/np.pi
        sec_targetPitch[np.where(energy_grid > lostCondition)[0]] = 0

        # modify the primary beam
        jN_targetPitch = beam_jN
        jN_targetPitch[np.where(beam_energy_grid < lostCondition)[0]] = 0
        jN_targetPitch[np.where(beam_energy_grid < V0)[0]] = 0

        return dgdPrim_targetPitch, sec_targetPitch, jN_targetPitch