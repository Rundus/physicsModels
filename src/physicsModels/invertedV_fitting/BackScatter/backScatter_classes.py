# --- model_primaryBeam_classes --
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np

from src.physicsModels.invertedV_fitting.BackScatter.Evans_Model.parameterizationCurves_Evans1974_classes import *
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

    def calcBackscatter(self, energy_Grid, beam_Energies, beam_IncidentElectronFlux):
        '''
        :param energy_Grid: 1D grid of energies for the output curves. Arbitrary Length
        :param beam_Energies: 1D array of energy values the for the input Beam.
        :param beam_OmniDiffFlux: 1D array of incident electron values of the beam [cm^-2s^-1]. Length = Len(beam_Energies). Calculated from integrating varPhi(E) over a deltaE for each energy to perserve the total number of electrons regardless of energy grid resolution.
        :return:
        upWard omniDiffFlux (Degraded Primaries) - 1D array of ionospheric degraded primaries flux in units of [cm^-2 s^-1 eV^-1]
        upWard omniDiffFlux (Secondaries) - 1D array of ionospheric secondaries flux in units of [cm^-2 s^-1 eV^-1]
        V0 (kwarg) - Scalar value of the parallel potential. Used to limit the secondary/backscatter flux. If unspecified, the minimum value of beam_Energies is taken
        onlySecondaries (kwarg) - boolean. returns only zeros for the backscatter flux on the energy_Grid variable.
        '''

        model = Evans1974()

        # --- define the outputs ---
        secondariesFlux = np.zeros(shape=(len(energy_Grid)))
        degradedPrimFlux = np.zeros(shape=(len(energy_Grid)))

        # --- loop over beam energies ---
        # print(len(beam_Energies),len(beam_OmniDiffFlux))
        for engyIdx, E_Incident in enumerate(beam_Energies):

            # --- Secondaries ---
            spline = model.generate_SecondariesCurve() # get the secondaries spline
            curve_secondaries = spline(energy_Grid)
            curve_secondaries[np.where(energy_Grid > 1E3)[0]] = 0 # no secondaries above the incident energy
            curve_secondaries[np.where(energy_Grid > E_Incident)[0]] = 0  # no secondaries above the incident energy
            curve_secondaries[np.where(curve_secondaries < 0)[0]] = 0 # no negative values
            secondariesFlux += curve_secondaries*beam_IncidentElectronFlux[engyIdx]

            # --- DegradedPrimaries ---
            spline = model.generate_BackScatterCurve(E_Incident) # get the degradedPrimaries
            curve_degradedPrimaries = spline(energy_Grid)
            curve_degradedPrimaries[np.where(energy_Grid < E_Incident * 1E-2)[0]] = 0 # only energies between 1E-2 E_energy and 1 E_energy
            curve_degradedPrimaries[np.where(energy_Grid > E_Incident)[0]] = 0
            curve_degradedPrimaries[np.where(curve_degradedPrimaries < 0)[0]] = 0 # no negative values
            degradedPrimFlux += curve_degradedPrimaries*beam_IncidentElectronFlux[engyIdx]

        return degradedPrimFlux, secondariesFlux

    def calcIonosphericResponse(self, beta, V0, targetPitch, response_energy_Grid, beam_EnergyGrid, beam_diffNFlux):
        '''
        :param beta: - Scalar. Value of B_max/B_min indicating the height of the lower boundary of the Inverted-V
        :param response_energy_Grid: - 1D array of energy values with arbitrary number of points. Must be between 0 to 1keV.
        :param targetPitch: - Scalar. Pitch angle (in deg) indicating the slice in angular space of the outputted secondary/backscatter values
        :param beam_Energies: - 1D array of energy grid values for the Primary Inverted-V beam of electrons. Arbitrary number of points allowed i.e. raw data or model data accepted.
        :param beam_diffNFlux: - 1D array of differential number flux values (j_N) for the beam.
        :param V0: parallel potential value of inverted-V
        :return:
        degradedPrimaries_Flux - 1D array of total electron differential Flux [cm^-2 s^-1 eV^-1] values for degraded primaries which has been iterated 6 times
        secondaries_Flux - 1D array of total electron differential Flux [cm^-2 s^-1 eV^-1] values for secondary electrons which has been iterated 6 times
        '''


        ######################
        # --- PRIMARY BEAM ---
        ######################

        # Description: Evans 1974 pointed out two parts to the beam pitch angle:
        # (1) The beam exiting the inverted-V will be collimated by alpha = arcsin(sqrt(E/(E + V0)))
        # (2) Magnetic mirroring effects will also widen the beam
        # At an arbitrary altitude the beam will widen due to (2), thus the beam itself may not be visible at certain eneriges for a given pitch angle
        # e.g. at low energies, the beam is really collimated, so low energies may not show up at ~60deg for a given altitude

        # inverted-V lower boundary - maximum pitch angles of the beam for a given energy
        alpha_m = deepcopy((180 / np.pi) * np.arcsin(np.sqrt(response_energy_Grid / (response_energy_Grid + V0))))

        # atmosphere boundary - pitch angle of beam electrons which had the highest pitch for a given energy
        alpha_atm = np.degrees(np.arcsin(np.sqrt(beta) * np.sin(np.radians(alpha_m))))
        alpha_atm = np.nan_to_num(alpha_atm, nan=90)

        # atmosphere boundary - beam pitch angle required to reach 90deg at Z_atm
        alpha_M_star = np.degrees(np.arcsin(1 / np.sqrt(beta)))

        # atmosphere boundary - filter beam for electrons that have widened enough to reach "target_pitch"
        # BE CAREFUL: alpha_m corresponds to the BEAM pitch angles, NOT the response_energy_Grid pitch angles.
        jN_targetPitch = deepcopy(beam_diffNFlux)
        jN_targetPitch[np.where(alpha_atm < targetPitch)[0]] = 0

        ######################
        # --- BACKSCATTER ---
        ######################
        nIterations = backScatterToggles.niterations_backscatter
        R = 0.1

        # --- define the outputs ---
        sec_Flux = deepcopy(np.zeros(shape=(nIterations + 1, len(response_energy_Grid))))
        dgdPrim_Flux = deepcopy(np.zeros(shape=(nIterations + 1, len(response_energy_Grid))))

        # --- get varPhi(E) of the beam [cm^-2 s^-1 eV^-1]---
        # Description: Integrate the beam over pitch angle by implementing a Gamma angle
        Gamma = deepcopy(np.array([alpha_atm[i] if alpha_m[i] < alpha_M_star else 90 for i in range(len(alpha_atm))]))
        varPhi_E_beam = np.pi * jN_targetPitch * np.power(np.sin(np.radians(Gamma)), 2)

        # --- incident Electron Flux ---
        # desciption: We integrate each varPhi(E) point around a deltaE to get the total number of electrons at that energy.
        # This integration preserves the total number of electrons in the beam
        beamSpline = CubicSpline(y=varPhi_E_beam, x=beam_EnergyGrid)
        deltaE = (beam_EnergyGrid[1] - beam_EnergyGrid[0]) / 2
        incident_ElecFlux = np.array([
            simpson(
                x=[beam_EnergyGrid[idx] - deltaE, beam_EnergyGrid[idx] + deltaE],
                y=[beamSpline(beam_EnergyGrid[idx] - deltaE), beamSpline(beam_EnergyGrid[idx] + deltaE)]
            )
            for idx in range(len(varPhi_E_beam))])
        incident_ElecEnergy = beam_EnergyGrid

        # --- initial Impact ---
        degradedPrimaries, secondaries = self.calcBackscatter(
            energy_Grid=response_energy_Grid,
            beam_Energies=incident_ElecEnergy,
            beam_IncidentElectronFlux=incident_ElecFlux
        )

        # --- apply the G(E) factor ---
        # this accounts for electrons with enough energy to escape the parallel potential
        lostCondition = V0 * (beta / (beta - np.power(np.sin(np.radians(targetPitch)),2)))
        G_E = []

        for idx, eVal in enumerate(response_energy_Grid):
            if eVal > lostCondition:
                G_E.append(0)
            elif V0 <= eVal <= lostCondition:
                G_E.append(1 - beta*(1 - V0/eVal))
            else:
                G_E.append(1)

        G_E = deepcopy(np.array(G_E))

        # store the first beam fluxes
        sec_Flux[0] = secondaries / (1 - R)  # account for the ENTIRE secondaries cascade by multiplying by 1/(1-R)
        dgdPrim_Flux[0] = degradedPrimaries

        ####################
        # --- ITERATIONS ---
        ####################
        # Perform same operation nIteration more times
        for loopIdx in range(1, nIterations + 1):
            # --- second impact ---
            varPhi_E_previous = dgdPrim_Flux[loopIdx - 1]
            beamSpline = CubicSpline(y=varPhi_E_previous, x=response_energy_Grid)
            deltaE = (response_energy_Grid[1] - response_energy_Grid[0]) / 2
            incident_ElecFlux = np.array([
                simpson(
                    x=[response_energy_Grid[idx] - deltaE, response_energy_Grid[idx] + deltaE],
                    y=[beamSpline(response_energy_Grid[idx] - deltaE), beamSpline(response_energy_Grid[idx] + deltaE)]
                )
                for idx in range(len(varPhi_E_previous))])
            incident_ElecEnergy = response_energy_Grid

            # Degraded primaries
            degradedPrimaries, secondaries = self.calcBackscatter(
                energy_Grid=response_energy_Grid,
                beam_Energies=incident_ElecEnergy,
                beam_IncidentElectronFlux=incident_ElecFlux
            )

            sec_Flux[loopIdx] = secondaries / (1 - R)
            dgdPrim_Flux[loopIdx] = degradedPrimaries

        # total up the flux and separate it into individual steradians
        sec_total = np.sum(sec_Flux, axis=0) / np.pi
        dgdPrim_total = np.sum(dgdPrim_Flux, axis=0) / np.pi

        # Apply G(E) factor
        sec_total = np.array(G_E)*sec_total
        dgdPrim_total = np.array(G_E)*dgdPrim_total

        return dgdPrim_total, sec_total, jN_targetPitch

