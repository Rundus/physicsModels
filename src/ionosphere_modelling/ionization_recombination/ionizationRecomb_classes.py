# --- imports ---
import numpy as np
from numpy import power,log,pi
from spaceToolsLib.variables import kB,gravG,Me,Re
import spaceToolsLib as stl
from src.physicsModels.ionosphere.plasma_environment.plasma_toggles import plasmaToggles


class fang2010:

    def __init__(self, altRange, Tn, m_eff_n, rho_n, inputEnergies, varPhi_E):
        self.paramCoefficents = np.array([
            #      j =0           j=1          j=2          j=3
            [  1.24516E0,    1.45903E0, -2.42269E-1,  5.95459E-2], # i=1
            [  2.23976E0,  -4.22918E-7,  1.36458E-2,  2.53332E-3], # i=2
            [  1.41754E0,   1.44597E-1,  1.70433E-2,  6.39717E-4], # i=3
            [ 2.48775E-1,  -1.50890E-1,  6.30894E-9,  1.23707E-3], # i=4
            [-4.65119E-1,  -1.05081E-1, -8.95701E-2,   1.2245E-2], # i=5
            [ 3.86019E-1,   1.75430E-3, -7.42960E-4,  4.60881E-4], # i=6
            [-6.45454E-1,   8.49555E-4, -4.28581E-2, -2.99302E-3], # i=7
            [ 9.48930E-1,   1.97385E-1, -2.50660E-3, -2.06938E-3]  # i=8
        ])
        self.altRange = altRange
        self.Tn = Tn
        self.m_eff_n = m_eff_n
        self.rho_n = rho_n
        self.inputEnergies = inputEnergies
        self.varPhi_E = varPhi_E

    def scaleHeight(self):
        T_atm = self.Tn
        m_avg = self.m_eff_n
        grav_accel = (Me*gravG)/(np.power(Re*1000 + self.altRange, 2)) # in meters/sec^2

        # scale height needs to be in centimetres (cm), hence the *100
        H = np.array(100*kB*T_atm/(m_avg*grav_accel))
        return H

    def atmColumnMass(self, engyVal):
        rho = 1E-3*self.rho_n #atmosphere mass density in g/cm^3
        H = self.scaleHeight()
        y =np.array( [np.array([2 * np.power(engy/6E-6, 0.7) for engy in rho*H])/ val for val in engyVal])
        return y

    def calcCoefficents(self, engyVal):
        coefficents = np.array([np.array([np.exp(np.sum([self.paramCoefficents[i][j] * np.power(np.log(engy), j) for j in range(4)],axis=0)) for i in range(8)]) for engy in engyVal])
        return coefficents

    def f(self,y,coefficents):
        value = np.array([ C[0]*np.power(y[idx],C[1]) * np.exp( -C[2]*np.power(y[idx],C[3])) + C[4]*np.power(y[idx],C[5]) * np.exp( -C[6]*np.power(y[idx],C[7])) for idx,C in enumerate(coefficents)])
        return value

    def ionizationRate(self):
        H = self.scaleHeight() # length: (2000)
        y = self.atmColumnMass(self.inputEnergies) # length: (len(monoEnergyProfile),2000)
        C = self.calcCoefficents(self.inputEnergies)
        f_profiles = self.f(y, C)
        epsilion = 0.035
        q_profiles = np.array([(fluxVal/epsilion)*np.array(f_profiles[idx]/H) for idx, fluxVal in enumerate(self.varPhi_E)])
        q_total = np.sum(q_profiles, axis=0)
        return q_profiles, q_total # output as cm^-3



class vickrey1982:

    def calcRecombinationRate(self, altRange):

        alpha = (2.5E-6)*np.exp(-altRange/(51.2*stl.m_to_km)) # in cm^3s^-1
        # alpha_m3 = alpha/(stl.cm_to_m**3) # convert to m^3s6-1
        return alpha, []





class schunkNagy2009:
    def calcRecombinationRate(self, altRange, data_dict_plasma):
        def functionalForm(Te,A,B,exponent):
            return A*np.power(B/Te,exponent)

        alpha_dissociated = {'NO+':functionalForm(data_dict_plasma['Te'][0], 4E-7,300,0.5),
                             'O2+':functionalForm(data_dict_plasma['Te'][0], 2.4E-7,300,0.7),
                             'N2+':functionalForm(data_dict_plasma['Te'][0], 2.2E-7,300,0.39)}

        alpha_radiative = {  'H+':functionalForm(data_dict_plasma['Te'][0], 4.8E-12,250,0.7),
                             'He+': functionalForm(data_dict_plasma['Te'][0], 4.8E-12, 250, 0.7),
                             'N+': functionalForm(data_dict_plasma['Te'][0], 3.6E-12, 250, 0.7),
                             'O+':functionalForm(data_dict_plasma['Te'][0], 3.7E-12, 250, 0.7)}

        partials = []

        for ionNam in plasmaToggles.wIons:
            try:
                partials.append(alpha_dissociated[f'{ionNam}'] * data_dict_plasma[f'C_{ionNam}'][0])
            except:
                partials.append(alpha_radiative[f'{ionNam}'] * data_dict_plasma[f'n_{ionNam}'][0])

        # output with units cm^3s^-1
        alpha_total = np.sum(partials, axis=0)
        alpha_profiles = np.array(partials)

        return alpha_total, alpha_profiles


