# --- model classes ---
# Description: Place to put all the classes that represent the various ionospheric models
import numpy as np
from src.physicsModels.ionosphere.simToggles_iono import plasmaToggles
from spaceToolsLib.variables import m_to_km, cm_to_m


class vickrey1982:

    def calcRecombinationRate(self,altRange,data_dict):

        alpha = (cm_to_m**3)*(2.5E-6)*np.exp(-altRange/(51.2*m_to_km)) # in m^3s^-1, the factor of 1000 is to account for ki
        return alpha


class schunkNagy2009:
    def calcRecombinationRate(self, altRange, data_dict):
        def functionalForm(Te,A,B,exponent):
            return A*np.power(B/Te,exponent)

        alpha_dissociated = {'NO+':functionalForm(data_dict['Te'][0], 4E-7,300,0.5),
                             'O2+':functionalForm(data_dict['Te'][0], 2.4E-7,300,0.7),
                             'N2+':functionalForm(data_dict['Te'][0], 2.2E-7,300,0.39)}

        alpha_radiative = {  'H+':functionalForm(data_dict['Te'][0], 4.8E-12,250,0.7),
                             'He+': functionalForm(data_dict['Te'][0], 4.8E-12, 250, 0.7),
                             'N+': functionalForm(data_dict['Te'][0], 3.6E-12, 250, 0.7),
                             'O+':functionalForm(data_dict['Te'][0], 3.7E-12, 250, 0.7)}

        ni_total = np.sum([data_dict[f"n_{ionNam}"][0] for ionNam in plasmaToggles.wIons],axis=0)
        partials = []

        for ionNam in plasmaToggles.wIons:
            try:

                partials.append(alpha_dissociated[f'{ionNam}'] * data_dict[f'n_{ionNam}'][0])
            except:

                partials.append(alpha_radiative[f'{ionNam}'] * data_dict[f'n_{ionNam}'][0])

        # Convert to m^3 from cm^3
        return (1E6)*np.sum(np.array(partials),axis=0)/ni_total
