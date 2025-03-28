# --- imports ---
import numpy as np
from numpy import power,log,pi
from spaceToolsLib.variables import kB,q0,m_e,amu

class Nicolet1953:
    def electronNeutral_CollisionFreq(self,data_dict_neutral,data_dict_plasma):
        # nn is given as m^-3 and must be converted to cubic centimeters
        nn = 1E-6 * data_dict_neutral['nn'][0] # put into cm^-3
        Te = data_dict_plasma['Te'][0]

        # nn is the total number of molecules per cc, so ALL neutrals and
        # This equation ONLY applies to N2, O2 -electron collisions.
        return (5.4E-10)*nn*power(Te,0.5)

    def electronIon_CollisionFreq(self,data_dict_neutral, data_dict_plasma):
        # nn is given as m^-3 and must be converted to cubic centimeters
        ne = 1E-6 * data_dict_plasma['ne'][0]
        Te = data_dict_plasma['Te'][0]  # convert back to kelvin
        A = np.log(1 + np.power(((4/(np.sqrt(pi)*np.power(q0,3)*np.sqrt(ne)))*np.power(kB*Te/2,3/2)),2)) # mean distance between particles. Can be debeye, can be 1/(2ne). We choose debeye
        u = 0
        return (4/3) * ((pi*np.power(q0,4))/np.sqrt(2*pi*m_e*np.power(kB*Te,3))) * (1+u)*ne*A
class Evans1977:
    def ionNeutral_CollisionsFreq(self,data_dict_neutral,data_dict_plasma):
        nn = 1E-6 * data_dict_neutral['nn'][0]
        m_i = data_dict_plasma['m_eff_i'][0]
        m_n = data_dict_neutral['m_eff_n'][0]
        mu = (m_i*m_n)/(m_i+m_n)
        Ti = data_dict_plasma['Ti'][0]
        Tn = data_dict_neutral['Tn'][0] # should already be in kelvin
        return (8E-15)*(4/3)*(mu/m_i)*power(8*kB/pi,1/2)*nn * power((Ti/m_i + Tn/m_n),1/2)
class Johnson1961:
    def ionNeutral_CollisionsFreq(self,data_dict_neutral,data_dict_plasma):
        # nn is given as m^-3 and must be converted to cubic centimeters
        nn = data_dict_neutral['nn'][0] * 1E-6  # must be in cm^-3
        ni = data_dict_plasma['ni'][0] * 1E-6  # must be in cm^-3
        A = data_dict_neutral['m_eff_n'][0]/(amu) # mean NEUTRAL molecular mass in ATOMIC MASS UNITS

        # nn is the total number of molecules per cc, so ALL neutrals and
        # This equation ONLY applies to N2, O2 -electron collisions.
        return (2.6E-9) * (nn + ni) * np.power(A, -0.5)

    def electronIon_CollisionFreq(self,data_dict_neutral, data_dict_plasma, **kwargs):
        Te = data_dict_plasma['Te'][0]

        if list(kwargs.get('ne_data', [])) != []:
            ne = kwargs.get('ne_data')*1E-6 # convert from m^-3 to cm^-3
        else:
            ne = data_dict_plasma['ne'][0]*1E-6 # convert from m^-3 to cm^-3
        return np.power(Te, -3/2)*ne*(34 + 4.18*np.log(np.power(Te, 3)/ne))


class Leda2019:

    # The Leda model posits that the dominat ions in the ionosphere_models are: No+, O2+, O+
    # Similarly, the dominant neutrals are: N2, O2 and O

    # There are TWO types of collisions depending on the temperature of the gas:
    # (1) nonresonant electric-polarization collisions. Occurs at lower temperatures (occurs between all ion-neutral pairs)
    # (2) resonant charge-exchange collisions. Occurs at ~higher temperatures (ONLY occurs between alike-elements e.g. O2+ and O2, O+ and O etc.)

    def ionNeutral_CollisionsFreq(self, data_dict_neutral, data_dict_plasma, ionKey):
        # ion key options: ['Op', 'O2p', 'NOp']
        # This method returns BOTH the resonant and non-resonant collision freq combined

        # get the average temperature
        Tr_grid = (data_dict_neutral['Tn'][0] + data_dict_plasma['Ti'][0]) / 2

        # output grid
        output = np.zeros(shape=(len(Tr_grid), len(Tr_grid[0]), 3))

        # create the collision freq matrix
        a00 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])),fill_value=4.355)
        a01 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])), fill_value=4.28)
        a02 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])), fill_value=2.445)
        a10 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])), fill_value=4.146)
        cO2pO2_grid = np.maximum(4.079, 0.2617 * np.sqrt(Tr_grid) * np.power(1 - 0.07351 * np.log10(Tr_grid), 2))
        a12 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])), fill_value=2.318)
        a30 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])), fill_value=6.847)
        a31 = np.full(shape=(len(Tr_grid), len(Tr_grid[0])), fill_value=6.661)
        cOpO_grid = np.maximum(4.014, 0.3683 * np.sqrt(Tr_grid) * np.power(1 - 0.06482 * np.log10(Tr_grid), 2))

        for idx1, LShell in enumerate(Tr_grid):
            for idx2, Alt in enumerate(Tr_grid[0]):

                matrix = np.array([   [a00[idx1][idx2], a01[idx1][idx2],         a02[idx1][idx2]],
                                      [a10[idx1][idx2], cO2pO2_grid[idx1][idx2], a12[idx1][idx2]],
                                      [a30[idx1][idx2], a31[idx1][idx2],         cOpO_grid[idx1][idx2]],
                                      ])

                vec = np.array([data_dict_neutral['N2'][0][idx1][idx2], data_dict_neutral['O2'][0][idx1][idx2], data_dict_neutral['O'][0][idx1][idx2]])

                output[idx1][idx2] = (1E-16)* np.matmul(matrix, vec)

        if ionKey == 'NO+':
            return output[:, :, 0]
        elif ionKey == 'O2+':
            return output[:, :, 1]
        elif ionKey == 'O+':
            return output[:, :, 2]
        else:
            raise Exception('Invalid Ion Key')

    def electronNeutral_CollisionFreq(self, data_dict_neutral, data_dict_plasma, neutralKey):
        Te = data_dict_plasma['Te'][0]

        if neutralKey == 'N2':
            n_N2 = data_dict_neutral['N2'][0]
            return 1E-16*0.233*(1- Te *(1.21E-4))*Te*n_N2
        elif neutralKey == 'O2':
            n_O2 = data_dict_neutral['O2'][0]
            return 1E-16*1.82*(1 + np.sqrt(Te)*3.6E-2)*np.sqrt(Te)*n_O2
        elif neutralKey == 'O':
            n_O = data_dict_neutral['O'][0]
            return 1E-16*0.89*(1 + Te*(5.7E-4))*np.sqrt(Te)*n_O
        else:
            raise Exception('Invalid Neutral Key')

