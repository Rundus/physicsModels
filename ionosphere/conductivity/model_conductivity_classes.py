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

    def electronIon_CollisionFreq(self,data_dict_neutral,data_dict_plasma):
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

    def electronIon_CollisionFreq(self,data_dict_neutral,data_dict_plasma):
        Te = data_dict_plasma['Te'][0]
        ne = data_dict_plasma['ne'][0]*1E-6
        return np.power(Te, -3/2)*ne*(34 + 4.18*np.log(np.power(Te,3)/ne))

