# --- Model Ionosphere ---
# Inputs: list containing altitudes of interest
# Outputs: list containing lists of plasma paramters at all the input altitudes with format:
# [InputAltitude,rho, Te (in K), Ti (in K), n(O2+), n(N)+), N(O+), n(e)]
# all number density are in cm^-3

def JonesRees1972_Ionosphere(inputAltitudes):

    # --- get the model data ---
    import pandas as pd
    modelFilePath = r'C:\Users\cfelt\PycharmProjects\UIOWA_CDF_operator\ACESII_code\supportCode\IonosphereModels\JonesRees_IonosphereValues.xlsx'
    pandas_dict = pd.read_excel(modelFilePath)
    VariableNams = [thing for thing in pandas_dict]
    modelData = [pandas_dict[key][1:] for key in VariableNams]


    # --- interpolate input altitude onto dataset ---
    interpData = []
    from scipy.interpolate import CubicSpline
    for varNam in VariableNams:
        if varNam.lower() not in 'height' and varNam != 'Variable':
            # --- cubic interpolation ---
            splCub = CubicSpline(pandas_dict['Height'][1:],pandas_dict[varNam][1:])

            # --- evaluate the interpolation at all the new Epoch points ---
            interpData.append(array([splCub(hVal) for hVal in inputAltitudes]))


    # calculate rho
    m_O2p= 5.3133E-26
    m_NOp= 4.9826E-26
    m_Op = 2.6566E-26
    rho = m_O2p*array(interpData[2]) + m_NOp*array(interpData[3]) + m_Op*array(interpData[4])

    finalData = [inputAltitudes, rho] + interpData

    return {'Height':[finalData[0], {'UNITS': 'km'}],
            'rho':   [finalData[1], {'UNITS': 'kg/cm^-3'}],
            'T_e':   [finalData[2], {'UNITS': 'Kelvin'}],
            'T_i':   [finalData[3], {'UNITS': 'Kelvin'}],
            'n_O2p':[finalData[4], {'UNITS': 'cm^-3'}],
            'n_NOp':[finalData[5], {'UNITS': 'cm^-3'}],
            'n_Op': [finalData[6], {'UNITS': 'cm^-3'}],
            'n_e':  [finalData[7], {'UNITS': 'cm^-3'}],
            }