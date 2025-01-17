# --- imports ---
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from myspaceToolsLib.physicsVariables import m_to_km,q0,m_e
from myspaceToolsLib.CDF_load import outputCDFdata


# --- SIMULATION TOGGLES ---
simLen = 600 # how many delta T steps to simulate
deltaT = 0.025 # in seconds
alt_Rez = 2000 # number of points in the altitude grid
simAltLow = 200*m_to_km # low altitude (in meters)
simAltHigh = 50000*m_to_km # high altitude (in meters)
simAlt = linspace(simAltLow, simAltHigh, alt_Rez)  # in METERS
simTime = linspace(0, deltaT * simLen, simLen + 1)  # in second

# --- WAVE ---
Z0Start = 40000*m_to_km# initial position of parallel E-Field
waveSpeed = 4000*m_to_km

# Wave - Sin
wavefreq = 1.5 # in Hz
tau0 = 1/wavefreq
lambdaPara = waveSpeed/wavefreq  # in km
EparaAmp = 5E-6

# Wave - Exponential
PhiAmp = 0.25
WaveFlatness = 0.65

# PARTICLES
elec_Z0 = np.array([38000, 46000, 10000])*m_to_km  # in km
elec_V0 =  np.array([0.8, 1.2,0])*waveSpeed
Nptcls = len(elec_V0)

# --- Plot toggles ---
Plot_theData = False
useSinPulse = True
figure_height = 6
figure_width = 12


def Epara_generator_sin(z, t, Vel, initialZ):
    if initialZ - Vel * (t - tau0) > z > initialZ - Vel * t:
        EperpVal = EparaAmp * (np.sin(((z - initialZ) + Vel * t) * (2 * np.pi / (Vel * tau0))))
    else:
        EperpVal = 0

    return EperpVal
def Phipara_generator_sin(z, t, Vel, initialZ):
    # the middle of wave
    if initialZ - Vel * (t - tau0) > z > initialZ - Vel * t:
        PhiPara = EparaAmp * (1-np.power( np.cos(   (((z - initialZ) + Vel * t) * (2 * np.pi / (Vel * tau0)))/2  ),2))
    else:
        PhiPara = 0

    return PhiPara

def Epara_generator_exp(z, t, Vel, initialZ):
    return (2 * PhiAmp * (z -initialZ- Vel * (t)) / np.sqrt(2 * np.pi * WaveFlatness * WaveFlatness)) * np.exp(-np.power( (z-initialZ - Vel * (t))/WaveFlatness, 2))
def Phipara_generator_exp(z, t, Vel, initialZ):
    return (PhiAmp / np.sqrt(2*np.pi*WaveFlatness*WaveFlatness)) * np.exp(-np.power((z -initialZ- Vel * (t))/WaveFlatness,2 ))


def Euler(yn, funcVal_n):
    yn1 = yn + deltaT * funcVal_n
    return yn1

def AdamsBashforth(yn1, funcVal_n, funcVal_n1):
    yn2 = yn1 + (3 / 2) * deltaT * funcVal_n1 - (1 / 2) * deltaT * funcVal_n
    return yn2


outputPath = r'C:\Data\Simulations\AlfvenResonanceVisualization.cdf'
data_dict_sim = {'EField_wf':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Epara'}],
                 'Phi_wf':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Wave Potential'}],
                 'elec_wf_x':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Position'}],
                 'elec_wf_y':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Energy'}],
                 'elec_wf_v':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS':None, 'LABLAXIS': 'Velocity'}],
                 'accel_wf':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Accelertion_LabFrame'}],
                 'EField_lf':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Epara'}],
                 'Phi_lf':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Wave Potential'}],
                 'elec_lf_x':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Position'}],
                 'elec_lf_y':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Energy'}],
                 'elec_lf_v':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'Velocity'}],
                 'accel_lf':[[],{'DEPEND_0': 'simTime', 'DEPEND_1': None, 'UNITS': None, 'LABLAXIS': 'acceleration_LabFrame'}],
                 'simAlt':[simAlt,{'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'simAlt'}],
                 'simTime':[simTime,{'DEPEND_0': 'simAlt', 'UNITS': None, 'LABLAXIS': 'simTime'}]}
#%%%%%%%%%%%%%%%%%%%%%%%%
#### LABRATORY FRAME ####
#%%%%%%%%%%%%%%%%%%%%%%%%
# --- Initialize Parallel E-Field ---


EFunc =Epara_generator_sin if useSinPulse else Epara_generator_exp
PhiFunc = Phipara_generator_sin if useSinPulse else Phipara_generator_exp
# populate the Epara Data
data_dict_sim['EField_wf'][0].append([EFunc(z=val, t=0, initialZ=Z0Start, Vel=waveSpeed) for val in simAlt])
data_dict_sim['Phi_wf'][0].append([PhiFunc(z=val, t=0, initialZ=Z0Start, Vel=waveSpeed) for val in simAlt])
for tmeIdx,tme in enumerate(simTime):
    data_dict_sim['EField_lf'][0].append([EFunc(z=val, t=tme, initialZ=Z0Start, Vel=waveSpeed) for val in simAlt])
    data_dict_sim['Phi_lf'][0].append([PhiFunc(z=val, t=tme, initialZ=Z0Start, Vel=waveSpeed) for val in simAlt])
    if tmeIdx > 0:
        data_dict_sim['EField_wf'][0].append(data_dict_sim['EField_wf'][0][-1])
        data_dict_sim['Phi_wf'][0].append(data_dict_sim['Phi_wf'][0][-1])

# --- populate the electrons ---
# INITALIZE - Lab Frame
data_dict_sim['elec_lf_x'][0].append(elec_Z0)
data_dict_sim['elec_lf_v'][0].append(-1*elec_V0)
EparaInit = data_dict_sim['EField_lf'][0][0]
data_dict_sim['accel_lf'][0].append([ (-q0/m_e)*EparaInit[np.abs(simAlt - data_dict_sim['elec_lf_x'][0][0][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])
data_dict_sim['elec_lf_y'][0].append([data_dict_sim['Phi_lf'][0][0][np.abs(simAlt - data_dict_sim['elec_lf_x'][0][0][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])

# INITALIZE - Wave Frame
data_dict_sim['elec_wf_x'][0].append(elec_Z0)
data_dict_sim['elec_wf_v'][0].append(-1*(elec_V0-waveSpeed))
EparaInit = data_dict_sim['EField_wf'][0][0]
data_dict_sim['accel_wf'][0].append([(-q0/m_e)*EparaInit[np.abs(simAlt - data_dict_sim['elec_wf_x'][0][0][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)] )
data_dict_sim['elec_wf_y'][0].append([data_dict_sim['Phi_wf'][0][0][np.abs(simAlt - data_dict_sim['elec_wf_x'][0][0][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])

# EULER - Lab Frame
data_dict_sim['elec_lf_v'][0].append([Euler(yn=data_dict_sim['elec_lf_v'][0][0][ptclIdx],funcVal_n=data_dict_sim['accel_lf'][0][0][ptclIdx]) for ptclIdx in range(Nptcls)])
data_dict_sim['elec_lf_x'][0].append([Euler(yn=data_dict_sim['elec_lf_x'][0][0][ptclIdx],funcVal_n=data_dict_sim['elec_lf_v'][0][0][ptclIdx]) for ptclIdx in range(Nptcls)])
data_dict_sim['elec_lf_y'][0].append([data_dict_sim['Phi_lf'][0][1][np.abs(simAlt - data_dict_sim['elec_lf_x'][0][1][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])
Epara_Euler = data_dict_sim['EField_lf'][0][1]
data_dict_sim['accel_lf'][0].append([(-q0/m_e)*Epara_Euler[np.abs(simAlt - data_dict_sim['elec_lf_x'][0][1][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])


# EULER - Wave Frame
data_dict_sim['elec_wf_v'][0].append([Euler(yn=data_dict_sim['elec_wf_v'][0][0][ptclIdx],funcVal_n=data_dict_sim['accel_wf'][0][0][ptclIdx]) for ptclIdx in range(Nptcls)])
data_dict_sim['elec_wf_x'][0].append([Euler(yn=data_dict_sim['elec_wf_x'][0][0][ptclIdx],funcVal_n=data_dict_sim['elec_wf_v'][0][0][ptclIdx])for ptclIdx in range(Nptcls)])
data_dict_sim['elec_wf_y'][0].append([data_dict_sim['Phi_wf'][0][1][np.abs(simAlt - data_dict_sim['elec_wf_x'][0][1][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])
Epara_Euler = data_dict_sim['EField_wf'][0][1]
data_dict_sim['accel_wf'][0].append([(-q0/m_e)*Epara_Euler[np.abs(simAlt -  data_dict_sim['elec_wf_x'][0][1][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])


for idx, t in enumerate(simTime):

    if idx > 1:
        # ---------------------------
        # Adams Bashforth - Lab Frame
        # ---------------------------
        Epara_n = data_dict_sim['EField_lf'][0][idx-2]
        elec_x_n = data_dict_sim['elec_lf_x'][0][idx-2]
        elec_v_n = data_dict_sim['elec_lf_v'][0][idx-2]
        accel_n = data_dict_sim['accel_lf'][0][idx-2]
        Epara_n1 = data_dict_sim['EField_lf'][0][idx - 1]
        elec_x_n1 = data_dict_sim['elec_lf_x'][0][idx - 1]
        elec_v_n1 = data_dict_sim['elec_lf_v'][0][idx - 1]
        accel_n1 = data_dict_sim['accel_lf'][0][idx - 1]
        Epara_n2 = data_dict_sim['EField_lf'][0][idx]

        # determine new velocity/position
        elec_v_n2 = [AdamsBashforth(yn1=elec_v_n1[ptclIdx],funcVal_n=accel_n[ptclIdx],funcVal_n1=accel_n1[ptclIdx]) for ptclIdx in range(Nptcls)]
        elec_x_n2 = [AdamsBashforth(yn1=elec_x_n1[ptclIdx],funcVal_n=elec_v_n[ptclIdx],funcVal_n1=elec_v_n1[ptclIdx]) for ptclIdx in range(Nptcls)]
        data_dict_sim['elec_lf_v'][0].append(elec_v_n2)
        data_dict_sim['elec_lf_x'][0].append(elec_x_n2)

        # determine new acceleration/yval
        data_dict_sim['accel_lf'][0].append([(-q0/m_e)*Epara_n2[np.abs(simAlt - elec_x_n2[ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])
        data_dict_sim['elec_lf_y'][0].append([data_dict_sim['Phi_lf'][0][idx][np.abs(simAlt - data_dict_sim['elec_lf_x'][0][idx][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])

        # ----------------------------
        # Adams Bashforth - Wave Frame
        # ----------------------------
        # get previous data
        Epara_n = data_dict_sim['EField_wf'][0][idx - 2]
        elec_x_n = data_dict_sim['elec_wf_x'][0][idx - 2]
        elec_v_n = data_dict_sim['elec_wf_v'][0][idx - 2]
        accel_n = data_dict_sim['accel_wf'][0][idx - 2]
        Epara_n1 = data_dict_sim['EField_wf'][0][idx - 1]
        elec_x_n1 = data_dict_sim['elec_wf_x'][0][idx - 1]
        elec_v_n1 = data_dict_sim['elec_wf_v'][0][idx - 1]
        accel_n1 = data_dict_sim['accel_wf'][0][idx - 1]
        Epara_n2 = data_dict_sim['EField_wf'][0][idx]

        # determine new velocity/position
        elec_v_n2 = [AdamsBashforth(yn1=elec_v_n1[ptclIdx], funcVal_n=accel_n[ptclIdx], funcVal_n1=accel_n1[ptclIdx]) for ptclIdx in range(Nptcls)]
        elec_x_n2 = [AdamsBashforth(yn1=elec_x_n1[ptclIdx], funcVal_n=elec_v_n[ptclIdx], funcVal_n1=elec_v_n1[ptclIdx]) for ptclIdx in range(Nptcls)]
        data_dict_sim['elec_wf_v'][0].append(elec_v_n2)
        data_dict_sim['elec_wf_x'][0].append(elec_x_n2)

        # determine new acceleration/yval
        data_dict_sim['accel_wf'][0].append([(-q0 / m_e) * Epara_n2[np.abs(simAlt - elec_x_n2[ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])
        data_dict_sim['elec_wf_y'][0].append([data_dict_sim['Phi_wf'][0][idx][np.abs(simAlt - data_dict_sim['elec_wf_x'][0][idx][ptclIdx]).argmin()] for ptclIdx in range(Nptcls)])


# "center" the lab frame data
centerPoint = (simAltHigh - simAltLow - lambdaPara)/2
for idx, t in enumerate(simTime):
    data_dict_sim['elec_wf_x'][0][idx] = np.array(data_dict_sim['elec_wf_x'][0][idx] - np.abs(centerPoint -Z0Start ))
    data_dict_sim['EField_wf'][0][idx] = [EFunc(z=val, t=0, initialZ=centerPoint, Vel=waveSpeed) for val in simAlt]
    data_dict_sim['Phi_wf'][0][idx] = [PhiFunc(z=val, t=0, initialZ=centerPoint, Vel=waveSpeed) for val in simAlt]

if Plot_theData:
    import itertools
    colors = itertools.cycle(["r", "b", "g"])
    colorChoice = [next(colors) for val in range(Nptcls)]

    for t in range(len(simTime)):
        fig, ax = plt.subplots(2)
        fig.set_size_inches(figure_width, figure_height)

        ax[0].set_title('Lab Frame')
        ax[0].plot(simAlt, data_dict_sim['EField_lf'][0][t],color='tab:blue')
        ax[0].plot(simAlt, data_dict_sim['Phi_lf'][0][t], color='tab:red')
        for ptclIdx in range(Nptcls):
            ax[0].scatter(data_dict_sim['elec_lf_x'][0][t][ptclIdx],data_dict_sim['elec_lf_y'][0][t][ptclIdx],color=colorChoice[ptclIdx],s=40)

        ax[0].set_xlim(0, simAltHigh)
        ax[0].set_ylim(-1.5*(EparaAmp),1.5*EparaAmp)
        E_textXY =(Z0Start - waveSpeed*t*deltaT + (lambdaPara/2), 0)
        Phi_textXY = (Z0Start - waveSpeed * t * deltaT+ (lambdaPara/2), EparaAmp)
        ax[0].annotate('$E_{\parallel}$', E_textXY, color='tab:blue')
        ax[0].annotate('$\Phi_{max}$',Phi_textXY, color='tab:red')
        ax[0].scatter(*E_textXY)
        ax[0].scatter(*Phi_textXY)

        ax[1].set_title('Wave Frame')
        ax[1].plot(simAlt, data_dict_sim['EField_wf'][0][t],color='tab:blue')
        ax[1].plot(simAlt, data_dict_sim['Phi_wf'][0][t], color='tab:red')
        for ptclIdx in range(Nptcls):
            ax[1].scatter(data_dict_sim['elec_wf_x'][0][t][ptclIdx], data_dict_sim['elec_wf_y'][0][t][ptclIdx], color=colorChoice[ptclIdx], s=40)

        ax[1].set_xlim(0, simAltHigh)
        ax[1].set_ylim(-1.5 * (EparaAmp), 1.5 * EparaAmp)

        E_textXY = (centerPoint + lambdaPara/2, 0)
        Phi_textXY = (centerPoint+lambdaPara/2, EparaAmp)
        ax[1].annotate('$E_{\parallel}$', E_textXY, color='tab:blue')
        ax[1].annotate('$\Phi_{max}$', Phi_textXY, color='tab:red')
        ax[1].scatter(*E_textXY)
        ax[1].scatter(*Phi_textXY)

        plt.tight_layout()
        plt.show()

#####################
# --- OUTPUT DATA ---
#####################
outputCDFdata(outputPath=outputPath,data_dict=data_dict_sim)



