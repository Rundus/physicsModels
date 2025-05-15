
# --- imports ---
import matplotlib.pyplot as plt
import spaceToolsLib as stl
import numpy as np
from scipy.signal import savgol_filter
from spacepy import pycdf

# --- import the data ---
data_dict_EFI = stl.loadDictFromFile(r'C:\Data\ACESII\science\auroral_coordinates\low\ACESII_36364_E_Field_Auroral_Coordinates.cdf')
E_T = data_dict_EFI['E_tangent'][0]
E_N = data_dict_EFI['E_normal'][0]
L_shell = data_dict_EFI['L-Shell'][0]
deltaT = (pycdf.lib.datetime_to_tt2000(data_dict_EFI['Epoch'][0][5001]) - pycdf.lib.datetime_to_tt2000(data_dict_EFI['Epoch'][0][5000]))/1E9


# Filter params
order = 4
fs = 1/deltaT
freq_cut = 0.05
filt_type= 'lowpass'
E_T_filt = stl.butter_filter(
    data=E_T,
    lowcutoff=freq_cut,
    highcutoff=freq_cut,
    filtertype=filt_type,
    fs=fs,
    order=order
)

E_N_filt = stl.butter_filter(
    data=E_N,
    lowcutoff=freq_cut,
    highcutoff=freq_cut,
    filtertype=filt_type,
    fs=fs,
    order=order
)


# wl = 1000
# deriv = 0
# polyorder = 3
# E_T_filt = savgol_filter(x=E_T,
#                        window_length=wl,
#                          deriv=deriv,
#                        polyorder=polyorder)
# E_N_filt = savgol_filter(x=E_N,
#                          window_length=wl,
#                          deriv=deriv,
#                          polyorder=polyorder)





# --- filter/smooth the data ---
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(L_shell, E_T, color='tab:blue', label='$E_{T}$ (base)')
ax[0].plot(L_shell,E_T_filt,color='tab:red',label='$E_{T}$ (filter)')
ax[0].set_ylabel('$E_{T}$ [mV/m]')
ax[1].plot(L_shell, E_N, color='tab:blue', label='$E_{N}$ (base)')
ax[1].plot(L_shell,E_N_filt,color='tab:red',label='$E_{N}$ (filter)')
ax[1].set_ylabel('$E_{N}$ [mV/m]')
ax[1].set_xlabel('L-Shell')

for i in range(2):
    ax[i].legend()
    ax[i].axhline(y=0, color='black', linestyle='--')
    if i == 0:
        ax[i].set_ylim(-0.13, -0.09)
    else:
        ax[i].set_ylim(-0.13,0.13)

plt.show()
# plt.savefig(r'C:\Data\physicsModels\ionosphere\electricField\test_scripts\E_field_smoothed')





