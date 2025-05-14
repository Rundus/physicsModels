
# --- imports ---
import matplotlib.pyplot as plt
import spaceToolsLib as stl
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import butter

# --- import the data ---
data_dict_EFI = stl.loadDictFromFile(r'C:\Data\ACESII\science\auroral_coordinates\low\ACESII_36364_E_Field_Auroral_Coordinates.cdf')

E_T = data_dict_EFI['E_tangent'][0]
E_N = data_dict_EFI['E_normal'][0]

# --- filter/smooth the data ---


print(data_dict_EFI.keys())

# fig, ax = plt.subplots(2)
# ax[0].plot(data_dict_EFI)





