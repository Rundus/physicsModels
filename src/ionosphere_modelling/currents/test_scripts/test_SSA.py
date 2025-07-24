import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np
# get the E-Field data
import spaceToolsLib as stl
data_dict_EField = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\electricField\electric_Field.cdf')
data_dict_conductivity = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\conductivity\conductivity.cdf')

# select the specific altitude slice
choice_idx = 50
E_Field = deepcopy(data_dict_EField['E_N'][0][:, choice_idx])
sigma_P = deepcopy(data_dict_conductivity['sigma_P'][0][:, choice_idx])
data = sigma_P
# data = E_Field

###########################
# --- Calculate the SSA ---
###########################
fH = 0.67
fL = 0.545
T = (2/(fH + fL))/0.05
num_of_spin_periods = 10
wLen = int(num_of_spin_periods*T)
SSA = stl.SSA(tseries=data, L=wLen, mirror_percent=0.1)
SSA_eigenvecs = SSA.components_to_df()
wComponents = [0, 1, 2, 3, 4, 5, 6, 7]
wGroups = [[0]] # conductivity sufficent
# wGroups = [[0, 1]] # conductivity sufficent
# SSA.plot_wcorr()
# SSA.plot_components(wComponents=wComponents,wGroups=wGroups)
# plt.show()

# Show the gradient
tseries = SSA.reconstruct(indices=[0,1,2,3])
grad = np.gradient(tseries)

fig, ax = plt.subplots(2)
ax[0].plot(tseries)
ax[1].axhline(y=0,color='red')
ax[1].plot(grad)
plt.show()

# NOTES:
# [1] The E-Field is more sensitive to it's own components than the conductivity
# [2] The conducticity wcorrelations are [0] and [1,2,3]
# [3] The E-Field wcorrelations are [0] and [1,2,3,4]


