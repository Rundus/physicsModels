

# --- imports ---
import matplotlib.pyplot as plt
import spaceToolsLib as stl
import numpy as np

my_cmap= stl.apl_rainbow_black0_cmap()
my_cmap.set_bad('black')


# get the data
data_dict_currents = stl.loadDictFromFile('C:\Data\physicsModels\ionosphere\currents\currents.cdf')
data_dict_LShell =stl.loadDictFromFile('C:\Data\ACESII\science\L_shell\high\ACESII_36359_Lshell.cdf')


# plot the data
fig, ax = plt.subplots(2,sharex=True)
xData = data_dict_LShell['L-Shell'][0]
yData = data_dict_LShell['Energy'][0]
zData = data_dict_LShell['diffNFlux'][0][:,2,:].T
ax[0].pcolormesh(xData,yData,zData,cmap=my_cmap,norm='log')
ax[0].set_yscale('log')
ax[0].set_ylabel('Energy [eV]')

xData = data_dict_currents['simLShell'][0]
yData = data_dict_currents['simAlt'][0]/1000
zData = data_dict_currents['J_parallel_per_meter'][0]
cmap = ax[1].pcolormesh(xData,yData,zData.T,cmap='coolwarm', vmin=-5E-13, vmax=5E-13, norm='symlog')
ax[1].set_ylabel('Alt [km]')
ax[1].set_xlabel('L-Shell')
ax[1].set_xlim(8.5, 9.75)
cax = fig.add_axes([0.91, 0.1, 0.0155, 0.3])
colorbar=fig.colorbar(cax=cax, mappable=cmap)
colorbar.set_label('Current Density per meter [A/m$^{3}$]')

plt.savefig(r'C:\Data\physicsModels\ionosphere\currents\test_scripts\plot_currents.png')