# --- backScatter_Plotting.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: get the data from the backscatter Curves and plot the data WITHOUT regenerating all the curves again

from invertedV_fitting.primaryBeam_fitting.primaryBeam_classes import *
import spaceToolsLib as stl
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, shutil
import datetime as dt

##################
# --- PLOTTING ---
##################

Title_FontSize = 20
Label_FontSize = 25
Label_Padding = 8
Text_FontSize = 20

Tick_FontSize = 22
Tick_Length = 3
Tick_Width = 1.5
Tick_FontSize_minor = 15
Tick_Length_minor = 1
Tick_Width_minor = 1
Plot_LineWidth = 0.5
plot_MarkerSize = 12
Legend_fontSize = 14
dpi = 200


fig, ax = plt.subplots()
fig.set_figwidth(Figure_width)
fig.set_figheight(Figure_height)
ax.set_title(f'Upward Flux - Primary Beam Only\n{EpochFitData[tmeIdx]} UTC')
ax.plot(detectorEnergies, diffNFlux_avg[tmeIdx][2],'o-',color='black',label=r"$\alpha = 15^{\circ}$", linewidth=Plot_LineWidth)
ax.plot(modelEnergyGrid, degradedPrim_OmniDiff/np.pi,color='tab:red',label='Upward Degraded Prim. (Beam)', linewidth=Plot_LineWidth)
ax.plot(modelEnergyGrid, secondaries_OmniDiff / np.pi, color='tab:green', label='Upward Secondaries (Beam)', linewidth=Plot_LineWidth)
ax.plot(modelEnergyGrid, secondaries_OmniDiff / np.pi + degradedPrim_OmniDiff/np.pi , color='tab:blue', label='Total Response (Beam)', linewidth=Plot_LineWidth)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('[cm$^{-2}$s$^{-1}$sr$^{-1}$eV$^{-1}$]', fontsize=Label_FontSize)
ax.set_ylim(1E4, 5E7)
ax.set_xlim(20, 3E3)
ax.set_xlabel('Energy [eV]', fontsize=Label_FontSize)
ax.legend(fontsize=Legend_FontSize)
ax.grid(alpha=0.5)
ax.tick_params(axis='y', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
ax.tick_params(axis='x', labelsize=Tick_FontSize, length=Tick_Length, width=Tick_Width)
plt.savefig(rf'C:\Data\physicsModels\invertedV\backScatter\firstBounce\firstBounce_{tmeIdx}.png')
