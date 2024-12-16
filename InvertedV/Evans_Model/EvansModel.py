# --- EvanModel.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: Use the Evans 1974 model to calculate the
# secondary/backscatter responses for our data

#################
# --- IMPORTS ---
#################
import matplotlib.pyplot as plt
import numpy as np
from ACESII_code.myImports import *
from my_matplotlib_Assets.colorbars.apl_rainbow_black0 import apl_rainbow_black0_cmap
from ACESII_code.class_var_func import EpochTo_T0_Rocket, q0,m_e
from scipy.interpolate import griddata
start_time = time.time()





##########################
# --- --- --- --- --- ---
# --- LOADING THE DATA ---
# --- --- --- --- --- ---
##########################
prgMsg('Loading Data')
from ACESII_code.Science.InvertedV.Evans_class_var_funcs import loadDiffNFluxData
diffNFlux,Epoch,Energy,Pitch = loadDiffNFluxData()
Done(start_time)