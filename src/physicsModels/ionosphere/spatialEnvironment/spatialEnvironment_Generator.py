# --- spatialEnvironment Generator ---
# Determines the Lat, Alt, ILat of the simulation environment and outputs the
# regularized grid of the 2D slice along a rocket trajectory

# --- imports ---

from spaceToolsLib.variables import m_to_km, Re
from spaceToolsLib.tools import CHAOS
from spaceToolsLib.tools.CDF_output import outputCDFdata
from copy import deepcopy
from numpy import array, degrees, arccos
from numpy.linalg import norm
from datetime import datetime
from spacepy import coordinates as coord
from spacepy.time import Ticktock


def generateSpatialEnvironment():

    from src.physicsModels.ionosphere.simToggles_Ionosphere import SpatialToggles

    # generate the 2D grid of magnetic field values


