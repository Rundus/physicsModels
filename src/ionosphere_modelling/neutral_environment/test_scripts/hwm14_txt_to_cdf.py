# --- hwm14_txt_to_cdf.py ---
# --- Author: C. Feltman ---
# DESCRIPTION:


# --- Import ---
import numpy as np
import pandas as pd

file_path = r'C:\Data\physicsModels\ionosphere\neutral_environment\hwm14\hwm14_neutral_winds_20221120_052000.txt'

def hwm14_txt_to_cdf():

    # prepare the output data
    data_dict_output = {'zonal_quiet': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Zonal Quiet'}],
                        'meridional_quiet': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Meridional Quiet'}],
                        'meridional_disturbed': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Meridional Disturbed'}],
                        'zonal_disturbed': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Zonal Disturbed'}],
                        'meridional_total': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Meridional Total'}],
                        'zonal_total': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Zonal Total'}],
                        'alt':[[],{'UNITS': 'km', 'LABLAXIS': 'altitude'}],
                        'lat':[np.array([71.5]), {'UNITS': 'deg', 'LABLAXIS': 'latitude'}],
                        'long': [np.array([15]), {'UNITS': 'deg', 'LABLAXIS': 'longitude'}],
                        'ap': [np.array([12]), {'UNITS': None, 'LABLAXIS': 'Ap Index'}],
                        }

    # store the data
    with open(file_path,'r') as file:
        lines = [line.rstrip().split() for line in file]
        values = np.array([[float(val) for val in arr] for arr in lines]).T

        data_dict_output['alt'][0] = values[0]
        data_dict_output['meridional_quiet'][0]= values[1]
        data_dict_output['zonal_quiet'][0]= values[2]
        data_dict_output['meridional_disturbed'][0]= values[3]
        data_dict_output['zonal_disturbed'][0]= values[4]
        data_dict_output['meridional_total'][0]= values[5]
        data_dict_output['zonal_total'][0]= values[6]

    # Convert everything to array
    for key in ['alt','meridional_quiet','zonal_quiet','meridional_disturbed','zonal_disturbed','meridional_total','zonal_total']:
        data_dict_output[key][0] = np.array(data_dict_output[key][0])

    # output the data
    import spaceToolsLib as stl
    stl.outputCDFdata(data_dict=data_dict_output, outputPath=r"C:\Data\physicsModels\ionosphere\neutral_environment\hwm14\ACESII_hwm14.cdf")




hwm14_txt_to_cdf()
