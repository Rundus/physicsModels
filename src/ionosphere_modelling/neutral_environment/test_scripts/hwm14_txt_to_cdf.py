# --- hwm14_txt_to_cdf.py ---
# --- Author: C. Feltman ---
# DESCRIPTION:



file_path = r'C:\Data\physicsModels\ionosphere\neutral_environment\hwm14_neutral_winds_20221120_052000.txt'

def hwm14_txt_to_cdf():

    with open(file_path) as file:
        for line in file:
            print(line)

    return



hwm14_txt_to_cdf()
