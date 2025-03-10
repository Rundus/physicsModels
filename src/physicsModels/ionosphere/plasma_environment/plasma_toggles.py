
########################
# --- PLASMA DENSITY ---
########################
class plasmaToggles:
    ##################
    # --- FILE I/O ---
    ##################
    outputFolder = 'C:\Data\physicsModels\ionosphere\plasma_environment'
    IRI_filePath = r'C:\Data\physicsModels\ionosphere\IRI\ACESII\IRI_3D_2022324.cdf'

    # --- --- --- ---
    ### ELECTRONS ###
    # --- --- --- ---
    useACESII_ne_Profile = False

    # --- --- --
    ### IONS ###
    # --- --- --
    wIons = ['NO+', 'O+', 'O2+']  # which neutrals to consider in the simulation, use the key format in spaceToolsLib
    # wIons = ['NO+','H+','N+','He+', 'O+', 'O2+']  # all ions

    useACESII_ni_Profile = False


