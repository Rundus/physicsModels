
########################
# --- PLASMA DENSITY ---
########################
class PlasmaToggles:

    ##################
    # --- FILE I/O ---
    ##################
    from src.ionosphere_modelling.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_root_path}\plasma_environment'
    IRI_filePath = r'C:\Data\physicsModels\ionosphere\IRI\ACESII\IRI_3D_2022324.cdf'

    # --- --- --- ---
    ### ELECTRONS ###
    # --- --- --- ---
    wBackground_density = 0

    match wBackground_density:
        case 0:
            useACESII_density_Profile = True
            useEISCAT_density_Profile = False
        case 1:
            useACESII_density_Profile = False
            useEISCAT_density_Profile = True

    # --- --- --
    ### IONS ###
    # --- --- --
    wIons = ['NO+', 'O+', 'O2+']  # which neutrals to consider in the simulation, use the key format in spaceToolsLib
    # wIons = ['NO+','H+','N+','He+', 'O+', 'O2+']  # all ions



