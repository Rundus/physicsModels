if plasmaToggles.useSchroeder_Te_Profile:
    # --- Ionosphere Temperature Profile ---
    # ASSUMES Ions and electrons have same temperature profile
    T0 = 2.5  # Temperature at the Ionospher (in eV)
    T1 = 0.0135  # (in eV)
    h0 = 2000 * m_to_km  # scale height (in meters)
    T_iono = T1 * np.exp(altRange / h0) + T0
    deltaZ = 0.3 * Re
    T_ps = 2000  # temperature of plasma sheet (in eV)
    # T_ps = 105  # temperature of plasma sheet (in eV)
    z_ps = 3.75 * Re  # height of plasma sheet (in meters)
    w = 0.5 * (1 - np.tanh((altRange - z_ps) / deltaZ))  # models the transition to the plasma sheet

    # determine the overall temperature profile
    T_e = np.array([T_iono[i] * w[i] + T_ps * (1 - w[i]) for i in range(len(altRange))])

if plasmaToggles.useSchroeder_Te_Profile:
    # --- Ionosphere Temperature Profile ---
    # ASSUMES Ions and electrons have same temperature profile
    T0 = 2.5  # Temperature at the Ionospher (in eV)
    T1 = 0.0135  # (in eV)
    h0 = 2000 * m_to_km  # scale height (in meters)
    T_iono = T1 * np.exp(altRange / h0) + T0
    deltaZ = 0.3 * Re
    T_ps = 2000  # temperature of plasma sheet (in eV)
    # T_ps = 105  # temperature of plasma sheet (in eV)
    z_ps = 3.75 * Re  # height of plasma sheet (in meters)
    w = 0.5 * (1 - np.tanh((altRange - z_ps) / deltaZ))  # models the transition to the plasma sheet

    # determine the overall temperature profile
    T_i = np.array([T_iono[i] * w[i] + T_ps * (1 - w[i]) for i in range(len(altRange))])

if plasmaToggles.useTanaka_ne_Profile:
    ##### TANAKA FIT #####
    # --- determine the density over all altitudes ---
    # Description: returns density for altitude "z [km]" in m^-3
    n0 = 24000000
    n1 = 2000000
    z0 = 600  # in km from E's surface
    h = 680  # in km from E's surface
    H = -0.74
    a = 0.0003656481654202569


    def fitFunc(x, n0, n1, z0, h, H, a):
        return a * (n0 * np.exp(-1 * (x - z0) / h) + n1 * (x ** (H)))


    ne_density = (cm_to_m ** 3) * np.array(
        [fitFunc(alt / m_to_km, n0, n1, z0, h, H, a) for alt in altRange])  # calculated density (in m^-3)

elif plasmaToggles.useKletzingS33_ne_Profile:
    #### KLETZING AND TORBERT MODEL ####
    # --- determine the density over all altitudes ---
    # Description: returns density for altitude "z [km]" in m^-3
    h = 0.06 * (Re / m_to_km)  # in km from E's surface
    n0 = 6E4
    n1 = 1.34E7
    z0 = 0.05 * (Re / m_to_km)  # in km from E's surface
    ne_density = (cm_to_m ** 3) * np.array(
        [(n0 * np.exp(-1 * ((alt / m_to_km) - z0) / h) + n1 * ((alt / m_to_km) ** (-1.55))) for alt in
         altRange])  # calculated density (in m^-3)

elif plasmaToggles.useChaston_ne_Profile:
    raise Exception('No Chaston Profile Available yet!')

if plasmaToggles.useSchroeder_Te_Profile:
    z_i = 2370 * m_to_km  #
    h_i = 1800 * m_to_km  # height of plasma sheet (in meters)
    n_Op = np.array(
        [data_dict['ne'][0][i] * 0.5 * (1 - np.tanh((altRange[i] - z_i) / h_i)) for i in range(len(altRange))])
    n_Hp = data_dict['ne'][0] - n_Op
    m_Op = IonMasses[1]
    m_Hp = IonMasses[2]
    ni_total = data_dict['ne'][0]
    m_eff_i = np.array(
        [m_Hp * 0.5 * (1 + np.tanh((altRange[i] - z_i) / h_i)) + m_Op * 0.5 * (1 - np.tanh((altRange[i] - z_i) / h_i))
         for i in range(len(altRange))])
    n_ions = [n_Op, n_Hp]