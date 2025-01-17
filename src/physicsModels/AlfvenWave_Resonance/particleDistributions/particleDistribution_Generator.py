# --- imports ---
from numpy import array,sqrt,degrees,arccos,abs
from ACESII_code.Science.Simulations.TestParticle.simToggles import GenToggles,ptclToggles
from random import seed, choice
from numpy.random import uniform
from scipy.special import erfinv


def generateInitial_Data_Dict(varNames, Bmag, Bgrad, forceFunc, **kwargs):
    plotting = kwargs.get('showPlot', False)

    # --- generate initial data dict ---
    data_dict = {}
    data_dict = {**data_dict, **{f"{vNam}": [] for vNam in varNames}}
    # for each energy, create a variable to hold the initial data
    initVars = [[] for thing in varNames]

    # --- Populate Initial Variables ---
    for h, Z0_ptcl in enumerate(ptclToggles.Z0_ptcl_ranges): # loop over all initial starting altitudes and fill in initial data

        # --- create initial particle velocites ---
        a = ptclToggles.ptcl_mass / (2 * ptclToggles.ptclTemperature*ptclToggles.ptcl_charge)
        U = uniform(low=1E-20, high=1,size=ptclToggles.N_ptcls)
        Vels = erfinv(2 * U - 1) / sqrt(a) # use the inverse error function to fabricate a Maxwellian disribution
        seed(ptclToggles.seedChoice)
        Vperp = array([choice(Vels) for ptcl in range(ptclToggles.N_ptcls)]) # in meters
        Vpar = array([choice(Vels) for ptcl in range(ptclToggles.N_ptcls)]) # in meters
        V_mag = array([sqrt(Vperp[i]**2 + Vpar[i]**2) for i in range(len(Vpar))]) # in meters
        ptclEnergies = [0.5*ptclToggles.ptcl_mass*V_mag[i]*V_mag[i]/ptclToggles.ptcl_charge for i in range(len(V_mag))] # in eV

        # --- determine the initial pitch angles ---
        ptclPitches = []
        for ptcl in range(ptclToggles.N_ptcls):
            perpVal = Vperp[ptcl]
            parVal  = Vpar[ptcl]
            magnitudeVal = sqrt(perpVal**2 + parVal**2)
            ptchVal = degrees(arccos( parVal/ magnitudeVal))

            if parVal >= 0 and perpVal >= 0:
                ptchMod = 1
            elif parVal < 0 and perpVal >= 0:
                ptchMod = 1
            elif parVal > 0 and perpVal < 0:
                ptchMod = -1
            elif parVal < 0 and perpVal < 0:
                ptchMod = -1

            ptclPitches.append(ptchVal*ptchMod)

        # --- determine the color to assign each particle ---
        ptclColor = []
        for engy in ptclEnergies:
            withinRange = 0

            # check all the energy ranges
            for d, engyRange in enumerate(ptclToggles.simEnergyRanges):
                if engyRange[0] <= engy < engyRange[1]:
                    ptclColor.append(GenToggles.simColors[d])
                    withinRange = 1

            # in case the particle falls out of my specified ranges
            if withinRange == 0:
                ptclColor.append('black')


        # --- fill in the initial data ---
        initVars[0].append([Z0_ptcl for ptcl in range(ptclToggles.N_ptcls)]) # initZpos
        initVars[1].append(Vpar) # initVpar
        initVars[2].append(Vperp) # initVperp
        initVars[3].append(ptclEnergies) # initEngy
        initVars[4].append([Bmag[abs(GenToggles.simAlt-zpos).argmin()] for zpos in initVars[0][h]]) # initBmag
        initVars[5].append([Bgrad[abs(GenToggles.simAlt-zpos).argmin()] for zpos in initVars[0][h]]) # initGrad
        initVars[6].append([0.5*ptclToggles.ptcl_mass*(initVars[2][h][k]**2)/initVars[4][h][k] for k in range(ptclToggles.N_ptcls)]) # initMoment
        initVars[7].append([forceFunc(simTime_Index=0, alt_Index=abs(GenToggles.simAlt - initVars[0][h][k]).argmin(), mu=initVars[6][h][k], deltaB=initVars[5][h][k]) for k in range(ptclToggles.N_ptcls)]) # initForce
        initVars[8].append([1 if initVars[0][h][k] <= GenToggles.obsHeight else 0 for k in range(ptclToggles.N_ptcls)]) # initObserved
        initVars[9].append(ptclColor)
        initVars[10].append(ptclPitches)


    # collapse the dimension of the initial variables and store them
    for j in range(len(initVars)):
        data_dict[f'{varNames[j]}'].append([item for sublist in initVars[j] for item in sublist])



    # --- plotting ---
    # if plotBool:
    #     # Show the Energies
    #     vthermal = np.sqrt(8 * q0 * ptclTemperature / m_e)
    #     fig, ax = plt.subplots()
    #     ax.scatter(data_dict['vperp'][0] / vthermal, data_dict['vpar'][0] / vthermal, color=data_dict['color'][0])
    #     ax.set_ylabel('$V_{\parallel}$')
    #     ax.set_xlabel('$V_{\perp}$')
    #     patches = [mpatches.Patch(color=simColors[i][0], label=f'<{simEnergyRanges[i][1]} eV') for i in
    #                range(len(simEnergyRanges))]
    #     patches.append(mpatches.Patch(color='black', label=f'outside range'))
    #     ax.legend(handles=patches)
    #     plt.show()


    return data_dict




