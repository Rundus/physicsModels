#############################
# --- Diagnostic Plotting ---
#############################

# --- Toggles ---
plot_Evans1974Curves = True
plot_Example_BackScatterCurve = False
plot_Example_totalEnergyFlux = False

if plot_Evans1974Curves or plot_Example_BackScatterCurve or plot_Example_totalEnergyFlux:
    import matplotlib.pyplot as plt

if plot_Evans1974Curves:

    Label_fontsize = 25
    Title_FontSize = 40
    tick_LabelSize = 60
    tick_SubplotLabelSize = 15
    tick_Width = 2
    tick_Length = 4
    plot_LineWidth = 3

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(12, 8)
    xData, yData = zip(*sorted(zip(Energy_DegradedPrimary,NFlux_up_PeriE_DegradedPrimary)))
    ax[0].set_title('Primaries Backscatter',fontsize=Title_FontSize)
    ax[0].plot(xData, yData,color='black', linewidth=plot_LineWidth)
    ax[0].set_yscale('log')
    ax[0].set_ylim(1E-10,1E-3)
    ax[0].set_xscale('log')
    ax[0].set_xlim(1E-2, 1)
    ax[0].set_ylabel('Upgoing Flux per Incident Electron\n $[E(Incident)/10000 ] \cdot [cm^{-2}sec^{-2}eV^{-1}$]', fontsize=Label_fontsize)
    ax[0].set_xlabel('E(Backscatter)/E(Incident)', fontsize=Label_fontsize)

    xData, yData = zip(*sorted(zip(Energy_secondaries, NFlux_up_PeriE_secondaries)))
    ax[1].set_title('Secondaries',fontsize=Title_FontSize)
    ax[1].plot(xData, yData, color='black', linewidth=plot_LineWidth)
    ax[1].set_yscale('log')
    ax[1].set_ylim(1E-8, 1E-1)
    ax[1].set_xscale('log')
    ax[1].set_xlim(1E1, 1E3)
    ax[1].set_ylabel('Upgoing Flux per Incident Electron \n $[cm^{-2}sec^{-2}eV^{-1}$]', fontsize=Label_fontsize)
    ax[1].set_xlabel('Energy (eV)', fontsize=Label_fontsize)

    for i in range(2):
        ax[i].tick_params(axis='y', which='major', labelsize=tick_SubplotLabelSize + 4, width=tick_Width,
                               length=tick_Length)
        ax[i].tick_params(axis='y', which='minor', labelsize=tick_SubplotLabelSize, width=tick_Width,
                               length=tick_Length / 2)
        ax[i].tick_params(axis='x', which='major', labelsize=tick_SubplotLabelSize, width=tick_Width,
                               length=tick_Length)
        ax[i].tick_params(axis='x', which='minor', labelsize=tick_SubplotLabelSize, width=tick_Width,
                               length=tick_Length / 2)

    plt.tight_layout()
    plt.show()

if plot_Example_BackScatterCurve:
    Eincdient_test = 610
    Erange = np.linspace(0.01*Eincdient_test, Eincdient_test, 1000)
    test_backscatterCurve = generate_DegradedPrimaryCurve(Eincdient_test)

    fig, ax = plt.subplots()
    ax.plot(Erange, test_backscatterCurve(Erange), label=f'{Eincdient_test} eV')
    ax.set_yscale('log')
    ax.set_ylabel('Flux (Upgoing) $cm^{-2}sec^{-1}eV^{-1}$ per incident electron')
    ax.set_xscale('log')
    ax.set_xlabel('Energy (eV)')
    ax.legend()
    plt.show()

if plot_Example_totalEnergyFlux:

    # Determine a common energy range to evaluate everything onto
    xData = np.linspace(10,11000,2000)

    # Generate the splines
    sCurve_spline = generate_SecondariesCurve()
    bCurve_splines = [generate_DegradedPrimaryCurve(engy) for engy in totalEnergyFlux_testEnergies]

    sCurveValues = []
    dpCurveValues = []

    for idxTestEnergy,testEnergy in enumerate(totalEnergyFlux_testEnergies):

        # evalulate the curves in their respective regions and store them into two variables
        sCurveData = np.zeros(shape=(len(xData)))
        dpCurveData = np.zeros(shape=(len(xData)))

        for idx, engy in enumerate(xData):

            if engy <= 1000 and engy <= testEnergy:
                sCurveData[idx] = sCurve_spline(engy)

            if 0.01*testEnergy <= engy <= 1*testEnergy:
                dpCurveData[idx] = bCurve_splines[idxTestEnergy](engy)

        sCurveValues.append(sCurveData)
        dpCurveValues.append(dpCurveData)


    fig,ax = plt.subplots()
    fig.set_size_inches(10, 10)
    for curvesetIdx in range(len(dpCurveValues)):
        ax.plot(xData,sCurveValues[curvesetIdx]+dpCurveValues[curvesetIdx],label=f'{totalEnergyFlux_testEnergies[curvesetIdx]} eV')

    ax.set_yscale('log')
    ax.set_ylabel('Flux (Upgoing) $cm^{-2}sec^{-1}eV^{-1}$ per incident electron')
    ax.set_xscale('log')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylim(1E-8,1E-2)
    ax.set_xlim(10,20000)
    ax.legend()
    plt.show()