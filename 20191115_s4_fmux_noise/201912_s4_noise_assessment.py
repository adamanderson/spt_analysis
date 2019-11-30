from noise_calculator import *
import matplotlib.pyplot as plt

def dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain):
    v_bias_rms = np.sqrt(Pelectrical * Rbolo)
    RL = Rp + Rsh
    return v_bias_rms / np.sqrt(2.) * (Rbolo + RL + loopgain*(Rbolo-RL)) / (Rbolo*loopgain)

# CMB-S4 detector configuration
s4_bands = [30, 40, 85, 145, 95, 155, 220, 270]
Pelectrical = np.array([0.8, 1.7, 3.3, 4.7, 3.0, 5.0, 9.5, 13.1]) * 1e-12
nep_else = np.array([9.8, 16.2, 27.4, 33.7, 25.9, 35.4, 60.5, 76]) * 1e-18
Rp=0.3
Rsh=0.03
loopgain=50

# Readout circuit parameters
rf1=300.
rf2=200.
rg=300.
r2=250.
r3=50.
r4=20.
wireharness=0.
rbias=0.03
rtes=1.5
r5=100.
r6=750.
r1d=10.
rf1d=150.
r2d=100.
rf2d=400.
r3d=50.
r4d=42.
r5d=100.
rdynsq=750.
zsquid=750.
L_squid=60e-9
Lstrip=46e-9
current_share_factor=False
rtes_v_f=False
freqs=[0.]
Tbias=4.
add_excess=[]
add_excess_demod=[]

f, i_n = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, rdynsq, zsquid,
                                  L_squid, Lstrip, current_share_factor,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, rdynsq, zsquid,
                                  L_squid, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, rtes)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, rdynsq, zsquid)
i_bias = calc_bias_resistor_noise(rbias, rtes, Tbias)
i_squid = calc_squid_noise()

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid, 
                               Lstrip=Lstrip, rtes=rtes)

e_total_c, i_total_c_1p5_ohm = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, 1.5)
e_total_c, i_total_c_2_ohm = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, 2.0)
print('carrier DAC noise for 1.5 Ohm bolo = {:.2E}'.format(i_total_c_1p5_ohm))
print('carrier DAC noise for 2.0 Ohm bolo = {:.2E}'.format(i_total_c_2_ohm))

i_bias_1p5_ohm = calc_bias_resistor_noise(rbias, 1.5, Tbias)
i_bias_2_ohm = calc_bias_resistor_noise(rbias, 2.0, Tbias)
print('bias resistor noise for 1.5 Ohm bolo = {:.2E}'.format(i_bias_1p5_ohm))
print('bias resistor noise for 2.0 Ohm bolo = {:.2E}'.format(i_bias_2_ohm))



# SCENARIO #1A: SPT-3G NOISE
Rn = 2.0
Rfrac = 0.75
Rbolo = Rn * Rfrac
nei_readout_white = np.array([10.4e-12, 16.0e-12])
nei_readout_0p1_hz = np.array([12.34e-12, 17.12e-12])

for name, nei_readout in {'white noise': nei_readout_white,
                          '0.1 Hz': nei_readout_0p1_hz}.items():
    print(name)
    min_nep = dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain) * np.max(nei_readout)
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18, max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.2f} - {:.2f}'.format(band,
                                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()


# SCENARIO #1B: SPT-3G MODEL NOISE, INCLUDING DEMOD FILTER EFFECT
# plot of filter factor on demod transfer function
plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq), label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #1B: SPT-3G noise in model')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_1b.png', dpi=200)

plt.figure()
plt.plot(f, 1./calc_rc_fit(f, Zd=rdynsq))
plt.xlabel('frequency [Hz]')
plt.ylabel('demod transfer function')
plt.tight_layout()

print('SCENARIO #1B:')
min_nei_2a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_2a = np.interp(5.2e6, f, i_n_currentsharing)
print('NEI at 3.0 MHz:')
print('carrier DAC = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_total_c*np.ones(f.shape))))
print('nuller DAC = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_total_n*np.ones(f.shape))))
print('demod chain = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq))))
print('bias resistor = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_bias*np.ones(f.shape))))
print('SQUID = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_squid*np.ones(f.shape)*amp_fac)))
print()


# SCENARIO #2A: Holding all constant, reduce Rdyn from 750 to 350 Ohm
rdynsq_new=350

f, i_n = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, rdynsq_new, zsquid,
                                  L_squid, Lstrip, current_share_factor,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, rdynsq_new, zsquid,
                                  L_squid, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, rtes)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, rdynsq_new, zsquid)
i_bias = calc_bias_resistor_noise(rbias, rtes, Tbias)
i_squid = calc_squid_noise()

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid, 
                               Lstrip=Lstrip, rtes=rtes)

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq_new), label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #2A: Reduce Rdyn from 750 to 350 Ohm')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_2a.png', dpi=200)

print('SCENARIO #2A:')
min_nei_2a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_2a = np.interp(5.2e6, f, i_n_currentsharing)
print('NEI at 1.5 MHz = {:.2f}'.format(1e12*min_nei_2a))
print('NEI at 5.2 MHz = {:.2f}'.format(1e12*max_nei_2a))


for name, nei_readout in {'white noise': [min_nei_2a, max_nei_2a*0.8]}.items():
    print(name)
    min_nep = dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain) * np.max(nei_readout)
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18, max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.2f} - {:.2f}'.format(band,
                                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()





# SCENARIO #2B: Holding everything constant, improve the demod TF by a factor
# of 2x
demod_tf_factor = 1.5

e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, rtes)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, rdynsq, zsquid)
i_bias = calc_bias_resistor_noise(rbias, rtes, Tbias)
i_squid = calc_squid_noise()

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid, 
                               Lstrip=Lstrip, rtes=rtes)
demod_tf = 1 - (1-calc_rc_fit(f, Zd=rdynsq)) / demod_tf_factor

i_n_total = np.sqrt((i_total_c*np.ones(f.shape))**2. + \
                    (i_total_n*np.ones(f.shape))**2. + \
                    (i_total_d*np.ones(f.shape)*amp_fac/demod_tf)**2. + \
                    (i_bias*np.ones(f.shape))**2. + \
                    (i_squid*np.ones(f.shape)*amp_fac)**2.)

plt.figure()
plt.plot(f, 1./demod_tf)
plt.xlabel('frequency [Hz]')
plt.ylabel('demod transfer function')
plt.tight_layout()

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/demod_tf, label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #2B: Improve demod TF by 2x')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_2b.png', dpi=200)

# plt.show()