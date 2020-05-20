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
Rp=0.15
Rsh=0.03
loopgain=10

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
roffset=100.
rdynsq=700.
zsquid=700.
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
                                  r3d, r4d, r5d, roffset, rdynsq, zsquid,
                                  L_squid, Lstrip, current_share_factor,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, roffset, rdynsq, zsquid,
                                  L_squid, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, rtes)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, roffset, rdynsq, zsquid)
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
print()


# SCENARIO #1A: SPT-3G NOISE
Rn = 2.0
Rfrac = 0.75
Rbolo = Rn * Rfrac
nei_readout_white = np.array([10.4e-12, 16.0e-12])
nei_readout_0p1_hz = np.array([12.34e-12, 17.12e-12])

print('SCENARIO #1A:')
for name, nei_readout in {'white noise': nei_readout_white,
                          '0.1 Hz': nei_readout_0p1_hz}.items():
    print(name)
    min_nep = dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, Rbolo, Rp, Rsh, loopgain) * np.max(nei_readout)
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} aW/rtHz'.format(band,
                min_nep[jband]*1e18, max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz:& {:.2f} - {:.2f} \\\\'.format(band,
                        np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                        np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()


# SCENARIO #1B: SPT-3G MODEL NOISE, INCLUDING DEMOD FILTER EFFECT
# plot of filter factor on demod transfer function
plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq),
label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]', fontsize=14)
plt.title(' SPT-3G noise  model', fontsize=14)
plt.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig('nei_v_f_1b.pdf')


print('SCENARIO #1B:')
min_nei_2a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_2a = np.interp(5.2e6, f, i_n_currentsharing)
print('NEI at 3.0 MHz:')
print('carrier DAC = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_total_c*np.ones(f.shape))))
print('nuller DAC = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_total_n*np.ones(f.shape))))
print('demod chain = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_total_d*np.ones(f.shape)*\
                                                            amp_fac/calc_rc_fit(f, Zd=rdynsq))))
print('bias resistor = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_bias*np.ones(f.shape))))
print('SQUID = {:2f}'.format(np.interp(3.0e6, f, 1e12*i_squid*amp_fac*np.ones(f.shape))))
print()

# SCENARIO #2A: Holding all constant, reduce Rdyn from 700 to 350 Ohm
rdynsq_new = 350
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, roffset, rdynsq_new, zsquid,
                                  L_squid, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, rtes)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, roffset, rdynsq_new, zsquid)
i_bias = calc_bias_resistor_noise(rbias, rtes, Tbias)
i_squid = calc_squid_noise()

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid, 
                               Lstrip=Lstrip, rtes=rtes)

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq_new),
         label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #2A: reduce $R_{dyn}$ from 750 to 350 Ohm')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_2a.pdf')

print('SCENARIO #2A:')
min_nei_2a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_2a = np.interp(5.2e6, f, i_n_currentsharing)
print('NEI at 1.5 MHz = {:.2f}'.format(1e12*min_nei_2a))
print('NEI at 5.2 MHz = {:.2f}'.format(1e12*max_nei_2a))

for name, nei_readout in {'white noise': [min_nei_2a, max_nei_2a]}.items():
    print(name)
    min_nep = dPdI(Pelectrical, rtes, Rp, Rsh, loopgain) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, rtes, Rp, Rsh, loopgain) * np.max(nei_readout)
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18,
                                                       max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz:& {:.2f} - {:.2f} \\\\'.format(band,
                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()



# SCENARIO #2B: Holding all constant, reduce Rdyn from 700 to 350 Ohm,
# and reduce L_squid from 60nH to 30nH
rdynsq_new = 350
L_squid_new = 30e-9
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, rbias, rtes, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, roffset, rdynsq_new, zsquid,
                                  L_squid_new, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          rbias, rtes)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, roffset, rdynsq_new, zsquid)
i_bias = calc_bias_resistor_noise(rbias, rtes, Tbias)
i_squid = calc_squid_noise()

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid_new, 
                               Lstrip=Lstrip, rtes=rtes)

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq_new),
         label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #2B: reduce $R_{dyn}$ from $750 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_2b.pdf')

print('SCENARIO #2B:')
min_nei_2a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_2a = np.interp(5.2e6, f, i_n_currentsharing)
print('NEI at 1.5 MHz = {:.2f}'.format(1e12*min_nei_2a))
print('NEI at 5.2 MHz = {:.2f}'.format(1e12*max_nei_2a))

for name, nei_readout in {'white noise': [min_nei_2a, max_nei_2a]}.items():
    print(name)
    min_nep = dPdI(Pelectrical, rtes, Rp, Rsh, loopgain) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, rtes, Rp, Rsh, loopgain) * np.max(nei_readout)
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18,
                                                       max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz:& {:.2f} - {:.2f} \\'.format(band,
                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()



# SCENARIO #3A: Reduce Rbolo to 0.5 Ohm
# First plot as a function of Rbolo
Rbolo_new = 0.5
Rp_new = Rbolo_new / 10
Rbias_new = Rbolo_new / 50

rtes_plot = np.linspace(0.2, 2.0, 200)
freqs_plot = np.linspace(1.5e6, 5.5e6, 100)
r_vs_f_plot = np.zeros((len(rtes_plot), len(freqs_plot)))
for jr, r in enumerate(rtes_plot):
    f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                    wireharness, rbias, r, r5,
                                    r6, r1d, rf1d, r2d, rf2d,
                                    r3d, r4d, r5d, roffset, rdynsq, zsquid,
                                    L_squid, Lstrip, True,
                                    rtes_v_f, freqs, Tbias, add_excess,
                                    add_excess_demod)
    r_vs_f_plot[jr,:] = np.interp(freqs_plot, f, i_n_currentsharing) * \
                            dPdI(Pelectrical[3], r, Rp=0.05, Rsh=r/50, loopgain=loopgain)

plt.figure()
plt.imshow(np.sqrt(nep_else[3]**2 + r_vs_f_plot**2) / nep_else[3],
                   origin='lower', extent=[np.min(freqs_plot)*1e-6, np.max(freqs_plot)*1e-6,
                                           np.min(rtes_plot), np.max(rtes_plot)], aspect='auto')
cset1 = plt.contour(np.sqrt(nep_else[3]**2 + r_vs_f_plot**2) / nep_else[3],
                    levels=[1.1, 1.2, 1.3],
                    origin='lower', extent=[np.min(freqs_plot)*1e-6, np.max(freqs_plot)*1e-6,
                                            np.min(rtes_plot), np.max(rtes_plot)])
plt.clabel(cset1, [1.1, 1.2, 1.3])
plt.xlabel('bias frequency [MHz]')
plt.ylabel('bolometer R [Ohm]')
plt.title('Scenario #3A: reduce $R_{bolo}$\n'
          '$\sqrt{NEP_{else}^2 + NEP_{ro}^2} / NEP_{else}$')
plt.tight_layout()
plt.savefig('nep_v_f_v_r_3a.pdf')

# plots fixing Rbolo to 0.5 Ohm
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, Rbias_new, Rbolo_new, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, roffset, rdynsq, zsquid,
                                  L_squid, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          Rbias_new, Rbolo_new)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, roffset, rdynsq, zsquid)
i_bias = calc_bias_resistor_noise(Rbias_new, Rbolo_new, Tbias)
i_squid = calc_squid_noise()

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid, 
                               Lstrip=Lstrip, rtes=Rbolo_new)

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=rdynsq),
         label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #3A: reduce $R_{bolo}$ from $1.5 \\rightarrow 0.5$ Ohm')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_3a.pdf')

print('SCENARIO #3A:')
min_nei_3a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_3a = np.interp(5.2e6, f, i_n_currentsharing)
print('NEI at 1.5 MHz = {:.2f}'.format(1e12*min_nei_3a))
print('NEI at 5.2 MHz = {:.2f}'.format(1e12*max_nei_3a))

for name, nei_readout in {'white noise': [min_nei_3a, max_nei_3a]}.items():
    print(name)
    min_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * np.max(nei_readout)
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18,
                                                       max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz:& {:.2f} - {:.2f} \\\\'.format(band,
                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()


# SCENARIO #3B: Reduce Rbolo to 1.0 Ohm + make all SQUID changes
# In the S4 slides, we are now calling this "scenario #1"
# First plot as a function of Rbolo
Rbolo_new = 1.0
Rp_new = 0.25
Rbias_new = Rbolo_new / 50
Rdyn_new = 350
L_squid_new = 60e-9
squid_noise_new = 4.5e-12
zsquid_new = 700

rtes_plot = np.linspace(0.2, 2.0, 200)
freqs_plot = np.linspace(1.5e6, 5.2e6, 100)
r_vs_f_plot = np.zeros((len(rtes_plot), len(freqs_plot)))
for jr, r in enumerate(rtes_plot):
    f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                    wireharness, Rbias_new, r, r5,
                                    r6, r1d, rf1d, r2d, rf2d,
                                    r3d, r4d, r5d, roffset, Rdyn_new, zsquid_new,
                                    L_squid_new, Lstrip, True,
                                    rtes_v_f, freqs, Tbias, add_excess,
                                    add_excess_demod, squid_noise_new)
    r_vs_f_plot[jr,:] = np.interp(freqs_plot, f, i_n_currentsharing) * \
                            dPdI(Pelectrical[3], r, Rp=r/10, Rsh=r/50, loopgain=loopgain) 

plt.figure()
plt.imshow(np.sqrt(nep_else[3]**2 + r_vs_f_plot**2) / nep_else[3],
                   origin='lower', extent=[np.min(freqs_plot)*1e-6, np.max(freqs_plot)*1e-6,
                                           np.min(rtes_plot), np.max(rtes_plot)], aspect='auto')
cset1 = plt.contour(np.sqrt(nep_else[3]**2 + r_vs_f_plot**2) / nep_else[3],
                    levels=[1.05, 1.1, 1.15],
                    origin='lower', extent=[np.min(freqs_plot)*1e-6, np.max(freqs_plot)*1e-6,
                                            np.min(rtes_plot), np.max(rtes_plot)])
plt.clabel(cset1, [1.05, 1.1, 1.15])
plt.xlabel('bias frequency [MHz]')
plt.ylabel('bolometer R [Ohm]')
plt.title('Scenario #1: reduce $R_{bolo}$ from $1.5 \\rightarrow 1.0$ Ohm =\n'
          'reduce $R_{dyn}$ from $700 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH\n'
          '$\sqrt{NEP_{else}^2 + NEP_{ro}^2} / NEP_{else}$')
plt.tight_layout()
plt.savefig('nep_v_f_v_r_3b.pdf')


# plots fixing Rbolo to 1.0 Ohm
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, Rbias_new, Rbolo_new, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, roffset, Rdyn_new, zsquid_new,
                                  L_squid_new, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod, squid_noise_new)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          Rbias_new, Rbolo_new)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, roffset, Rdyn_new, zsquid_new)
i_bias = calc_bias_resistor_noise(Rbias_new, Rbolo_new, Tbias)
i_squid = calc_squid_noise(squid_noise_new)

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid_new, 
                               Lstrip=Lstrip, rtes=Rbolo_new)

plt.figure()
dPdI_plot = dPdI(Pelectrical[3], Rbolo_new, Rp_new, Rbias_new, loopgain)
plt.plot(f, 1e18*i_n_currentsharing * dPdI_plot, label='total (with current sharing)')
plt.plot(f, 1e18*i_total_c*np.ones(f.shape) * dPdI_plot, label='carrier DAC')
plt.plot(f, 1e18*i_total_n*np.ones(f.shape) * dPdI_plot, label='nuller DAC')
plt.plot(f, 1e18*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=Rdyn_new) * dPdI_plot,
         label='demod chain')
plt.plot(f, 1e18*i_bias*np.ones(f.shape) * dPdI_plot, label='bias resistor')
plt.plot(f, 1e18*i_squid*np.ones(f.shape)*amp_fac * dPdI_plot, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout NEP [aW/$\sqrt{Hz}$]')
plt.title('Scenario #1: reduce $R_{bolo}$ from $1.5 \\rightarrow 1.0$ Ohm =\n'
          'reduce $R_{dyn}$ from $700 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH')
plt.legend()
plt.tight_layout()
plt.savefig('nep_v_f_3b.pdf')

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=Rdyn_new),
         label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #1: reduce $R_{bolo}$ from $1.5 \\rightarrow 0.5$ Ohm =\n'
          'reduce $R_{dyn}$ from $700 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_3b.pdf')

print('SCENARIO #3B:')
min_nei_3a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_3a = np.interp(5.2e6, f, i_n_currentsharing)
mean_nei_3a = np.mean(i_n_currentsharing[(f>1.5e6) & (f<5.2e6)])
print('NEI at 1.5 MHz = {:.2f}'.format(1e12*min_nei_3a))
print('NEI at 5.2 MHz = {:.2f}'.format(1e12*max_nei_3a))
print('mean NEI = {:.2f}'.format(1e12*mean_nei_3a))

# DAC noise
mean_dac_nei_3b = np.mean(np.sqrt(i_total_c**2 + i_total_n**2))
print('mean DAC NEI = {:.2f}'.format(1e12*mean_dac_nei_3b))

low_f_ratio = nei_readout_0p1_hz[0] / nei_readout_white[0]
for name, nei_readout in {'white noise': [min_nei_3a, max_nei_3a, mean_nei_3a],
                          'low-f': [min_nei_3a*low_f_ratio, max_nei_3a*low_f_ratio, mean_nei_3a*low_f_ratio]}.items():
    print(name)
    min_nei = nei_readout[0]
    max_nei = nei_readout[1]
    mean_nei = nei_readout[2]
    min_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * min_nei
    max_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * max_nei
    mean_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * mean_nei
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18,
                                                                mean_nep[jband]*1e18,
                                                                max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz:& {:.2f} - {:.2f} - {:.2f} \\\\'.format(band,
                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(mean_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()


# SCENARIO #3C: Reduce Rbolo to 0.5 Ohm + make all SQUID changes
# First plot as a function of Rbolo
Rbolo_new = 0.5
Rp_new = Rbolo_new / 10
Rbias_new = Rbolo_new / 50
Rdyn_new = 350
L_squid_new = 30e-9
squid_noise_new=4.5e-12 / np.sqrt(2.) / 2.8
zsquid_new = 700

rtes_plot = np.linspace(0.2, 2.0, 200)
freqs_plot = np.linspace(1.5e6, 5.2e6, 100)
r_vs_f_plot = np.zeros((len(rtes_plot), len(freqs_plot)))
for jr, r in enumerate(rtes_plot):
    f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                    wireharness, Rbias_new, r, r5,
                                    r6, r1d, rf1d, r2d, rf2d,
                                    r3d, r4d, r5d, roffset, Rdyn_new, zsquid_new,
                                    L_squid_new, Lstrip, True,
                                    rtes_v_f, freqs, Tbias, add_excess,
                                    add_excess_demod, squid_noise_new)
    r_vs_f_plot[jr,:] = np.interp(freqs_plot, f, i_n_currentsharing) * \
                            dPdI(Pelectrical[3], r, Rp=r/10, Rsh=r/50, loopgain=loopgain) 

plt.figure()
plt.imshow(np.sqrt(nep_else[3]**2 + r_vs_f_plot**2) / nep_else[3],
                   origin='lower', extent=[np.min(freqs_plot)*1e-6, np.max(freqs_plot)*1e-6,
                                           np.min(rtes_plot), np.max(rtes_plot)], aspect='auto')
cset1 = plt.contour(np.sqrt(nep_else[3]**2 + r_vs_f_plot**2) / nep_else[3],
                    levels=[1.05, 1.1, 1.15],
                    origin='lower', extent=[np.min(freqs_plot)*1e-6, np.max(freqs_plot)*1e-6,
                                            np.min(rtes_plot), np.max(rtes_plot)])
plt.clabel(cset1, [1.05, 1.1, 1.15])
plt.xlabel('bias frequency [MHz]')
plt.ylabel('bolometer R [Ohm]')
plt.title('Scenario #2: reduce $R_{bolo}$ from $1.5 \\rightarrow 0.5$ Ohm =\n'
          'reduce $R_{dyn}$ from $750 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH\n'
          '$\sqrt{NEP_{else}^2 + NEP_{ro}^2} / NEP_{else}$')
plt.tight_layout()
plt.savefig('nep_v_f_v_r_3c.pdf')


# plots fixing Rbolo to 0.5 Ohm
f, i_n_currentsharing = calc_total_noise_current(rf1, rf2, rg, r2, r3, r4,
                                  wireharness, Rbias_new, Rbolo_new, r5,
                                  r6, r1d, rf1d, r2d, rf2d,
                                  r3d, r4d, r5d, roffset, Rdyn_new, zsquid_new,
                                  L_squid_new, Lstrip, True,
                                  rtes_v_f, freqs, Tbias, add_excess,
                                  add_excess_demod, squid_noise_new)
e_total_c, i_total_c = calc_carrier_noise(rf1, rf2, rg, r2,
                                          r3, r4, 0.,
                                          Rbias_new, Rbolo_new)
i_total_n = calc_nuller_noise(rf1, rf2, rg, r2, r3, r5, r6)
i_total_d = calc_demod_noise(r1d, rf1d, r2d, rf2d, r3d,
                             r4d, r5d, roffset, Rdyn_new, zsquid_new)
i_bias = calc_bias_resistor_noise(Rbias_new, Rbolo_new, Tbias)
i_squid = calc_squid_noise(squid_noise_new)

amp_fac = calc_current_sharing(freq=f, L_squid=L_squid_new, 
                               Lstrip=Lstrip, rtes=Rbolo_new)

plt.figure()
plt.plot(f, 1e12*i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, 1e12*i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, 1e12*i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, 1e12*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=Rdyn_new),
         label='demod chain')
plt.plot(f, 1e12*i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, 1e12*i_squid*np.ones(f.shape)*amp_fac, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout noise [pA/$\sqrt{Hz}$]')
plt.title('Scenario #2: reduce $R_{bolo}$ from $1.5 \\rightarrow 0.5$ Ohm =\n'
          'reduce $R_{dyn}$ from $750 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH')
plt.legend()
plt.tight_layout()
plt.savefig('nei_v_f_3c.pdf')

plt.figure()
dPdI_plot = dPdI(Pelectrical[3], Rbolo_new, Rp_new, Rbias_new, loopgain)
plt.plot(f, 1e18*i_n_currentsharing * dPdI_plot, label='total (with current sharing)')
plt.plot(f, 1e18*i_total_c*np.ones(f.shape) * dPdI_plot, label='carrier DAC')
plt.plot(f, 1e18*i_total_n*np.ones(f.shape) * dPdI_plot, label='nuller DAC')
plt.plot(f, 1e18*i_total_d*np.ones(f.shape)*amp_fac/calc_rc_fit(f, Zd=Rdyn_new) * dPdI_plot,
         label='demod chain')
plt.plot(f, 1e18*i_bias*np.ones(f.shape) * dPdI_plot, label='bias resistor')
plt.plot(f, 1e18*i_squid*np.ones(f.shape)*amp_fac * dPdI_plot, label='SQUID')
plt.xlabel('frequency [Hz]')
plt.ylabel('readout NEP [aW/$\sqrt{Hz}$]')
plt.title('Scenario #2: reduce $R_{bolo}$ from $1.5 \\rightarrow 0.5$ Ohm =\n'
          'reduce $R_{dyn}$ from $700 \\rightarrow 350$ Ohm +\n'
          'reduce $L_{SQUID}$ from $60 \\rightarrow 30$ nH')
plt.legend()
plt.tight_layout()
plt.savefig('nep_v_f_3c.pdf')

print('SCENARIO #3C:')
min_nei_3a = np.interp(1.5e6, f, i_n_currentsharing)
max_nei_3a = np.interp(5.2e6, f, i_n_currentsharing)
mean_nei_3a = np.mean(i_n_currentsharing[(f>1.5e6) & (f<5.2e6)])
print('NEI at 1.5 MHz = {:.2f}'.format(1e12*min_nei_3a))
print('NEI at 5.2 MHz = {:.2f}'.format(1e12*max_nei_3a))
print('mean NEI = {:.2f}'.format(1e12*mean_nei_3a))


# DAC noise
mean_dac_nei_3c = np.mean(np.sqrt(i_total_c**2 + i_total_n**2))
print('mean DAC NEI = {:.2f}'.format(1e12*mean_dac_nei_3c))


low_f_ratio = nei_readout_0p1_hz[0] / nei_readout_white[0]
for name, nei_readout in {'white noise': [min_nei_3a, max_nei_3a, mean_nei_3a],
                          'low-f': [min_nei_3a*low_f_ratio, max_nei_3a*low_f_ratio, mean_nei_3a*low_f_ratio]}.items():
    print(name)
    min_nei = nei_readout[0]
    max_nei = nei_readout[1]
    mean_nei = nei_readout[2]
    min_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * min_nei
    max_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * max_nei
    mean_nep = dPdI(Pelectrical, Rbolo_new, Rp_new, Rbias_new, loopgain) * mean_nei
    print('NEP:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz: {:.1f} - {:.1f} - {:.1f} aW/rtHz'.format(band, min_nep[jband]*1e18,
                                                                mean_nep[jband]*1e18,
                                                                max_nep[jband]*1e18))
    print()

    print('NEP_total / NEP_else:')
    for jband, band in enumerate(s4_bands):
        print('{} GHz:& {:.2f} - {:.2f} - {:.2f} \\\\'.format(band,
                    np.sqrt(min_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(mean_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband],
                    np.sqrt(max_nep[jband]**2 + nep_else[jband]**2) / nep_else[jband]))
    print()

plt.show()