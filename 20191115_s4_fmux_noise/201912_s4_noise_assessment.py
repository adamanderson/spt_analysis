from noise_calculator import *
import matplotlib.pyplot as plt

def dPdI(Pelectrical, Rbolo):
    v_bias_rms = np.sqrt(Pelectrical * Rbolo)
    return v_bias_rms / np.sqrt(2.)

# CMB-S4 detector configuration
s4_bands = [30, 40, 85, 145, 95, 155, 220, 270]
Pelectrical = np.array([0.8, 1.7, 3.3, 4.7, 3.0, 5.0, 9.5, 13.1]) * 1e-12
nep_else = np.array([9.8, 16.2, 27.4, 33.7, 25.9, 35.4, 60.5, 76]) * 1e-18

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
rdynsq=500.
zsquid=500.
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

plt.plot(f, i_n_currentsharing, label='total (with current sharing)')
plt.plot(f, i_total_c*np.ones(f.shape), label='carrier DAC')
plt.plot(f, i_total_n*np.ones(f.shape), label='nuller DAC')
plt.plot(f, i_total_d*np.ones(f.shape)*amp_fac, label='demod chain')
plt.plot(f, i_bias*np.ones(f.shape), label='bias resistor')
plt.plot(f, i_squid*np.ones(f.shape), label='SQUID')
plt.legend()

# SCENARIO #1: SPT-3G NOISE
Rn = 2.0
Rfrac = 0.75
Rbolo = Rn * Rfrac
nei_readout_white = np.array([10.4e-12, 16.0e-12])
nei_readout_0p1_hz = np.array([12.34e-12, 17.12e-12])


for name, nei_readout in {'white noise': nei_readout_white,
                          '0.1 Hz': nei_readout_0p1_hz}.items():
    print(name)
    min_nep = dPdI(Pelectrical, Rbolo) * np.min(nei_readout)
    max_nep = dPdI(Pelectrical, Rbolo) * np.max(nei_readout)
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
