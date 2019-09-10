---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3 (v2)
    language: python
    name: python3-v2
---

# Figures for Instrument Paper

```python
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
from glob import glob
import pydfmux
import pickle
import pandas as pd
import os.path
```

## $1/f$ Noise Figure

```python
fr = list(core.G3File('gainmatching_noise_73124800.g3'))[1]
print(fr)
```

```python
def readout_noise(x, readout):
    return np.sqrt(readout)*np.ones(len(x))
def photon_noise(x, photon, tau):
    return np.sqrt(photon / (1 + 2*np.pi*((x*tau)**2)))
def atm_noise(x, A, alpha):
    return np.sqrt(A * (x)**(-1*alpha))
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout + (A * (x)**(-1*alpha)) + photon / (1 + 2*np.pi*((x*tau)**2)))
def horizon_model(x, readout, A, alpha):
    return np.sqrt(readout + (A * (x)**(-1*alpha)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout
def horizon_knee_func(x, readout, A, alpha):
    return (A * (x)**(-1*alpha)) - readout
```

```python
plt.figure(1, figsize=(10,4))

# subplot 1
ax1 = plt.subplot(1,2,1)

group = '150.0_w204'

ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
asd_diff = np.array(fr['AverageASDDiff'][group])
ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
asd_sum = np.array(fr['AverageASDSum'][group])

par_diff = fr["AverageASDDiffFitParams"][group]
par_sum = fr["AverageASDSumFitParams"][group]
plt.loglog(ff_diff, asd_diff, 'k', label='mean 150 GHz pair-difference')
plt.loglog(ff_sum, asd_sum, '0.6', label='mean 150 GHz pair-sum')
plt.loglog(ff_diff, noise_model(ff_diff, *list(par_diff)), 'C0--', label='noise model fit (see text)')

plt.xlabel('frequency [Hz]')
plt.ylabel('NET [$\mu$K$\sqrt{s}$]')
plt.legend(frameon=False)



# subplot 2
ax2 = plt.subplot(1,2,2)

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}
colors = {90.: 'C2', 150.: 'C3', 220.: 'C0'}
# colors = {90.: 'r', 150.: 'g', 220.: 'b'}

for jband, band in enumerate([90., 150., 220.]):
    A_sqrt = []
    alpha = []
    whitenoise = []
    noise_at_100mHz = []
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)

        par = fr["AverageASDDiffFitParams"][group]
        
        whitenoise.append(np.sqrt(par[0]))
        A_sqrt.append(np.sqrt(par[1]))
        alpha.append(par[2])
        
        noise_at_100mHz.append(atm_noise(0.1, par[1], par[2]))
        
    if band == 90.:
        bandname = 95.
    else:
        bandname = band
    _ = plt.hist(noise_at_100mHz, histtype='step', bins=np.linspace(0, 2000, 21),
                 linewidth=1.4, color=colors[band])
    _ = plt.hist(noise_at_100mHz, histtype='stepfilled', bins=np.linspace(0, 2000, 21),
                 label='{:.0f} GHz (median = {:.0f} $\mu$K$\sqrt{{s}}$)'\
                 .format(bandname, np.median(noise_at_100mHz)),
                 alpha=0.25, color=colors[band])
    plt.xlabel('residual atmospheric noise at 0.1 Hz\n($\sqrt{A (0.1 Hz)^{-\\alpha}}$) [$\mu$K$\sqrt{s}$]')
    plt.ylabel('wafer-averaged observations')
plt.legend(frameon=False)


plt.tight_layout()
plt.savefig('pair_differenced_noise.pdf')
```

```python
fig2 = plt.figure(2)

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}

for jband, band in enumerate([90., 150., 220.]):
    A_sqrt = []
    alpha = []
    whitenoise = []
    noise_at_100mHz = []
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)

        par = fr["AverageASDDiffFitParams"][group]
        
        whitenoise.append(np.sqrt(par[0]))
        A_sqrt.append(np.sqrt(par[1]))
        alpha.append(par[2])
        
        noise_at_100mHz.append(atm_noise(0.1, par[1], par[2]))
        
    _ = plt.hist(noise_at_100mHz, histtype='step', bins=np.linspace(0, 2000, 21),
                 label='{:.0f} GHz (median = {:.0f} $\mu$K$\sqrt{{s}}$)'.format(band, np.median(noise_at_100mHz)))
    plt.xlabel('residual atmospheric noise ($\sqrt{A (0.1 Hz)^{-\\alpha}}$) [$\mu$K$\sqrt{s}$]')
    plt.ylabel('wafer-averaged observations')
plt.legend()
```

```python
plt.figure(10)
ax1 = fig1.add_subplot(1,2,1)
ax1 = fig1.add_subplot(1,2,2)
```

```python
noise_fnames = glob('/spt/user/adama/20190329_gainmatching/downsampled/*.g3')
```

```python
noise_params = {}
for fname in noise_fnames:
    print(fname)
    fr = list(core.G3File(fname))[1]
    print(fname)
    
    for jband, band in enumerate([90., 150., 220.]):
        for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                        'w181', 'w188', 'w203', 'w204', 'w206']):
            group = '{:.1f}_{}'.format(band, wafer)
            noise_params[group] = fr["AverageASDDiffFitParams"][group]
```

## Noise Calculations
One table in the paper is a bunch of numbers related to noise. This section calculates some relevant numbers.

```python
from mapping_speed import tes_noise
```

```python
correlation = 1.0
phonon_gamma = 0.5

nu = {90:93.5e9, 150:146.9e9, 220:220.0e9}
delta_nu = {90:23.3e9, 150:30.7e9, 220:46.4e9}
P_optical = {90:(2.31e-12+2.67e-12), 150:(4.5e-12+3.28e-12), 220:(6.29e-12+3.7e-12)}
P_sat = {90:10e-12, 150:15e-12, 220:20e-12}
Tc = {90:0.450, 150:0.450, 220:0.450}
Tload = 4.0
Tbath = 0.315
Rsh = 0.03
Rn = 2.0
Rfrac = 0.8
R0 = Rfrac*Rn
loopgain = 10
```

```python
readout_johnson_noise_i = {90:10.4e-12, 150:13.0e-12, 220:16.0e-12}
shot_noise         = {band: tes_noise.shot_noise(nu=nu[band],
                                                 power=P_optical[band]) for band in nu}
correlation_noise  = {band: tes_noise.correlation_noise(nu=nu[band],
                                                        power=P_optical[band],
                                                        delta_nu=delta_nu[band],
                                                        correlation=correlation) for band in nu}
G                  = {band: tes_noise.G(Tc=Tc[band],
                                        Psat=P_sat[band],
                                        Popt=P_optical[band],
                                        Tbath=Tbath) for band in Tc}
v_bias             = {band: tes_noise.Vbias_rms(P_sat[band], P_optical[band], R0) for band in P_sat}
dIdP               = {band: tes_noise.dIdP(Vbias_rms=v_bias[band]) for band in v_bias}
phonon_noise       = {band: tes_noise.tes_phonon_noise_P(Tbolo=Tc[band],
                                                         G=G[band],
                                                         gamma=phonon_gamma) for band in Tc}
johnson_noise_p    = {band: tes_noise.tes_johnson_noise_P(f=0,
                                                          Tc=Tc[band],
                                                          Psat=P_sat[band],
                                                          L=loopgain,
                                                          Popt=P_optical[band]) for band in Tc}
johnson_noise_i    = {band: tes_noise.tes_johnson_noise_I(f=0,
                                                          Tc=Tc[band],
                                                          R0=R0,
                                                          L=loopgain) for band in Tc}
readout_johnson_noise_p = {band: readout_johnson_noise_i[band] / dIdP[band] for band in dIdP}
readout_noise_i    = {band: np.sqrt(readout_johnson_noise_i[band]**2 - johnson_noise_i[band]**2)
                      for band in Tc}
readout_noise_p    = {band: np.sqrt(readout_johnson_noise_p[band]**2 - johnson_noise_p[band]**2)
                      for band in Tc}
total_noise        = {band: np.sqrt(shot_noise[band]**2 + \
                                    correlation_noise[band]**2 + \
                                    phonon_noise[band]**2 + \
                                    readout_johnson_noise_p[band]**2) for band in shot_noise}
```

```python
for band in nu.keys():
    print('{} GHz:'.format(band))
    print('Vbias = {:.1f} uV'.format(v_bias[band]*1e6))
    print('Shot noise = {:.1f} aW/rtHz'.format(shot_noise[band]*1e18))
    print('Correlation noise = {:.1f} aW/rtHz'.format(correlation_noise[band]*1e18))
    print('Phonon noise = {:.1f} aW/rtHz'.format(phonon_noise[band]*1e18))
    print('Johnson noise (NEP) = {:.1f} aW/rtHz'.format(johnson_noise_p[band]*1e18))
    print('Johnson noise (NEI) = {:.1f} pA/rtHz'.format(johnson_noise_i[band]*1e12))
    print('readout + Johnson noise (NEP) = {:.1f} aW/rtHz'.format(readout_johnson_noise_p[band]*1e18))
    print('readout + Johnson noise (NEI) = {:.1f} pA/rtHz'.format(readout_johnson_noise_i[band]*1e12))
    print('readout noise (NEP) = {:.1f} aW/rtHz'.format(readout_noise_p[band]*1e18))
    print('readout noise (NEI) = {:.1f} pA/rtHz'.format(readout_noise_i[band]*1e12))
    print('total noise = {:.1f} aW/rtHz'.format(total_noise[band]*1e18))
    print()
```

## Yield
Let's add up some numbers from Donna's spreadsheet on the trac wiki:

https://spt-trac.grid.uchicago.edu/trac_south_pole/attachment/wiki/SummerJournal2018-2019/compare_all_wafers_to_last_year_19Nov2018.xlsx

```python
n_good_dets_by_wafer = {'w172':1437, 'w174':1433, 'w176':1480, 'w177':1530, 'w180':1405,
                        'w181':1367, 'w188':1241, 'w203':1456, 'w204':1409, 'w206':1408}
n_good_dets = np.sum([n_good_dets_by_wafer[wafer] for wafer in n_good_dets_by_wafer])
n_total_dets = 10*(66*24 - 6*2) # subtract off channels that go to alignment pixels
```

```python
hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm_pole = pydfmux.load_session(open(hwm_fname, 'r'))['hardware_map']
```

```python
bolos = hwm_pole.query(pydfmux.Bolometer)
freqs = [b.channel_map.lc_channel.frequency \
         for b in bolos \
         if b.channel_map.lc_channel.frequency > 1.e5]
n_resonances = len(freqs)
```

```python
tuned_fnames = []
for dirstub in ['201908']:
    fnames = glob('/big_scratch/pydfmux_output/{}*/'
                  '*drop_bolos_*/data/TOTAL_DATA.pkl'.format(dirstub))
    tuned_fnames += [fname for fname in fnames if 'tweak' not in fname]
```

```python
ntuned = {}
for fname in tuned_fnames:
    with open(fname, 'rb') as f:
        print(fname)
        ntuned[fname] = 0
        
        d = pickle.load(f)
        for mod in d.keys():
            if type(d[mod])==dict and 'results_summary' in d[mod].keys():
                ntuned[fname] += d[mod]['results_summary']['ntuned']
```

```python
n_tuned_median = np.median(list(ntuned.values()))
```

```python
# A huge number of detectors are removed because they don't show a clear transition
# in response to HR-10. These detectors are split between many different exclusion
# lists, so let's combine them all to remove duplicates.
exclude_path = '/home/adama/SPT/hardware_maps_southpole/2019/global/exclude'
exclude_fnames = ['warmVcold_exclude_hr10.csv', 'exclude_hr10_dkan_20181213.csv',
                  'exclude_suspecthr10_dkan_20181213.csv']
exclude_names = []
for fn in exclude_fnames:
    df = pd.read_csv(os.path.join(exclude_path, fn))
    exclude_names.extend(list(df['name']))
exclude_names = np.unique(exclude_names)
```

```python
print('Total number of detectors = {}'.format(n_total_dets))
print('Detectors with good warm pinout = {} ({:.1f}%)'\
      .format(n_good_dets, 100 * n_good_dets / n_total_dets))
print('Detectors with matched resonances = {} ({:.1f}%)'\
      .format(n_resonances, 100 * n_resonances / n_total_dets))
print('Detectors removed by HR-10 exclusion = {}'\
      .format(len(exclude_names)))
print('Tuned bolometers = {:.0f} ({:.1f}%)'\
      .format(n_tuned_median, 100 * n_tuned_median / n_total_dets))
```

Some other useful statistics:

* Number of full modules removed = 10 or 660 channels
 * These are a combination of noisy SQUIDs, and modules with shorts to ground that generated thermal heating on the wafer.

```python

```
