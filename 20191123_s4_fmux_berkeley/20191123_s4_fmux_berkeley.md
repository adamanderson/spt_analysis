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

# Plots for Berkeley Readout Assessment
For the sake of traceability, let's collate all the data and calculations for the readout review in this jupyter notebook.

```python
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
from spt3g.dfmux import HousekeepingForBolo
from spt3g.dfmux.unittransforms import bolo_bias_voltage_rms
```

## Horizon Noise


### Voltage bias
Let's extract the voltage bias from the housekeeping data. We will use the average voltage bias (or some corrected version thereof) to convert NEI into NEP.

```python
dhk = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/0000.g3'))
d = list(core.G3File('horizon_noise_77863968_bender_ltd_perbolo_only.g3'))
dcal = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/offline_calibration.g3'))
bps = dcal[0]["BolometerProperties"]
vbias_rms = {90:[], 150:[], 220:[]}
bias_freqs = {90:[], 150:[], 220:[]}

for jbolo, bolo in enumerate(d[1]["ASDFitParams"].keys()):
    boardhk, mezzhk, modhk, chanhk = HousekeepingForBolo(dhk[3]["DfMuxHousekeeping"],
                                                         dhk[2]["WiringMap"], bolo, True)
    
    v = bolo_bias_voltage_rms(dhk[2]["WiringMap"], dhk[3]["DfMuxHousekeeping"],
                              bolo, "ICE", tf='spt3g_filtering_2017_full')
    if bps[bolo].band / core.G3Units.GHz in vbias_rms.keys():
        vbias_rms[bps[bolo].band / core.G3Units.GHz].append(v / (1e-6 * core.G3Units.volt))
        bias_freqs[bps[bolo].band / core.G3Units.GHz].append(chanhk.carrier_frequency / core.G3Units.Hz)
```

```python
Rsh = 0.03
Lsh = 0.9e-9
Rtotal = 2.0*0.8
Rp = 0.25

def bias_correction(f):
    return np.sqrt(Rsh**2 + (2*np.pi * f * Lsh)**2) / \
                   Rsh * (Rtotal - Rp) / Rtotal
```

```python
for band in vbias_rms.keys():
    plt.hist(np.array(vbias_rms[band]) * \
             bias_correction(np.array(bias_freqs[band])),
             bins=np.linspace(1,6,101),
             histtype='step', label='{:d} GHz'.format(band))
plt.legend()
```

```python
for band in vbias_rms.keys():
    bias_factor = bias_correction(np.array(bias_freqs[band]))
    plt.plot(bias_freqs[band], np.array(vbias_rms[band]) * bias_factor, 
             '.', label='{:d} GHz'.format(band))
plt.legend()
plt.xlim([1.5e6, 5.5e6])
plt.xlabel('bias frequency [Hz]')
plt.ylabel('$V_{bias}^{rms}$ [$\mu$V]')
plt.tight_layout()

plt.figure()
fplot = np.linspace(1.5e6, 5.2e6)
plt.plot(fplot, bias_correction(fplot))
plt.xlabel('bias frequency [Hz]')
plt.ylabel('bias voltage correction factor')
plt.tight_layout()
```

There is a lot of variation in the bias voltages because of wafer-to-wafer variation and unmodeled parasitics in the 220 GHz, but let's calculate the median RMS bias voltages anyway as a benchmark for converting between NEI and NEP scales in the readout noise figures below for the readout assessment.

```python
median_vbias_rms = {}

print('rms Vbias:')
for band in vbias_rms.keys():
    bias_factor = bias_correction(np.array(bias_freqs[band]))
    v = np.array(vbias_rms[band]) * bias_factor * 1e-6
    median_vbias_rms[band] = np.median(v[(v>2e-6) & (v<8e-6)])
    print('{} GHz : {:.2f} uV (rms)'.format(band, median_vbias_rms[band] * 1e6))
```

### PSDs ("Noise Performance I (Existing)")

```python
fr = list(core.G3File('horizon_noise_77863968_bender_ltd.g3'))[1]

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}
band_labels = {90:'95', 150:'150', 220:'220'}
band_freqs = {90:'[1.6 - 2.3 MHz]', 150:'[2.3 - 3.5 MHz]', 220:'[3.5 - 5.2 MHz]'}
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

def nei2nep(nei, vbias_rms):
    dPdI = vbias_rms / np.sqrt(2)
    return nei * dPdI
```

```python
median_vbias_rms[band]
```

```python
nei_min = 5
nei_max = 1000

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8,8))
fig.subplots_adjust(wspace=0)
for jband, band in enumerate([90., 150., 220.]):

    group = '{:.1f}_w180'.format(band)

    ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
    ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
    asd_diff = np.array(fr['AverageASDDiff'][group]) / np.sqrt(2.)
    asd_sum = np.array(fr['AverageASDSum'][group]) / np.sqrt(2.)

    par = fr["AverageASDDiffFitParams"][group]
    ax[jband].loglog(ff_sum[ff_sum<75], asd_sum[ff_sum<75],
                     label='pair sum (measured)', color='0.6')
    ax[jband].loglog(ff_diff[ff_diff<75], asd_diff[ff_diff<75],
                     label='pair difference (measured)', color='k')
    ax[jband].loglog(ff_sum, atm_noise(ff_sum, par[1], par[2]) / np.sqrt(2.),
                     'C0--', label='low-frequency noise')
    ax[jband].loglog(ff_sum, readout_noise(ff_sum, par[0]) / np.sqrt(2.),
                     'C2--', label='white noise')
    ax[jband].loglog(ff_sum, horizon_model(ff_sum, *list(par)) / np.sqrt(2.),
                     'C3--', label='total noise model')
    
    ax[jband].set_ylabel('current noise [pA/$\sqrt{Hz}$]', fontsize=14)
    ax[jband].grid()
    ax[jband].set_ylim([nei_min, nei_max])
    ax[jband].set_xlim([0.003, 100])
    ax[jband].annotate('{} GHz ($f_{{bias}} \in ${})'.format(band_labels[band],
                                                       band_freqs[band]),
                 (0.01,500), fontsize=14)
    ax[jband].tick_params(axis='both', labelsize=14)

    ax2 = ax[jband].twinx()
    ax2.set_ylabel('NEP [aW/$\sqrt{Hz}$]', fontsize=14)
    ax2.set_ylim([nei2nep(nei_min*1e-12, median_vbias_rms[band]) * 1e18,
                  nei2nep(nei_max*1e-12, median_vbias_rms[band]) * 1e18])
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=14)
    
ax[2].legend(fontsize=12)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
ax[2].set_xlabel('frequency [Hz]', fontsize=14)

plt.savefig('w180_horizon_noise_ltd.png', dpi=150)
```

### Histograms


#### Flat readout component

```python
dcal = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/'
                        'offline_calibration.g3'))
bps = dcal[0]["BolometerProperties"]
d = list(core.G3File('horizon_noise_77863968_bender_ltd_perbolo_only.g3'))
noise = np.array([np.sqrt(d[1]["ASDFitParams"][bolo][0])
                  for bolo in d[1]["ASDFitParams"].keys()
                  if len(d[1]["ASDFitParams"][bolo])==3])
bolo_band = np.array([bps[bolo].band / core.G3Units.GHz
                      for bolo in d[1]["ASDFitParams"].keys()
                      if len(d[1]["ASDFitParams"][bolo])==3])
```

```python
plt.figure(1)
band_labels = {90:95, 150:150, 220:220}
for band in [90, 150, 220]:
    _ = plt.hist(noise[np.isfinite(noise) & (noise>1) & (bolo_band==band)],
                 bins=np.arange(0,35,1), alpha=0.5,
                 label='{} GHz'.format(band_labels[band]))
    print(np.median(noise[np.isfinite(noise) & (noise>1) & (bolo_band==band)]))
plt.xlim([0, 35])
plt.xlabel('current noise (pA/$\sqrt{Hz}$)', fontsize=14)
plt.ylabel('readout channels', fontsize=14)
plt.gca().tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid()
plt.savefig('NEI_no_photons.png', dpi=200)
```

#### Histograms as a function of bias frequency ("Noise Performance II (Existing)")

```python
print(fr)
```

```python
dcal = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/'
                        'offline_calibration.g3'))
bps = dcal[0]["BolometerProperties"]
d = list(core.G3File('horizon_noise_77863968_bender_ltd_perbolo_only.g3'))

f_ranges = [(0.01,0.1), (0.1,1), (10,20)]
avg_nei = {}
for f_range in f_ranges:
    print(f_range)
    avg_nei[f_range] = {90:[], 150:[], 220:[]}
    
    for bolo in d[1]["ASDFitParams"].keys():
        if bps[bolo].band / core.G3Units.GHz in avg_nei[f_range]:
            par = d[1]["ASDFitParams"][bolo]
            if len(par) == 3:
                f_points = np.linspace(f_range[0], f_range[1])
                avg_nei[f_range][bps[bolo].band / core.G3Units.GHz]\
                    .append(np.mean(horizon_model(f_points, *list(par))))
```

```python
d2 = list(core.G3File('/home/adama/SPT/spt_analysis/20190329_gainmatching/horizon_noise_77863968.g3'))
# print(d2[1])
for fr in d2:
    print(fr)
```

```python
nei_min = 0
nei_max = 45

fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(8,8))
fig.subplots_adjust(hspace=0)
for jrange, f_range in enumerate(f_ranges):
    for jband, band in enumerate(avg_nei[f_range].keys()):
        ax[jrange].grid()
        ax[jrange].hist(avg_nei[f_range][band], bins=np.arange(0.01,45,1),
                        alpha=0.5, color='C{}'.format(jband), label='{} GHz'.format(band))
        ax[jrange].hist(avg_nei[f_range][band], bins=np.arange(0.01,45,1),
                        color='C{}'.format(jband), histtype='step')
        ax[jrange].tick_params(axis='both', labelsize=14)
        ax[jrange].set_ylabel('bolometers', fontsize=14)
    ax[jrange].annotate('{} - {} Hz'.format(f_range[0], f_range[1]),
                        (0.03, 0.8), fontsize=16, xycoords='axes fraction')

ax[2].set_xlim([nei_min, nei_max])
ax[2].set_xlabel('current noise [pA/$\sqrt{Hz}$]', fontsize=14)
ax[0].legend(fontsize=14)

# ax2 = ax[0].twiny()
# ax2.set_xlabel('NEP (150 GHz) [aW/$\sqrt{Hz}$]', fontsize=14)
# ax2.set_xlim([nei2nep(nei_min*1e-12, median_vbias_rms[150]) * 1e18,
#               nei2nep(nei_max*1e-12, median_vbias_rms[150]) * 1e18])
# ax2.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('horizon_noise_histograms.png', dpi=150)
```

```python
n = np.array(avg_nei[f_range][90])
len(n[n>0])
```

### Readout noise at 0.1 Hz ("Low-frequency Noise Performance (Projected)")

```python
dcal = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/'
                        'offline_calibration.g3'))
bps = dcal[0]["BolometerProperties"]
d = list(core.G3File('horizon_noise_77863968_bender_ltd_perbolo_only.g3'))

nei_0p1_hz = {90:[], 150:[], 220:[]}
for bolo in d[1]["ASDFitParams"].keys():
    if bps[bolo].band / core.G3Units.GHz in avg_nei[f_range]:
        par = d[1]["ASDFitParams"][bolo]
        if len(par) == 3:
            nei_0p1_hz[bps[bolo].band / core.G3Units.GHz]\
                .append(horizon_model(0.1, *list(par)))
```

```python
plt.figure(1)
band_labels = {90:95, 150:150, 220:220}
for jband, band in enumerate([90, 150, 220]):
    _ = plt.hist(nei_0p1_hz[band],
                 bins=np.arange(0.01,35,1), alpha=0.5,
                 label='{} GHz'.format(band_labels[band]), color='C{}'.format(jband))
    _ = plt.hist(nei_0p1_hz[band],
                 bins=np.arange(0.01,35,1), histtype='step',
                 color='C{}'.format(jband))
    noise = np.array(nei_0p1_hz[band])
    print('{} GHz median NEI = {:.2f} pA/rtHz'.format(band, np.median(noise[np.isfinite(noise) & \
                                                                            (noise>1)])))
plt.xlim([0, 35])
plt.annotate('0.1 Hz', (0.05, 0.9), fontsize=16, xycoords='axes fraction')
plt.xlabel('current noise (pA/$\sqrt{Hz}$)', fontsize=14)
plt.ylabel('readout channels', fontsize=14)
plt.gca().tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid()
plt.savefig('NEI_0p1_hz_no_photons.png', dpi=200)
```

```python

```
