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

# FMUX Noise Studies with Montgojo

15 October 2019

Joshua raised some reasonable concerns about the frequency-dependent transfer function used in noise measurements that we plan to show at the CMB-S4 collaboration meeting. This note describes some super-quick investigations.

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np
from spt3g import core, calibration
from spt3g.dfmux import HousekeepingForBolo
from spt3g.dfmux.unittransforms import counts_to_rms_amps
```

## In-transition Noise
This was just a quick look at the in-transition noise from the `measure_noise` algorithm. Not much to see here, other than things look similar to what we have seen before.

```python
noise_fname = '/big_scratch/pydfmux_output/20190910/20190911_035009_measure_noise_323/'\
              'data/11411_BOLOS_INFO_AND_MAPPING.pkl'
with open(noise_fname, 'rb') as f:
    d = pickle.load(f)
```

```python
i_noise = [d[chan]['noise']['i_phase']['median_noise'] for chan in d.keys()]
freqs = [d[chan]['frequency'] for chan in d.keys()]
```

```python
plt.plot(freqs, i_noise, '.', markersize=1)
plt.ylim([0,50])
```

## Horizon Noise Studies
The obsid of the horizon noise stare was 77863968. This noise stare was processed through the usual pipeline. Let's just do a quick plot as a sanity check.

```python
d = list(core.G3File('/spt/user/production/calibration/noise/77863968.g3'))
dhk = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/0000.g3'))
dcal = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/offline_calibration.g3'))
bps = dcal[0]["BolometerProperties"]
```

```python
noise = np.array([d[0]["NEI_30.0Hz_to_40.0Hz"][bolo] for bolo in d[0]["NEI_30.0Hz_to_40.0Hz"].keys()]) / \
        (core.G3Units.amp*1e-12 / np.sqrt(core.G3Units.Hz))

freqs = np.zeros(len(noise))
for jbolo, bolo in enumerate(d[0]["NEI_30.0Hz_to_40.0Hz"].keys()):
    boardhk, mezzhk, modhk, chanhk = HousekeepingForBolo(dhk[3]["DfMuxHousekeeping"],
                                                         dhk[2]["WiringMap"], bolo, True)
    freqs[jbolo] = chanhk.carrier_frequency / core.G3Units.Hz / 1e6
```

```python
_ = plt.hist(noise[np.isfinite(noise)], bins=np.linspace(0,30,31))
```

```python
plt.plot(freqs, noise, '.', markersize=1)
plt.axis([1.5, 5.5, 5.0, 35])
plt.xlabel('frequency [MHz]')
plt.ylabel('current noise [pA/rtHz]')
plt.tight_layout()
plt.savefig('nei_f_dependence.png', dpi=200)
```

## S/N Plots
Next, Joshua has claimed that the S/N ratio is a immune to TF systematics that might affect the noise alone. In this ratio, the "signal" is the DC level of the nuller current being applied to cancel the carrier tone just before the SQUID input coil. The "noise" is the white noise floor of the nuller current, i.e. what we usually just refer to as the bolometer current noise.

Is Joshua correct in his assertion that this is more robust to TF systematics, and therefore more reliable than the current/nuller noise alone? Some comments:
1. **TF systematics cancel:** If there is a TF miscalibration that inserts a frequency dependence in the nuller TF, then yes, this will cancel in the S/N ratio, and Joshua is fully correct.
1. **S and N are sensitive to Psat in different bands:** Because the "signal" is really the bolometer bias current (with some possible calibration error), it will be dramatically different between 90, 150, 220 GHz because the bands have different Psat. The frequency-dependent comparison is therefore only valid within a band--unless some additional correction is applied.
1. **Current-division cancels in S/N, but is still "real" noise:** (Less confident that there isn't a logical error in this one) The S/N being flat within the band does not strictly imply that the frequency-dependence observed in N is an unphysical TF effect. The current-division is a good example of this. Suppose for the sake of argument that we were entirely dominated by 1st-stage amplifier noise. The frequency-dependence of that would be the same as the frequency-dependence of the nuller amplitude induced by current-division. Yet, the noise enhancement is definitely not spurious.

It's fair to say that Joshua has a point that the TF may be confusing our noise measurements, and certain calibration errors absolutely do cancel in the S/N ratio. But the S/N ratio alone isn't totally unambiguous, and we should undertake a careful TF calibration program. Luckily for us, Amy Bender has already started this program at Argonne.

```python
nuller_signal = []
for bolo in d[0]["NEI_30.0Hz_to_40.0Hz"].keys():
    cal_factor = counts_to_rms_amps(dhk[2]["WiringMap"], dhk[3]["DfMuxHousekeeping"],
                                    bolo, 'ICE', tf='spt3g_filtering_2017_full')
    nuller_signal.append(np.median(dhk[3]["RawTimestreams_I"][bolo]) * cal_factor / core.G3Units.microamp)
nuller_signal = np.array(nuller_signal)
```

```python
plt.figure(1)
plt.plot(freqs, noise, '.', markersize=1)
plt.axis([1.5, 5.5, 5.0, 35])
plt.xlabel('frequency [MHz]')
plt.ylabel('current noise [pA/rtHz]')
plt.tight_layout()

plt.figure(2)
plt.plot(freqs, nuller_signal, '.', markersize=1)
plt.axis([1.5, 5.5, 0, 3.5])
plt.xlabel('frequency [MHz]')
plt.ylabel('current at SQUID [uArms]')
plt.tight_layout()

plt.figure(3)
plt.plot(freqs, nuller_signal / noise * (1e-6) / (1e-12), '.', markersize=1)
plt.axis([1.5, 5.5, 1000, 250000])
plt.xlabel('frequency [MHz]')
plt.ylabel('S/N ratio at SQUID [1/sqrt{Hz}]')
plt.tight_layout()

# plt.savefig('nei_f_dependence.png', dpi=200)
```

## Remaking Amy's Plot with Photon NEP

```python
d2 = list(core.G3File('/home/adama/SPT/spt_analysis/20190329_gainmatching/'
                     'horizon_noise_77863968_bender_ltd_perbolo_only.g3'))
```

```python
noise = np.array([np.sqrt(d2[1]["ASDFitParams"][bolo][0])
                  for bolo in d2[1]["ASDFitParams"].keys() if len(d2[1]["ASDFitParams"][bolo])==3])
bolo_band = np.array([bps[bolo].band / core.G3Units.GHz
                      for bolo in d2[1]["ASDFitParams"].keys() if len(d2[1]["ASDFitParams"][bolo])==3])
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
plt.xlabel('White Noise Level (pA/$\sqrt{Hz}$)', fontsize=14)
plt.ylabel('Number of Readout Channels', fontsize=14)
plt.gca().tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('NEI_no_photons.png', dpi=200)
```

## Thermal conductivity estimates

```python
from scipy.integrate import quad

def kA(T): return 6 * T**0.92

length_1K_100mK = 300 #[mm]
length_4K_1K = 300 #[mm]
n_stripline_pairs = 14
```

```python
print('1K to 100mK:')
quad(kA, 0.1, 1)[0] / length_1K_100mK * n_stripline_pairs
```

```python
print('4K to 1K:')
quad(kA, 1, 4)[0] / length_4K_1K * n_stripline_pairs
```

```python
print('350mK to 250mK:')
quad(kA, 0.25, 0.35)[0] / 100 * 12
```

```python
Vbias = 2e-6 #[volts]
R = 1 #[Ohm]
Ndet = 1800

Vbias**2 / R * Ndet 
```

```python

```
