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

# Gain-matching in Field Observations
The purpose of this note is to test out nuclear gain-matching on field observations, and to compare the performance with noise stares. Let's start out by just plotting some PSDs as a sanity check.

```python
from spt3g import core, calibration, mapmaker
import numpy as np
import matplotlib.pyplot as plt
import os.path
from glob import glob
```

```python
datadir = '/spt/user/adama/20190811_field_gainmatch_test/downsampled/'
d = list(core.G3File(os.path.join(datadir, 'gainmatching_ra0hdec-44.75_80296951.g3')))
boloprops = list(core.G3File('/spt/data/bolodata/downsampled/noise/'
                             '73798315/offline_calibration.g3'))[0]['BolometerProperties']
```

```python
for fr in d:
    for band in ['90.0']:
        if 'AverageASDDiff' in fr.keys():
            if 'frequency' in fr['AverageASDDiff'].keys():
                freq = fr['AverageASDDiff']['frequency'] / core.G3Units.Hz
                asd = fr['AverageASDDiff']['{}_w204'.format(band)]
                plt.loglog(freq, asd)
plt.ylim([100,1e4])
```

Now let's move on to the main attraction of this note. We load up the data from the noise stares and field observations with and without gain-matching, and then plot the ratio of the integrated low-frequency noise in these two normalization schemes. The effect of imperfect gain-matching in a fixed frequency range is to increase the total noise because the entire 1/f spectrum is shifting to higher frequencies as we scan. This is fully consistent with expectations.

```python
fnames_gainmatch = glob('/spt/user/adama/20190811_field_gainmatch_test/downsampled/*g3')
fnames_rcw38 = glob('/spt/user/adama/20190811_field_rcw38match_test/downsampled/*g3')

fnames_gainmatch_stare = glob('/spt/user/adama/20190809_noise_gainmatch_cal/downsampled/*g3')
fnames_rcw38_stare = glob('/spt/user/adama/20190809_noise_rcw38_cal/downsampled/*g3')
```

```python
all_coeffs = {90:np.array([]), 150:np.array([]), 220:np.array([])}
all_lowf_power_nuclear = {90:[], 150:[], 220:[]}
all_lowf_power_rcw38 = {90:[], 150:[], 220:[]}
for fn_rcw38 in fnames_rcw38[:10]:
    filename = os.path.basename(fn_rcw38)
    fn_gainmatch = np.unique([fn for fn in fnames_gainmatch if filename in fn])
    
    if len(fn_gainmatch) != 0:
        d_rcw38 = list(core.G3File(fn_rcw38))
        d_nuclear = list(core.G3File(fn_gainmatch))
        
        for fr_rcw38, fr_nuclear in zip(d_rcw38, d_nuclear):
            if 'GainMatchCoeff' in fr_rcw38.keys():
                for jband, band in enumerate([90, 150, 220]):
                    coeffs = np.array([fr_nuclear["GainMatchCoeff"][bolo] \
                                       for bolo in fr_nuclear["GainMatchCoeff"].keys() \
                                       if boloprops[bolo].band/core.G3Units.GHz == band])
                    all_coeffs[band] = np.append(all_coeffs[band], coeffs)

                    if len(fr_nuclear['AverageASDDiff']) > 0 and \
                       len(fr_rcw38['AverageASDDiff']) > 0:
                        freqs = np.array(fr_nuclear['AverageASDDiff']['frequency']) / core.G3Units.Hz
                        for group in fr_nuclear['AverageASDDiff'].keys():
                            if str(band) in group:
                                asd_nuclear = np.array(fr_nuclear['AverageASDDiff'][group])
                                asd_rcw38 = np.array(fr_rcw38['AverageASDDiff'][group])
                                all_lowf_power_nuclear[band].append(np.mean(asd_nuclear[(freqs>0.01) & (freqs<0.1)]))
                                all_lowf_power_rcw38[band].append(np.mean(asd_rcw38[(freqs>0.01) & (freqs<0.1)]))
```

```python
all_coeffs_stare = {90:np.array([]), 150:np.array([]), 220:np.array([])}
all_lowf_power_nuclear_stare = {90:[], 150:[], 220:[]}
all_lowf_power_rcw38_stare = {90:[], 150:[], 220:[]}
for fn_rcw38 in fnames_rcw38_stare:
    filename = os.path.basename(fn_rcw38)
    fn_gainmatch = np.unique([fn for fn in fnames_gainmatch_stare if filename in fn])
    
    if len(fn_gainmatch) != 0:
        fr_rcw38 = list(core.G3File(fn_rcw38))[1]
        fr_nuclear = list(core.G3File(fn_gainmatch))[1]

        for jband, band in enumerate([90, 150, 220]):
            coeffs = np.array([fr_nuclear["GainMatchCoeff"][bolo] \
                               for bolo in fr_nuclear["GainMatchCoeff"].keys() \
                               if boloprops[bolo].band/core.G3Units.GHz == band])
            all_coeffs_stare[band] = np.append(all_coeffs[band], coeffs)

            if len(fr_nuclear['AverageASDDiff']) > 0 and \
               len(fr_rcw38['AverageASDDiff']) > 0:
                freqs = np.array(fr_nuclear['AverageASDDiff']['frequency']) / core.G3Units.Hz
                for group in fr_nuclear['AverageASDDiff'].keys():
                    if str(band) in group:
                        asd_nuclear = np.array(fr_nuclear['AverageASDDiff'][group])
                        asd_rcw38 = np.array(fr_rcw38['AverageASDDiff'][group])
                        all_lowf_power_nuclear_stare[band].append(np.mean(asd_nuclear[(freqs>0.01) & (freqs<0.1)]))
                        all_lowf_power_rcw38_stare[band].append(np.mean(asd_rcw38[(freqs>0.01) & (freqs<0.1)]))
```

```python
plt.figure(figsize=(12,4))
for jband, band in enumerate([90, 150, 220]):
    plt.subplot(1,3,jband+1)
    plt.hist(np.array(all_lowf_power_rcw38[band]) / \
             np.array(all_lowf_power_nuclear[band]),
             bins=np.linspace(0.9, 1.9),
             histtype='step', label='field observations',
             normed=True)
    plt.hist(np.array(all_lowf_power_rcw38_stare[band]) / \
             np.array(all_lowf_power_nuclear_stare[band]),
             bins=np.linspace(0.9, 1.9),
             histtype='step', label='noise stares',
             normed=True)
    plt.title('{} GHz'.format(band))
plt.subplot(1,3,2)
plt.xlabel('ratio average power in 0.01-0.1 Hz, RCW calib / nuclear calib')

plt.subplot(1,3,1)
plt.ylabel('normalized scans [arb.]')
plt.tight_layout()
plt.legend()
plt.savefig('figures/lowf_power_ratio_rcw38_nuclear_noise+field.png', dpi=120)
```

```python
coeffs
```

```python
fnames_gainmatch
```

```python
fnames_rcw38
```

```python
d = list(core.G3File('/spt/user/adama/20190811_field_gainmatch_test/downsampled/gainmatching_ra0hdec-44.75_80314513.g3'))
```

```python
d[4]['GainMatchCoeff']['2019.0p9']
```

```python

```
