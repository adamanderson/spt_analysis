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

# Gain-matching with Elnods
Name says it all. Elnods match gains on atmosphere, while our current calibration pipeline does not necessarily.

```python
from spt3g import core, calibration
import numpy as np
from scipy.stats import sigmaclip
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import decimate
```

## Initial investigation
Let's get these coefficients from the elnod autoprocessing output and then compare to standard gain-matching.

```python
fname_elnod = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
              'gainmatching_noise_89970605_elnod_match.g3'
fname_nuclear = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_89970605_nuclear_match.g3'

d_elnod = list(core.G3File(fname_elnod))[1]
d_nuclear = list(core.G3File(fname_nuclear))[1]
coeffs_elnod = [d_nuclear['GainMatchCoeff'][bolo] \
                for bolo in d_elnod['GainMatchCoeff'].keys()
                if bolo in d_nuclear['GainMatchCoeff'].keys()]
coeffs_nuclear = [d_elnod['GainMatchCoeff'][bolo] \
                  for bolo in d_elnod['GainMatchCoeff'].keys()
                  if bolo in d_nuclear['GainMatchCoeff'].keys()]
```

```python
_ = plt.hist(coeffs_elnod, bins=np.linspace(0.5,1.5,101), histtype='step',
             label='elnod-derived')
_ = plt.hist(coeffs_nuclear, bins=np.linspace(0.5,1.5,101), histtype='step',
             label='nuclear-derived')
plt.xlabel('gain-matching coefficient')
plt.legend()
plt.tight_layout()
plt.savefig('gain_coeff_comparison.png', dpi=200)
```

```python
plt.plot(coeffs_elnod, coeffs_nuclear, '.', markersize=1)
plt.axis([0.5, 1.5, 0.7, 1.3])
```

## All observations

```python
fnames_elnod = glob('/spt/user/adama/instrument_paper_2019_noise_poly1_elnod/fullrate/*g3')
fnames_default = glob('/spt/user/adama/instrument_paper_2019_noise_poly1_default/fullrate/*g3')
fnames_nuclear = glob('/spt/user/adama/instrument_paper_2019_noise_poly1_0p1_to_1p0_nuclear/fullrate/*g3')

coeffs_elnod = []
coeffs_default = []
coeffs_nuclear = []

for fname in fnames_elnod:
    d_elnod = list(core.G3File(fname))[1]
    coeffs_elnod.extend([d_elnod['GainMatchCoeff'][bolo] \
                         for bolo in d_elnod['GainMatchCoeff'].keys()])
for fname in fnames_default:
    d_default = list(core.G3File(fname))[1]
    coeffs_default.extend([d_default['GainMatchCoeff'][bolo] \
                           for bolo in d_default['GainMatchCoeff'].keys()])
for fname in fnames_nuclear:
    d_nuclear = list(core.G3File(fname))[1]
    coeffs_nuclear.extend([d_nuclear['GainMatchCoeff'][bolo] \
                           for bolo in d_nuclear['GainMatchCoeff'].keys()])
```

```python
arr, low, upp = sigmaclip(coeffs_elnod)
coeffs_elnod_std = arr.std()

arr, low, upp = sigmaclip(coeffs_nuclear)
coeffs_nuclear_std = arr.std()

_ = plt.hist(coeffs_elnod, bins=np.linspace(0.75,1.25,101),
             histtype='step', label='elnod matching (std = {:.3f})'.format(coeffs_elnod_std))
_ = plt.hist(coeffs_nuclear, bins=np.linspace(0.75,1.25,101),
             histtype='step', label='nuclear matching (std = {:.3f})'.format(coeffs_nuclear_std))
# _ = plt.hist(coeffs_default, bins=np.linspace(0.75,1.25,101),
#              histtype='step', label='default matching')
plt.legend()
# plt.gca().set_yscale('log')
plt.xlim([0.75, 1.25])
plt.xlabel('gain-matching coefficient')
plt.ylabel('bolometer x scans')
plt.tight_layout()
plt.savefig('gain_match_coeff_comparison.png', dpi=150)
```

## Investigating low-frequency noise in the horizon noise stare
Bill was wondering whether the hypothesis that spurious non-atmospheric noise sources dominate at very low-frequency is true. We can test this by inspecting the horizon noise stares and seeing if the low-frequency spectrum is comparable in amplitude to the low-frequency noise in our usual noise stares.

```python
horizon_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_77863968_horizon_default_current.g3'
horizon_data = list(core.G3File(horizon_fname))
```

```python
normal_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_81433244_default_current.g3'
normal_data = list(core.G3File(normal_fname))
```

```python
from spt3g.calibration.template_groups import get_template_groups
bolo_tgroups = get_template_groups(horizon_data[0]["BolometerProperties"], 
                                            per_band = True,
                                            per_wafer = True,
                                            include_keys = True)
```

```python
wafers = ['w172', 'w174', 'w176', 'w177', 'w180',
          'w181', 'w188', 'w203', 'w204', 'w206']
bands = [90, 150, 220]

for band in bands:
    plt.figure(figsize=(10,20))
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        nbolos = 0
        avg_asd_horizon = []
        for bolo in group:
            if bolo in horizon_data[1]['ASD'] and \
               np.all(np.isfinite(horizon_data[1]['ASD'][bolo])):
                if len(avg_asd_horizon) == 0:
                    avg_asd_horizon = horizon_data[1]['ASD'][bolo]
                else:
                    avg_asd_horizon += horizon_data[1]['ASD'][bolo]
            nbolos += 1
        avg_asd_horizon /= nbolos

        nbolos = 0
        avg_asd_normal = []
        for bolo in group:
            if bolo in normal_data[1]['ASD'] and \
               np.all(np.isfinite(normal_data[1]['ASD'][bolo])):
                if len(avg_asd_normal) == 0:
                    avg_asd_normal = normal_data[1]['ASD'][bolo]
                else:
                    avg_asd_normal += normal_data[1]['ASD'][bolo]
            nbolos += 1
        avg_asd_normal /= nbolos

        plt.subplot(5,2,jwafer+1)
        plt.semilogx(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz,
                   avg_asd_horizon)
        plt.semilogx(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz,
                   avg_asd_normal)
        plt.ylim([10,30])
        plt.title('{} GHz: {}'.format(band, wafer))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('current noise [pA/rtHz]')
    plt.tight_layout()
    plt.savefig('horizon_vs_intransition_{}.png'.format(band))
```

```python
np.sqrt(15**2 - 10**2)
```

```python

```
