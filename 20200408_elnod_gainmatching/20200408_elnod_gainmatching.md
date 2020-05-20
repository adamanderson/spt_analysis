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

```python

print(arr.std())
```

```python
sigmaclip?
```

```python

```
