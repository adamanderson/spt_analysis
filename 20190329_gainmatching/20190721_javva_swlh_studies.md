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

# Some studies for Bill and Jessica
Our Berkeley collaborators have been asking me some questions. This note is my attempt to finally look into some of them.

```python
from spt3g import core, dfmux, calibration
import numpy as np
import matplotlib.pyplot as plt
```

## Updating the gain-matching
Well this is awkward. The gain-matching scheme that I worked out in my previous note on this subject isn't quite right since the renormalization coefficient obviously depends on the gain-matching coefficient. This means that the gain-matching coefficient therefore doesn't quite minimize the integrated power in the target band, and we really should not have applied a post-hoc renormalization.

To do this the correct way, we use the method of Lagrange multipliers to minimize the quantity
$$
\sum_{i \in F} \left| f_X X_i - f_Y Y_i \right|^2
$$
subject to the constraint that the power in the summed timestreams is the same before and after the gain-matching
$$
\sum_{i \in F} \left| f_X X_i + f_Y Y_i \right|^2 = \sum_{i \in F} \left| X_i + Y_i \right|^2,
$$
where $X_i$ and $Y_i$ are the $i$th bin of the Fourier transform of the two detectors in a polarization pair, and the sum is over a specific frequency range to use in the optimization. We form the Lagrangian
$$
\mathcal{L}(f_X, f_Y, \lambda) = \sum_{i \in F} \left| f_X X_i - f_Y Y_i \right|^2 - \lambda \left( \sum_{i \in F} \left| f_X X_i + f_Y Y_i \right|^2 - \sum_{i \in F} \left| X_i + Y_i \right|^2\right),
$$
and compute partial derivatives with respect to each variable
\begin{align}
\partial_{f_X}: 0 &= 2 f_X \sum_{i \in F} X_i^* X_i - 2 f_Y \sum_{i \in F} \Re(X_i^* Y_i) - \lambda \left( 2 f_X \sum_{i \in F} X_i^* X_i + 2 f_Y \sum_{i \in F} \Re(X_i^* Y_i) \right) \\
\partial_{f_y}: 0 &= 2 f_Y \sum_{i \in F} Y_i^* Y_i - 2 f_X \sum_{i \in F} \Re(X_i^* Y_i) - \lambda \left( 2 f_Y \sum_{i \in F} Y_i^* Y_i + 2 f_X \sum_{i \in F} \Re(X_i^* Y_i) \right) \\
\partial_\lambda: 0 &= f_X^2 \sum_{i \in F} X_i^* X_i + f_Y^2 \sum_{i \in F} Y_i^* Y_i + 2 f_X f_Y \sum_{i \in F} \Re(X_i^* Y_i) - \sum_{i \in F} X_i^* X_i - \sum_{i \in F} Y_i^* Y_i - 2 \sum_{i \in F} \Re(X_i^* Y_i)
\end{align}
Solving for $\lambda$ between $\partial_{f_X}$ and $\partial_{f_Y}$, we get
\begin{align}
\left( 2 f_X \sum_{i \in F} X_i^* X_i - 2 f_Y \sum_{i \in F} \Re(X_i^* Y_i) \right) \left( 2 f_Y \sum_{i \in F} Y_i^* Y_i + 2 f_X \sum_{i \in F} \Re(X_i^* Y_i) \right) &= \left( 2 f_X \sum_{i \in F} X_i^* X_i + 2 f_Y \sum_{i \in F} \Re(X_i^* Y_i) \right) \left( 2 f_Y \sum_{i \in F} Y_i^* Y_i - 2 f_X \sum_{i \in F} \Re(X_i^* Y_i) \right) \\
f_X^2 \sum_{i \in F} X_i^* X_i &= f_Y^2 \sum_{i \in F} Y_i^* Y_i\\
f_Y &= f_X \sqrt{\frac{\sum_{i \in F} X_i^* X_i}{\sum_{i \in F} Y_i^* Y_i}}
\end{align}
Substituting this result into $\partial_\lambda$, we can solve for $f_Y$
\begin{align}
0 &= 2 f_X^2 \sum_{i \in F} X_i^* X_i + 2 f_X^2 \sqrt{\frac{\sum_{i \in F} X_i^* X_i}{\sum_{i \in F} Y_i^* Y_i}} \sum_{i \in F} \Re(X_i^* Y_i) - \sum_{i \in F} X_i^* X_i - \sum_{i \in F} Y_i^* Y_i - 2 \sum_{i \in F} \Re(X_i^* Y_i)\\
f_X &= \sqrt{\frac{\sum_{i \in F} X_i^* X_i + \sum_{i \in F} Y_i^* Y_i + 2 \sum_{i \in F} \Re(X_i^* Y_i)}{2\sum_{i \in F} X_i^* X_i + 2 \sum_{i \in F} \Re(X_i^* Y_i) \sqrt{\frac{\sum_{i \in F} X_i^* X_i}{\sum_{i \in F} Y_i^* Y_i}}}}.
\end{align}
By symmetry arguments, we have
$$
f_Y = \sqrt{\frac{\sum_{i \in F} X_i^* X_i + \sum_{i \in F} Y_i^* Y_i + 2 \sum_{i \in F} \Re(X_i^* Y_i)}{2\sum_{i \in F} Y_i^* Y_i + 2 \sum_{i \in F} \Re(X_i^* Y_i) \sqrt{\frac{\sum_{i \in F} Y_i^* Y_i}{\sum_{i \in F} X_i^* X_i}}}}.
$$


## Nuclear vs. RCW38/MAT5A gain matching
I compared the nuclear gain-matching coefficients the RCW38/MAT5A gain-matching coefficients in a previous note, and they agree very well. Let's check here how much this translates into better/worse low-frequency noise.

```python
fr_nuclear = list(core.G3File('noise_stare_73798315_nuclear_cal_v2.g3'))[1]
# fr_nuclear = list(core.G3File('test.g3'))[1]
fr_rcw38 = list(core.G3File('noise_stare_73798315_rcw38_cal_v2.g3'))[1]
boloprops = list(core.G3File('/spt/data/bolodata/downsampled/noise/'
                             '73798315/offline_calibration.g3'))[0]['BolometerProperties']
```

```python
power_nuclear = {}
ff = np.array(fr_nuclear["ASD"]['frequency'] / core.G3Units.Hz)
for bolo in fr_nuclear["ASD"].keys():
    if bolo != 'frequency':
        asd = np.array(fr_nuclear["ASD"][bolo])
        power_nuclear[bolo] = np.sum(asd[(ff>0.01) & (ff<0.1)])
        
power_rcw38 = {}
ff = np.array(fr_rcw38["ASD"]['frequency'] / core.G3Units.Hz)
for bolo in fr_rcw38["ASD"].keys():
    if bolo != 'frequency':
        asd = np.array(fr_rcw38["ASD"][bolo])
        power_rcw38[bolo] = np.sum(asd[(ff>0.01) & (ff<0.1)])
```

```python
powers = np.array(list(power_nuclear.values()))
_ = plt.hist(powers[np.isfinite(powers)],
             bins=np.linspace(0,5e5,101),
             histtype='step')

powers = np.array(list(power_rcw38.values()))
_ = plt.hist(powers[np.isfinite(powers)],
             bins=np.linspace(0,5e5,101),
             histtype='step')
```

```python
band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}

for jband, band in enumerate([90., 150., 220.]):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, num=jband+1, figsize=(20,6))
    ax = ax.flatten()
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)
        
        ff_nuclear = np.array(fr_nuclear['AverageASDDiff']['frequency']/core.G3Units.Hz)
        asd_nuclear = np.array(fr_nuclear['AverageASDDiff'][group]) / np.sqrt(2.)
        ff_rcw38 = np.array(fr_rcw38['AverageASDDiff']['frequency']/core.G3Units.Hz)
        asd_rcw38 = np.array(fr_rcw38['AverageASDDiff'][group]) / np.sqrt(2.)

        ax[jwafer].loglog(ff_nuclear, asd_nuclear, label='nuclear gain-matching')
        ax[jwafer].loglog(ff_rcw38, asd_rcw38, label='rcw38 gain-matching')
        ax[jwafer].set_title('{}, {} GHz'.format(group.split('_')[1], int(float(group.split('_')[0]))))       
    for jwafer in [5,6,7,8,9]:
        ax[jwafer].set_xlabel('frequency [Hz]')
    
    ax[0].set_ylabel('N [pA$ / \sqrt{Hz}$]')
    ax[5].set_ylabel('NEI [pA$ / \sqrt{Hz}$]')
    plt.ylim([1e2,1e5])
    plt.legend()
    plt.tight_layout()
        
    subplot_numbers[band] +=1
        
for band, jplot in band_numbers.items():
    plt.figure(jplot)
    plt.savefig('difference_noise_{}_73798315.png'.format(int(band)), dpi=120)
```

```python
band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}

for jband, band in enumerate([90., 150., 220.]):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, num=jband+1, figsize=(20,6))
    ax = ax.flatten()
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)
        
        ff_nuclear = np.array(fr_nuclear['AverageASDSum']['frequency']/core.G3Units.Hz)
        asd_nuclear = np.array(fr_nuclear['AverageASDSum'][group]) / np.sqrt(2.)
        ff_rcw38 = np.array(fr_rcw38['AverageASDSum']['frequency']/core.G3Units.Hz)
        asd_rcw38 = np.array(fr_rcw38['AverageASDSum'][group]) / np.sqrt(2.)

        ax[jwafer].loglog(ff_nuclear, asd_nuclear, label='nuclear gain-matching')
        ax[jwafer].loglog(ff_rcw38, asd_rcw38, label='rcw38 gain-matching')
        ax[jwafer].set_title('{}, {} GHz'.format(group.split('_')[1], int(float(group.split('_')[0]))))       
    for jwafer in [5,6,7,8,9]:
        ax[jwafer].set_xlabel('frequency [Hz]')
    
    ax[0].set_ylabel('N [pA$ / \sqrt{Hz}$]')
    ax[5].set_ylabel('NEI [pA$ / \sqrt{Hz}$]')
    plt.ylim([1e2,1e6])
    plt.legend()
    plt.tight_layout()
        
    subplot_numbers[band] +=1
```

```python
band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}

for jband, band in enumerate([90., 150., 220.]):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, num=jband+1, figsize=(20,6))
    ax = ax.flatten()
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)
        
        ff_nuclear = np.array(fr_nuclear['AverageASDSum']['frequency']/core.G3Units.Hz)
        asd_sum = np.array(fr_nuclear['AverageASDSum'][group]) / np.sqrt(2.)
        asd_diff = np.array(fr_nuclear['AverageASDDiff'][group]) / np.sqrt(2.)

        ax[jwafer].loglog(ff_nuclear, asd_sum, label='sum')
        ax[jwafer].loglog(ff_nuclear, asd_diff, label='difference')
        ax[jwafer].set_title('{}, {} GHz'.format(group.split('_')[1], int(float(group.split('_')[0]))))       
    for jwafer in [5,6,7,8,9]:
        ax[jwafer].set_xlabel('frequency [Hz]')
    
    ax[0].set_ylabel('N [pA$ / \sqrt{Hz}$]')
    ax[5].set_ylabel('NEI [pA$ / \sqrt{Hz}$]')
    plt.ylim([1e2,1e6])
    plt.legend()
    plt.tight_layout()
        
    subplot_numbers[band] +=1
```

```python
for band in [90, 150, 220]:
    coeffs_nuclear = np.array([fr_nuclear["GainMatchCoeff"][bolo]\
                               for bolo in fr_nuclear["GainMatchCoeff"].keys()\
                               if boloprops[bolo].band/core.G3Units.GHz == band])
    #_ = plt.hist(coeffs_rcw38, bins=np.linspace(0.75,1.25,41))
    _ = plt.hist(coeffs_nuclear,
                 bins=np.linspace(0.5,1.5,41),
                 histtype='step',
                 label='{} GHz'.format(band))
plt.legend()
plt.xlabel('gain-matching coefficient')
plt.tight_layout()
plt.savefig('gain_match_coeff.png', dpi=120)
```

```python

```
