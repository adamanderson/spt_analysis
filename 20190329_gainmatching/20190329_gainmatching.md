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

# Fourier-space gain-matching
We would ideally gain-match using a continuously-running optical signal which would monitor the instantaneous gain of the detectors in each polarization pair. In the limit that we are not doing this, we need to gain-match to another optical signal present in each scan of the telescope. On a scan-by-scan basis, the power spectrum of a bolometer timestream consists of a.) a relatively flat noise floor dominated by photon noise with a non-negligible readout noise component, and b.) a 1/f component that is dominated by optical atmospheric fluctuations with possible contributions from exotic readout noise terms. This situation leaves us with only one option: we must gain-match on the 1/f noise, implicitly assuming that it is due to optical signals.

Let $d_x(t)$ and $d_y(t)$ be the two detector timestreams in a polarization pair. Define the gain-matched pair-summed and pair-differenced timestream by:
$$
d_+(t) = C (d_x(t) + Ad_y(t))
d_-(t) = C (d_x(t) - Ad_y(t))
$$
where $A$ is a free gain-matching parameter, and after fixing $A$ we set $C$ to normalize the pair-summed timestream to be equal to pre-gain matched pair-summed timestream. Note that this choice of normalization convention is not necessarily unique. In our calibration pipeline, a relative gain-matching error between x and y implies that the absolute calibration of x and y has an error, so the absolute calibration of the sum must also have an error. But gain-matching only improves the relative calibration, and it does nothing about absolute calibration. 

We want to choose an $A$ that provides "optimal" gain-matching in some sense. Ultimately, the entire purpose of gain-matching is to reduce 1/f noise, so it is natural to choose $A$ such that the noise in some low-frequency interval is minimized. In other words, we want to find
$$
\begin{align}
\hat{A} &= \underset{A}{\arg\min} \sum_{i \in F} \left| \tilde{d}_-(f_i) \right|^2 \\
&= \underset{A}{\arg\min} \sum_{i \in F} \left| \tilde{d}_x(f_i) - A \tilde{d}_y(f_i) \right|^2
\end{align}
$$

To find $\hat{A}$, we simply solve for the first-order conditions:
$$
\begin{align}
0 &= \frac{d}{dA}  \sum_{i \in F} \left(\tilde{d}^*_x(f_i) - A \tilde{d}^*_y(f_i) \right) \left(\tilde{d}_x(f_i) - A \tilde{d}_y(f_i) \right) |_{A = \hat{A}}\\
0 &=\sum_{i \in F} 2 \hat{A} \tilde{d}^*_y(f_i)\tilde{d}_y(f_i) - \tilde{d}^*_x(f_i)\tilde{d}_y(f_i) - \tilde{d}^*_y(f_i)\tilde{d}_x(f_i)\\
\hat{A} &= \frac{\sum_{i \in F} \left(\tilde{d}^*_x(f_i)\tilde{d}_y(f_i) + \tilde{d}^*_y(f_i)\tilde{d}_x(f_i) \right)}{\sum_{i \in F} 2\tilde{d}^*_y(f_i)\tilde{d}_y(f_i)}
\end{align}
$$

Having fixed the gain-matching parameter $A$, we can compute the normalization parameter $C$ from the constraint that the power of the new pair-summed timestream must be equal to the old pair-summed timestream:
$$
\sum_{i \in F} \left| C(\tilde{d}^*_x(f_i) + \tilde{d}^*_y(f_i)) \right|^2 = \sum_{i \in F} \left| \tilde{d}^*_x(f_i) + \hat{A}\tilde{d}^*_y(f_i) \right|^2 \\
C = \sqrt{\frac{\sum_{i \in F} \left| \tilde{d}^*_x(f_i) + \hat{A}\tilde{d}^*_y(f_i) \right|^2}{\sum_{i \in F} \left|\tilde{d}^*_x(f_i) + \tilde{d}^*_y(f_i)\right|^2}}
$$

Let's implement this in a noise stare and then perform some Monte Carlo simulations to check for biases and optimality.

```python
from spt3g import core, dfmux, calibration
from spt3g.todfilter import dftutils
import numpy as np
import matplotlib.pyplot as plt
import os.path
import adama_utils
from importlib import reload
from scipy.signal import welch, periodogram
from scipy.optimize import curve_fit, newton, bisect
from glob import glob

%matplotlib inline
```

```python
from spt3g.calibration.template_groups import get_template_groups

d = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/68609192/offline_calibration.g3')]
bps = d[0]['BolometerProperties']
groups = get_template_groups(bps, per_pixel=True, per_wafer=True, include_keys=True)
good_pairs = []
for group, pair in groups.items():
    if len(pair) == 2:
        good_pairs.append(pair)
```

## Identify good polarization pairs
Let's write a pipeline segment to do some flagging and identify good polarization pairs for gain-matching analysis.

```python
pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=['/spt/data/bolodata/fullrate/noise/68609192/offline_calibration.g3',
                                  '/spt/data/bolodata/fullrate/noise/68609192/0000.g3'])
pipe.Add(core.FlagIncompletePixelPairs)
pipe.Add(core.)
```

## Quick test of method with a noise stare

```python
good_pairs
```

```python
reload(adama_utils)
```

```python
# get the timestreams
ts_data = adama_utils.get_raw_timestreams(['2019.038', '2019.1fv'], 70005920, file_name='0000.g3', scan_num=[0],
                              plot=False, data=True, cut_turnarounds=False,
                              psd=False, units=core.G3TimestreamUnits.Tcmb)
```

```python
ts_data
```

```python
ff, psd = welch(ts_data['2019.038'], fs=152.5, nperseg=1024)
plt.loglog(ff, psd)
ff, psd = welch(ts_data['2019.1fv'], fs=152.5, nperseg=1024)
plt.loglog(ff, psd)
```

```python
tsx = ts_data['2019.038'] - np.mean(ts_data['2019.038'])
tsy = ts_data['2019.1fv'] - np.mean(ts_data['2019.1fv'])
```

```python
fftx = np.fft.fft(tsx)
ffty = np.fft.fft(tsy)
Ahat = np.sum(np.conj(fftx)*ffty + np.conj(ffty)*fftx) / (2.*np.sum(np.abs(ffty)**2.))
```

```python
ff, psd = welch(tsx, fs=152.5, nperseg=1024*4)
plt.loglog(ff, psd)
ff, psd = welch(tsy, fs=152.5, nperseg=1024*4)
plt.loglog(ff, psd)

ff, psd = welch(tsx - tsy, fs=152.5, nperseg=1024*4)
plt.loglog(ff, psd)
```

```python
plt.plot(tsx)
plt.plot(tsy)
plt.plot(tsx - Ahat*tsy)
plt.xlim([0,300])
```

## Fitting PSDs
The other thing that we would like to do after gain-matching is to fit the 1/f knee of the timestreams. Let's write some prototype code on test timestreams, then we'll write a production version of the code. The model that we use is from an old SPT thesis, ultimately derived from something Brad used for SPT-SZ:

$$
\textrm{NEP}(f) = \sqrt{A_{\textrm{white}} + A_{\textrm{red}} f^{-\alpha} + \frac{A_{\textrm{photon}}}{1 + 2\pi (f \tau)^2}}
$$

```python
d = list(core.G3File('gain_match_test_timestreams.g3'))
```

```python
f, psd_diff = welch(d[4]["PairDiffTimestreams"]['2019.7zn_2019.dbl'], fs=152.5, nperseg=2048)
f, psd_sum = welch(d[4]["PairSumTimestreams"]['2019.7zn_2019.dbl'], fs=152.5, nperseg=2048)
f_pg, pg_diff = periodogram(d[4]["PairDiffTimestreams"]['2019.7zn_2019.dbl'], fs=152.5, window='hanning')
f_pg, pg_sum = periodogram(d[4]["PairSumTimestreams"]['2019.7zn_2019.dbl'], fs=152.5, window='hanning')

plt.figure(1)
plt.loglog(f, np.sqrt(psd_diff) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6)
plt.loglog(f, np.sqrt(psd_sum) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6)

plt.figure(2)
plt.loglog(f_pg, np.sqrt(pg_diff) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6)
plt.loglog(f_pg, np.sqrt(pg_sum) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6)
plt.ylim([10, 1e5])
```

```python
def readout_noise(x, readout):
    return readout*np.ones(len(x))
def photon_noise(x, photon, tau):
    return photon / np.sqrt(1 + 2*np.pi*((x*tau)**2))
def atm_noise(x, A, alpha):
    return A * (x)**(-1*alpha)
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha))**2 + photon**2 / (1 + 2*np.pi*((x*tau)**2)))
```

```python
par, cov = curve_fit(noise_model,
                     f[(f>0) & (f<60)],
                     np.sqrt(psd_sum[(f>0) & (f<60)]) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6,
                     bounds=(0, np.inf))
```

```python
plt.loglog(f, np.sqrt(psd_diff) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6, label='x - y')
plt.loglog(f, np.sqrt(psd_sum) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6, label='x + y')
plt.plot(f, noise_model(f,*par), 'k--', label='total')
plt.plot(f, np.sqrt(readout_noise(f,par[0])**2 + \
                    photon_noise(f,par[3],par[4])**2),
         'C2--', label='readout + photon')
plt.plot(f, atm_noise(f,par[1],par[2]), 'C4--',
         label='atmospheric ($f^{-\\alpha}$)')
plt.ylim([100, 1e4])
plt.legend()
```

```python
plt.loglog(f_pg, np.sqrt(pg_diff) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6, label='x - y')
plt.loglog(f_pg, np.sqrt(pg_sum) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6, label='x + y')
plt.plot(f_pg, noise_model(f_pg,*par), 'k--', label='total')
plt.plot(f_pg, np.sqrt(readout_noise(f_pg,par[0])**2 + \
                       photon_noise(f_pg,par[3],par[4])**2),
         'C2--', label='readout + photon')
plt.plot(f_pg, atm_noise(f_pg,par[1],par[2]), 'C4--',
         label='atmospheric ($f^{-\\alpha}$)')
plt.ylim([10, 1e5])
plt.legend()
```

```python
par
```

```python
d = list(core.G3File('gain_match_fit_test.g3'))
```

```python
for bolo in d[4]["SumPSDFitParams"].keys():
    print(d[4]["SumPSDFitParams"][bolo])
```

```python
testpar = [192.323, 46.0894, 2.07056, 119.257, 0.0188139]
plt.loglog(f, noise_model(f,*testpar), 'k--', label='total')
```

## Plotting pair-summed vs. pair-differenced ASDs

```python
fr = list(core.G3File('gain_match_fit_test_73798315.g3'))[1]
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
#     return np.sqrt(readout**2 + (A * (x)**(-1*alpha))**2 + photon**2 / (1 + 2*np.pi*((x*tau)**2)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout


band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}

for jband, band in enumerate([90., 150., 220.]):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, num=jband+1, figsize=(20,6))
    ax = ax.flatten()
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)
        
#         plt.subplot(2, 5, subplot_numbers[band])
        ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
        asd_diff = np.array(fr['AverageASDDiff'][group])
        ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
        asd_sum = np.array(fr['AverageASDSum'][group])

        par_diff = fr["AverageASDDiffFitParams"][group]
        par_sum = fr["AverageASDSumFitParams"][group]
        ax[jwafer].loglog(ff_diff, asd_diff, label='(x - y) / $\sqrt{2}$')
        ax[jwafer].loglog(ff_sum, asd_sum, label='(x + y) / $\sqrt{2}$')
        ax[jwafer].loglog(ff_diff, noise_model(ff_diff, *list(par_diff)), 'k--')
        ax[jwafer].loglog(ff_sum, noise_model(ff_sum, *list(par_sum)), 'k--')
#         ax[jwafer].loglog(ff_sum, readout_noise(ff_sum, par_sum[0]), 'k--')
#         ax[jwafer].loglog(ff_sum, atm_noise(ff_sum, par_sum[1], par_sum[2]), 'k--')
#         ax[jwafer].loglog(ff_sum, photon_noise(ff_sum, par_sum[3], par_sum[4]), 'k--')
        
        try:
            f_knee = bisect(knee_func, a=0.01, b=1.0, args=tuple(par_diff))
            ax[jwafer].set_title('{}, {} GHz ($f_{{knee}}^{{diff}}$ = {:.3f}, '
                                 '$\\alpha^{{diff}}={:.2f}$)'.format(group.split('_')[1],
                                                             int(float((group.split('_')[0]))),
                                                             f_knee, par_diff[2]))
        except ValueError:
            ax[jwafer].set_title('{}, {} GHz'.format(group.split('_')[1], int(float(group.split('_')[0]))))
            
#         plt.tight_layout()
    for jwafer in [5,6,7,8,9]:
        ax[jwafer].set_xlabel('frequency [Hz]')
    
    ax[0].set_ylabel('NET [uK$\sqrt{s}$]')
    ax[5].set_ylabel('NET [uK$\sqrt{s}$]')
    plt.ylim([2e2,1e5])
    plt.legend()
    plt.tight_layout()
        
    subplot_numbers[band] +=1
        
        
for band, jplot in band_numbers.items():
    plt.figure(jplot)
    plt.savefig('pair_differenced_{}_75722403.png'.format(int(band)), dpi=120)
```

## Scratch work

```python
fr = list(core.G3File('/spt/user/adama/20190329_gainmatching/downsampled/gainmatching_noise_73124800.g3'))[1]
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
#     return np.sqrt(readout**2 + (A * (x)**(-1*alpha))**2 + photon**2 / (1 + 2*np.pi*((x*tau)**2)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout

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
        
#         plt.subplot(2, 5, subplot_numbers[band])
        ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
        asd_diff = np.array(fr['AverageASDDiff'][group])
        ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
        asd_sum = np.array(fr['AverageASDSum'][group])

        par_diff = fr["AverageASDDiffFitParams"][group]
        par_sum = fr["AverageASDSumFitParams"][group]
        ax[jwafer].loglog(ff_diff, asd_diff, label='(x - y) / $\sqrt{2}$')
        ax[jwafer].loglog(ff_sum, asd_sum, label='(x + y) / $\sqrt{2}$')
        ax[jwafer].loglog(ff_diff, noise_model(ff_diff, *list(par_diff)), 'k--')
        ax[jwafer].loglog(ff_sum, noise_model(ff_sum, *list(par_sum)), 'k--')
#         ax[jwafer].loglog(ff_sum, readout_noise(ff_sum, par_sum[0]), 'k--')
#         ax[jwafer].loglog(ff_sum, atm_noise(ff_sum, par_sum[1], par_sum[2]), 'k--')
#         ax[jwafer].loglog(ff_sum, photon_noise(ff_sum, par_sum[3], par_sum[4]), 'k--')
        
        try:
            f_knee = bisect(knee_func, a=0.01, b=1.0, args=tuple(par_diff))
            ax[jwafer].set_title('{}, {} GHz ($f_{{knee}}^{{diff}}$ = {:.3f}, '
                                 '$\\alpha^{{diff}}={:.2f}$)'.format(group.split('_')[1],
                                                             int(float((group.split('_')[0]))),
                                                             f_knee, par_diff[2]))
        except ValueError:
            ax[jwafer].set_title('{}, {} GHz'.format(group.split('_')[1], int(float(group.split('_')[0]))))
            
#         plt.tight_layout()
    for jwafer in [5,6,7,8,9]:
        ax[jwafer].set_xlabel('frequency [Hz]')
    
    ax[0].set_ylabel('NET [uK$\sqrt{s}$]')
    ax[5].set_ylabel('NET [uK$\sqrt{s}$]')
    plt.ylim([2e2,1e5])
    plt.legend()
    plt.tight_layout()
        
    subplot_numbers[band] +=1
        
        
```

```python
noise_fnames = glob('/spt/user/adama/20190329_gainmatching/downsampled/*.g3')
```

```python
f_knee_dict = {}
for fname in noise_fnames:
    print(fname)
    fr = list(core.G3File(fname))[1]
    
    for jband, band in enumerate([90., 150., 220.]):
        for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                        'w181', 'w188', 'w203', 'w204', 'w206']):
            group = '{:.1f}_{}'.format(band, wafer)
            
            if band not in f_knee_dict:
                f_knee_dict[band] = {}
            if wafer not in f_knee_dict[band]:
                f_knee_dict[band][wafer] = []
                
            try:
                ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
                asd_diff = np.array(fr['AverageASDDiff'][group])
                par_diff = fr["AverageASDDiffFitParams"][group]
            
                f_knee = bisect(knee_func, a=0.01, b=1.0, args=tuple(par_diff))
                f_knee_dict[band][wafer].append(f_knee)
            except:
                pass
                
```

```python
for jband, band in enumerate(f_knee_dict):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, num=jband+1, figsize=(20,6))
    ax = ax.flatten()
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        _ = ax[jwafer].hist(np.asarray(f_knee_dict[band][wafer])*1e3,
                     bins=np.linspace(0,500,31),
                     label=wafer, histtype='step')
        ax[jwafer].set_title('{}, {} GHz'.format(wafer, int(band)))
    plt.xlim([0, 500])
    for j in [5,6,7,8,9]:
        ax[j].set_xlabel('1/f knee [mHz]')
    ax[0].set_ylabel('bolometers')
    ax[5].set_ylabel('bolometers')
    plt.savefig('figures_grid/fknee_noise_{}_{}.png'.format(int(band), wafer), dpi=200)
```

```python

```
