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
from spt3g.calibration.template_groups import get_template_groups

%matplotlib inline
```

```python
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

## Fitting pair-summed vs. pair-differenced ASDs

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

## Calculating $f_{knee}$ for many noise stares processed on grid

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
from scipy.stats import cumfreq
for jband, band in enumerate(f_knee_dict):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, num=jband+1, figsize=(15,4))
    ax = ax.flatten()
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        n, bins, _ = ax[jwafer].hist(np.asarray(f_knee_dict[band][wafer])*1e3,
                              bins=np.linspace(0,500,31),
                              label=wafer, histtype='step', color='C0')
        ax2 = ax[jwafer].twinx()
        cdf = np.array([np.sum(n[:end]) for end in range(len(n))])
        
        ax2.step(bins[:-1], cdf / len(f_knee_dict[band][wafer]), color='C1')
        ax2.set_ylim([0, 1.0])
        ax[jwafer].set_title('{}, {} GHz'.format(wafer, int(band)))
        
        if jwafer==4 or jwafer==9:
            ax2.set_ylabel('empirical CDF')
            
    plt.xlim([0, 500])
    for j in [5,6,7,8,9]:
        ax[j].set_xlabel('1/f knee [mHz]')
    ax[0].set_ylabel('bolometers')
    ax[5].set_ylabel('bolometers')
    
    plt.tight_layout()
    plt.savefig('figures_grid/fknee_noise_{}.png'.format(int(band), wafer), dpi=100)
```

## Gain-matching coefficients
How close to 1 are the new gain-matching coefficients? In other words, how bad is an "optimized" calibration relative to our nominal fast-point calibration?

```python
fr = list(core.G3File('/spt/user/adama/20190329_gainmatching/downsampled/gainmatching_noise_73124800.g3'))[1]

gainmatch_coeffs = [fr['GainMatchCoeff'][bolo] for bolo in fr['GainMatchCoeff'].keys()]
_ = plt.hist(gainmatch_coeffs, bins=np.linspace(0.7,1.3,101), histtype='step')
plt.xlabel('gain-matching coefficient')
plt.ylabel('bolometers')
plt.title('noise stare 73124800')
plt.tight_layout()
plt.savefig('figures_grid/gain_matching_coeffs_73124800.png', dpi=150)
```

## Horizon Noise Stare

```python
fr = list(core.G3File('horizon_noise_77863968.g3'))[1]
fr_poly1 = list(core.G3File('horizon_noise_77863968_poly1.g3'))[1]
print(fr)
```

```python
plt.loglog(fr["AverageASD"]['frequency'] / core.G3Units.Hz,
           fr['AverageASD']['90.0_w172'])
plt.loglog(fr["AverageASDDiff"]['frequency'] / core.G3Units.Hz,
           fr['AverageASDDiff']['90.0_w172']/np.sqrt(2.))
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
        ff = np.array(fr['AverageASD']['frequency']/core.G3Units.Hz)
        ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
        ff_poly1 = np.array(fr_poly1['AverageASD']['frequency']/core.G3Units.Hz)
        asd = np.array(fr['AverageASD'][group])
        asd_diff = np.array(fr['AverageASDDiff'][group]) / np.sqrt(2.)
        asd_poly1 = np.array(fr_poly1['AverageASD'][group])

        par = fr_poly1["AverageASDFitParams"][group]
#         ax[jwafer].loglog(ff, asd, label='all bolos (poly 0)')
        ax[jwafer].loglog(ff, asd, label='all bolos (poly 1)')
        ax[jwafer].loglog(ff_diff, asd_diff, label='(x - y) / \sqrt{2}')
        try:
            ax[jwafer].loglog(ff, horizon_model(ff, *list(par)), 'k--')
        except:
            pass

        ax[jwafer].set_title('{}, {} GHz, white noise = {:.1f} pA$ / \sqrt{{Hz}}$'.format(group.split('_')[1],
                                                                       int(float(group.split('_')[0])),
                                                                       np.mean(asd[(ff>10) & (ff<15)])))
        try:
            f_knee = bisect(horizon_knee_func, a=0.001, b=1.0, args=tuple(par))
            ax[jwafer].set_title('{}, {} GHz\nwhite noise = {:.1f} '
                                 'pA$ / \sqrt{{Hz}}$ '
                                 '$f_{{knee}}^{{all}}$ = {:.3f}'.format(group.split('_')[1],
                                                             int(float(group.split('_')[0])),
                                                             np.mean(asd[(ff>10) & (ff<15)]),
                                                                         f_knee))
        except:
            ax[jwafer].set_title('{}, {} GHz\nwhite noise = {:.1f} '
                                 'pA$ / \sqrt{{Hz}}$'.format(group.split('_')[1],
                                                             int(float(group.split('_')[0])),
                                                             np.mean(asd[(ff>10) & (ff<15)])))
            
    for jwafer in [5,6,7,8,9]:
        ax[jwafer].set_xlabel('frequency [Hz]')
    
    ax[0].set_ylabel('NEI [pA$ / \sqrt{Hz}$]')
    ax[5].set_ylabel('NEI [pA$ / \sqrt{Hz}$]')
    plt.ylim([1,1000])
    plt.legend()
    plt.tight_layout()
        
    subplot_numbers[band] +=1
        
for band, jplot in band_numbers.items():
    plt.figure(jplot)
    plt.savefig('figures_grid/horizon_noise_{}_77863968.png'.format(int(band)), dpi=120)
```

```python
band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}


A_sqrt_all = []
alpha_all = []
whitenoise_all = []
# plt.figure(1)
fig1, ax = plt.subplots(1, 3, sharex=False, sharey=True, num=1, figsize=(12,4))
for jband, band in enumerate([90., 150., 220.]):
    A_sqrt = []
    alpha = []
    whitenoise = []
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)

        par = fr_poly1["AverageASDFitParams"][group]
        
        whitenoise.append(np.sqrt(par[0]))
        A_sqrt.append(np.sqrt(par[1]))
        alpha.append(par[2])
        
        A_sqrt_all = np.hstack([A_sqrt_all, A_sqrt])
        alpha_all = np.hstack([alpha_all, alpha])
        whitenoise_all = np.hstack([whitenoise_all, whitenoise])
        
    _ = ax[0].hist(whitenoise, bins=np.linspace(5,25,21), alpha=0.5,
                   label='{} GHz, median = {:.2f}'.format(band, np.median(whitenoise)))
    ax[0].set_xlabel('white noise [pA/rtHz]')
    ax[0].set_ylabel('groups')
    _ = ax[1].hist(A_sqrt, bins=np.linspace(0, 3, 16), alpha=0.5,
                   label='{} GHz, median = {:.2f}'.format(band, np.median(A_sqrt)))
    ax[1].set_xlabel('$\sqrt{A}$ [pA/rtHz]')
    _ = ax[2].hist(alpha, bins=np.linspace(0,4,16), alpha=0.5,
                   label='{} GHz, median = {:.2f}'.format(band, np.median(alpha)))
    ax[2].set_xlabel('$\\alpha$')
ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.savefig('figures_grid/horizon_noise_params_77863968.png', dpi=120)

# plt.figure(2)
fig2, ax = plt.subplots(1, 3, sharex=False, sharey=True, num=2, figsize=(12,4))
_ = ax[0].hist(whitenoise_all, bins=np.linspace(5,25,21), alpha=0.5)
ax[0].set_xlabel('white noise [pA/rtHz]')
ax[0].set_ylabel('groups')
_ = ax[1].hist(A_sqrt_all, bins=np.linspace(0, 3, 16), alpha=0.5)
ax[1].set_xlabel('$\sqrt{A}$ [pA/rtHz]')
_ = ax[2].hist(alpha_all, bins=np.linspace(0,4,16), alpha=0.5)
ax[2].set_xlabel('$\\alpha$')

plt.tight_layout()


```

### Noise-to-carrier ratio

```python
# plots for split by squid
d = list(core.G3File('horizon_noise_77863968_poly1_squid_split.g3'))
```

```python
A_sqrt = []
Irms = []
alpha = []
for g in d[1]["AvgCurrent"].keys():
    par = d[1]["AverageASDFitParams"][g]
    A_sqrt.append(np.sqrt(par[1]))
    Irms.append(d[1]["AvgCurrent"][g])
    alpha.append(par[2])
A_sqrt = np.array(A_sqrt)
Irms = np.array(Irms)
NCR = A_sqrt / (Irms*1e12)
alpha = np.array(alpha)
```

```python
plt.figure(figsize=(12,3))
plt.subplot(1,4,1)
_ = plt.hist(A_sqrt, bins=np.linspace(0,3,16))
plt.xlabel('$\sqrt{A}$ [pA/$\sqrt{Hz}]$')
plt.subplot(1,4,2)
_ = plt.hist(Irms*1e6, bins=np.linspace(0,3,16))
plt.xlabel('average rms TES current [$\mu$A]')
ax = plt.subplot(1,4,3)
_ = plt.hist(NCR, bins=np.linspace(0,1.5e-6,16))
ax.ticklabel_format(axis='x', style='sci')
plt.xlabel('noise-to-carrier ratio [1/$\sqrt{Hz}$]')
plt.subplot(1,4,4)
_ = plt.hist(alpha, bins=np.linspace(0,4,16))
plt.xlabel('$\\alpha$')

plt.tight_layout()
plt.savefig('figures_grid/NCR_squid_split.png', dpi=200)
```

```python
# split by wafer/band
d = list(core.G3File('horizon_noise_77863968_poly1_squidband_split.g3'))
```

```python
A_sqrt = []
Irms = []
alpha = []
for g in d[1]["AvgCurrent"].keys():
    par = d[1]["AverageASDFitParams"][g]
    A_sqrt.append(np.sqrt(par[1]))
    Irms.append(d[1]["AvgCurrent"][g])
    alpha.append(par[2])
A_sqrt = np.array(A_sqrt)
Irms = np.array(Irms)
NCR = A_sqrt / (Irms*1e12)
alpha = np.array(alpha)
```

```python
plt.figure(figsize=(12,3))
plt.subplot(1,4,1)
_ = plt.hist(A_sqrt, bins=np.linspace(0,3,16))
plt.xlabel('$\sqrt{A}$ [pA/$\sqrt{Hz}]$')
plt.subplot(1,4,2)
_ = plt.hist(Irms*1e6, bins=np.linspace(0,3,16))
plt.xlabel('average rms TES current [$\mu$A]')
ax = plt.subplot(1,4,3)
_ = plt.hist(NCR, bins=np.linspace(0,1.5e-6,16))
ax.ticklabel_format(axis='x', style='sci')
plt.xlabel('noise-to-carrier ratio [1/$\sqrt{Hz}$]')
plt.subplot(1,4,4)
_ = plt.hist(alpha, bins=np.linspace(0,4,16))
plt.xlabel('$\\alpha$')

plt.tight_layout()
plt.savefig('figures_grid/NCR_squidband_split.png', dpi=200)
```

### readout noise plots for Amy's LTD proceedings

```python
fr = list(core.G3File('horizon_noise_77863968_bender_ltd.g3'))[1]
```

```python
print(fr)
```

```python
band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}
band_labels = {90:'95', 150:'150', 220:'220'}

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,4))
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

    ax[jband].set_title('{} GHz'.format(band_labels[band]))
    ax[jband].set_xlabel('frequency [Hz]')
    ax[jband].grid()
ax[0].set_ylabel('current noise [pA/$\sqrt{Hz}$]')

plt.ylim([5,1000])
plt.legend()
plt.tight_layout()
plt.savefig('w180_horizon_noise_ltd.pdf')
```

## Are we scanning fast enough?


Let's adopt a very crude model of the atmosphere, in which the atmosphere consists of blobs at characteristic angular scale $\Delta \theta$, which move at an angular velocity $\vec{\omega}_{a}$. Assume further that the telescope is scanning at an angular velocity $\vec{\omega}_{s}$. The relative velocity of the blobs of atmosphere across the focal plane is therefore $\vec{\omega} = \vec{\omega}_{s} + \vec{\omega}_{a}$. The square norm of the angular velocity is therefore
$$
\left| \vec{\omega} \right|^2 = \left| \vec{\omega}_{s} \right|^2 + \left| \vec{\omega}_{a} \right|^2 + 2\left| \vec{\omega}_{s} \right| \left| \vec{\omega}_{a} \right| \cos \alpha.
$$
Next assume that the angle between the scan direction and the atmosphere velocity is uniformly distributed over many observations. This assumption should actually be very good since the atmosphere is usually moving in the same direction while our scan direction moves in a circle. The time-average of the square velocity is therefore just:
$$
\left| \vec{\omega} \right|^2 = \left| \vec{\omega}_{s} \right|^2 + \left| \vec{\omega}_{a} \right|^2.
$$
A blob of atmosphere of size $\Delta \theta$ appears at a characteristic frequency squared of 
$$
f^2(\omega_s, \omega_a) = \frac{\omega_s^2 + \omega_a^2}{\Delta \theta^2}.
$$

Assuming that the 1/f knee frequency corresponds roughly to the angular scale of blobs of atmosphere on the sky, from our field scans and our noise stares, we have measurements of $f(\omega_s=0, \omega_a)$ and $f(\omega_s\simeq1\textrm{deg/s}, \omega_a)$. We cannot estimate the $1/\ell$ knee from just these data, but we can estimate how close we are to our best achievable $1/\ell$ knee. By definition, the best possible $1/\ell$ knee occurs when we take $\omega_a = 0$ (i.e. atmosphere is comoving with the telescope scan), so define the figure of merit ratio:
$$
r = \frac{f^2(\omega_s, \omega_a)}{f^2(\omega_s, \omega_a=0)} = 1 + \frac{\omega_a^2}{\omega_s^2}
$$
When $r=1$, our 1/f knee is what it would be in the absence of atmospheric fluctuations, and $r>1$ corresponds to a situation that could be improved by scanning faster. Note that $r$ is not just a ratio of 1/f knees, it is also the ratio of $1/\ell$ knees, since the scan speed divides out in the ratio. We measure the numerator, but not the denominator, of $r$ in field scans. In noise stares, we also measure $f(\omega_s=0, \omega_a)$. With some algebra, we can solve for $r$. Define
$$
s = \frac{f^2(\omega_s, \omega_a)}{f^2(\omega_s=0, \omega_a)} = \frac{\omega_s^2 + \omega_a^2}{\omega_a^2} = 1 + \frac{\omega_s^2}{\omega_a^2}.
$$
So
$$
\frac{\omega_a^2}{\omega_s^2} = \frac{1}{s - 1},
$$
and
$$
r = 1 + \frac{1}{s-1} = \frac{s}{s-1}.
$$
I emphasize again that $s$ is a quantity that we measure: it is the 1/f knee of field scans divided by the 1/f knee of noise stares. The intuition here is also clear: if our 1/f knee in field scans is *way* higher than noise stares, then $s$ is large and atmosphere is changing much more slowly than we are scanning. That makes $r$ close to 1, so we have little room to improve by scanning yet faster.


## 1/f in Field Scans

```python
fr = list(core.G3File('gainmatching_ra0hdec-67.25_76707699.g3'))
```

```python
print(fr[20])
```

```python

```
