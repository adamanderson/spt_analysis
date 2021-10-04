---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import timefittools
from scipy.optimize import minimize
from glob import glob
import os
from scipy.signal import periodogram
```

```python
pager_fig_dir = '/sptlocal/user/adama/public_html/spt_notes/20210120_limit_tests/figures_save'
```

## Quick tests

```python
with open('sim_results.pkl', 'rb') as f:
    d = pickle.load(f)
```

```python
amps = [d['results'][j]['min_chi2']['x'][0] for j in d['results']]
```

```python
_ = plt.hist(amps, np.linspace(0,4))
```

```python
result = timefittools.minimize_chi2(d['results'][1]['data'])
```

```python
plt.figure(figsize=(10,4))
markerline, stemlines, baseline = plt.stem(d['results'][1]['data']['times'], d['results'][1]['data']['angles'],
                                           basefmt='none', label='fake data (amp = 1 deg)')
markerline.set_markerfacecolor('none')
for ll in stemlines: ll.set_linewidth(0.5)

plot_times = np.linspace(np.min(d['results'][1]['data']['times']),
                         np.max(d['results'][1]['data']['times']), 1000)
plt.plot(plot_times, timefittools.time_domain_model(*result.x, plot_times),
         'C3', label='sinusoidal fit')

plt.xlabel('observation ID')
plt.ylabel('polarization rotation angle [deg]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(pager_fig_dir, 'fake_tod_example.png'), dpi=200)
```

```python
posterior_fun = lambda A: -2*np.log(timefittools.posterior_marginal(A, 0.100000000000001/(60*60*24),
                                                             d['results'][1]['data'],
                                                             d['results'][1]['min_chi2']['fun'], simpson=False))
result = minimize(posterior_fun, 1.0, method='Powell')
```

```python
ul = timefittools.upper_limit_bayesian(0.1 / (60*60*24),
                                    d['results'][1]['data'], 0.95,
                                    [0.5, 2], d['results'][1]['min_chi2']['fun'])
```

```python
ul
```

```python
A_plot = np.linspace(0.75, 1.25, 100)
posterior_plot = [timefittools.posterior_marginal(A, 0.100000000000001/(60*60*24),
                  d['results'][1]['data'],
                  d['results'][1]['min_chi2']['fun']) for A in A_plot]
# plt.plot(A_plot, -2*np.log(posterior_plot), 'o-')
plt.plot(A_plot, posterior_plot, '-')
plt.xlabel('amplitude ($A$) [deg]')
plt.ylabel('$P(A)$')
plt.tight_layout()
plt.savefig(os.path.join(pager_fig_dir, 'amp_posterior_example.png'), dpi=200)
# plt.plot([float(result.x), float(result.x)], plt.gca().get_ylim())
# plt.plot([ul, ul], plt.gca().get_ylim())
```

```python
for jsim in d['results']:
    freqs = list(d['results'][jsim]['A_fit'].keys())
    A_fits = list(float(d['results'][jsim]['A_fit'][freq]) for freq in d['results'][jsim]['A_fit'])

    plt.plot(freqs, np.abs(A_fits))
plt.xlabel('frequency [d^{-1}]')
plt.ylabel('best fit amplitude [deg.]')
```

```python
with open('sim_results_0p01_to_1p0_invdays.pkl', 'rb') as f:
    d = pickle.load(f)
```

```python
for jsim in d['results']:
    freqs = list(d['results'][jsim]['A_fit'].keys())
    A_fits = list(float(d['results'][jsim]['A_fit'][freq]) for freq in d['results'][jsim]['A_fit'])

    plt.plot(freqs, np.abs(A_fits))
plt.xlabel('frequency [d^{-1}]')
plt.ylabel('best fit amplitude [deg.]')
```

## Dense scans

```python
datadir = '/sptlocal/user/adama/axions/20210103_massbinning'
freq_ranges = ['0.005-0.015', '0.095-0.105', '0.995-1.005']
freqs = ['0.010', '0.100', '1.000']
signal_amp = '1.000'
best_fit_amp = {}

for freq_range, freq in zip(freq_ranges, freqs):
    fnames = glob(os.path.join(datadir, 'massbinning_fitfreq={}_amp={}_freq={}-*.pkl'.format(freq_range, signal_amp, freq)))
    best_fit_amp[freq] = {}
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            for jexpt in d['results']:
                for fitfreq, amp in d['results'][jexpt]['A_fit'].items():
                    if fitfreq not in best_fit_amp[freq]:
                        best_fit_amp[freq][fitfreq] = []
                    best_fit_amp[freq][fitfreq].append(float(amp))
                    
```

```python
for jfreq, freq in enumerate(best_fit_amp):
    fitfreqs = list(best_fit_amp[freq].keys())
    mean_amp_fit = np.array([np.mean(best_fit_amp[freq][fitfreq]) for fitfreq in best_fit_amp[freq]])
    down1sigma_amp_fit = np.array([np.percentile(best_fit_amp[freq][fitfreq], 16) for fitfreq in best_fit_amp[freq]])
    up1sigma_amp_fit = np.array([np.percentile(best_fit_amp[freq][fitfreq], 84) for fitfreq in best_fit_amp[freq]])
    
    plt.figure(jfreq)
    plt.gca().fill_between(fitfreqs, down1sigma_amp_fit, up1sigma_amp_fit, alpha=0.5, color='C2')
    plt.plot(fitfreqs, mean_amp_fit)
    plt.xlabel('frequency [1/d]')
    plt.ylabel('amplitude [deg]')
    plt.title('best-fit amplitude for {} deg signal at {} 1/d'.format(signal_amp, freq))
    plt.tight_layout()
    plt.savefig(os.path.join(pager_fig_dir, 'mass_binning_{}.png'.format(freq)), dpi=200)
```

```python
(max(d['results'][1]['data']['times']) - min(d['results'][1]['data']['times'])) / (24*60*60)
```

```python
1/270.
```

## Sparse Scans

```python
datadir = '/sptgrid/user/adama/20210103_massbinning'
freq_range = '0.010-2.000'
freqs = ['0.010', '0.452', '0.894', '1.337', '1.558', '1.779', '2.000']
signal_amp = '1.000'
best_fit_amp = {}

for freq in freqs:
    fnames = glob(os.path.join(datadir, 'massbinning_test_fitfreq={}_amp={}_freq={}-*.pkl'.format(freq_range, signal_amp, freq)))
    best_fit_amp[freq] = {}
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            for jexpt in d['results']:
                for fitfreq, amp in d['results'][jexpt]['A_fit'].items():
                    if fitfreq not in best_fit_amp[freq]:
                        best_fit_amp[freq][fitfreq] = []
                    best_fit_amp[freq][fitfreq].append(float(amp))
```

```python
fitfreqs = list(best_fit_amp['0.894'].keys())
fitfreq = fitfreqs[20]
fit_amplitudes = best_fit_amp['0.894'][fitfreq]
_ = plt.hist(fit_amplitudes, bins=np.linspace(-1, 1, 100))
```

```python
for jfreq, freq in enumerate(best_fit_amp):
    fitfreqs = list(best_fit_amp[freq].keys())
    mean_amp_fit = np.array([np.mean(best_fit_amp[freq][fitfreq]) for fitfreq in best_fit_amp[freq]])
    down1sigma_amp_fit = np.array([np.percentile(best_fit_amp[freq][fitfreq], 16) for fitfreq in best_fit_amp[freq]])
    up1sigma_amp_fit = np.array([np.percentile(best_fit_amp[freq][fitfreq], 84) for fitfreq in best_fit_amp[freq]])
    
    plt.figure(jfreq)
    plt.gca().fill_between(fitfreqs, down1sigma_amp_fit, up1sigma_amp_fit, alpha=0.5, color='C2')
    plt.plot(fitfreqs, mean_amp_fit)
    plt.xlabel('frequency [1/d]')
    plt.ylabel('amplitude [deg]')
    plt.title('best-fit amplitude for {} deg signal at {} 1/d'.format(signal_amp, freq))
    plt.tight_layout()
    plt.savefig(os.path.join(pager_fig_dir, 'sparse_scan_{}.png'.format(freq)), dpi=200)
```

## Transfer function

```python
datadir = '/sptgrid/user/adama/20210103_massbinning'
signal_amp = '1.000'
best_fit_amp = {}


fnames = glob(os.path.join(datadir, 'transfer_function*.pkl'))
best_fit_amp = {}

for fname in fnames:
    with open(fname, 'rb') as f:
        d = pickle.load(f)
        for jexpt in d['results']:
            for fitfreq, amp in d['results'][jexpt]['A_fit'].items():
                if fitfreq not in best_fit_amp:
                    best_fit_amp[fitfreq] = []
                best_fit_amp[fitfreq].append(float(amp))
```

```python
fitfreqs = np.sort(list(best_fit_amp.keys()))
mean_amp_fit = np.array([np.mean(best_fit_amp[fitfreq]) for fitfreq in best_fit_amp])
down1sigma_amp_fit = np.array([np.percentile(best_fit_amp[fitfreq], 16) for fitfreq in best_fit_amp])
up1sigma_amp_fit = np.array([np.percentile(best_fit_amp[fitfreq], 84) for fitfreq in best_fit_amp])

plt.figure(jfreq)
plt.gca().fill_between(fitfreqs, down1sigma_amp_fit, up1sigma_amp_fit, alpha=0.5, color='C2')
plt.plot(fitfreqs, mean_amp_fit, '-')
plt.gca().set_xscale('log')
plt.xlabel('frequency [1/d]')
plt.ylabel('amplitude [deg]')
plt.title('best-fit amplitude for {} deg signal at {} 1/d'.format(signal_amp, freq))
plt.tight_layout()
plt.savefig('transfer_function.png', dpi=200)
```

## Scratch

```python
datadir = '/sptgrid/user/adama/20210103_massbinning'
freq_range = '0.01000-2.00000'
freqs = ['0.01000'] #, '0.452', '0.894', '1.337', '1.558', '1.779', '2.000']
signal_amp = '0.00000'
best_fit_amp = {}

for freq in freqs:
    fnames = glob(os.path.join(datadir,
                               'massbinning_test_ul_fitfreq={}_amp={}_freq={}-*.pkl'.format(freq_range, signal_amp, freq)))

    best_fit_amp[freq] = {}
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            for jexpt in d['results']:
                for fitfreq, amp in d['results'][jexpt]['A_fit'].items():
                    if fitfreq not in best_fit_amp[freq]:
                        best_fit_amp[freq][fitfreq] = []
                    best_fit_amp[freq][fitfreq].append(float(amp))
```

```python
d['results'][0]['A_upperlimit'][0.01][0]
```

```python
400*10/3600
```

```python
2 / 5e-4
```

```python
times = np.linspace(0,270,1549)*24*3600
pol_angles = np.random.normal(loc=0, scale=1.5, size=1549)
pol_angle_errs = np.array([1.5 for jobs in np.arange(1549)])

data = {'times': times,
        'deltat': 1*np.ones(len(times)),
        'angles': pol_angles,
        'errs': pol_angle_errs}
model = timefittools.TimeDomainModel(data)

Afits = {}
Acls = {}
A_test = np.linspace(0, 3, 20)
freq = 0.1 / (60*60*24)
def posterior(A):
    if A < 0:
        return 99999
    else:
        return -2*np.log(model.posterior_marginal(A, freq))
result = minimize(posterior, 0.1, method='Powell')
Afits[freq] = result['x']

Aplot = np.linspace(0, 0.2, 100)
posterior_plot = [posterior(A) for A in Aplot]
plt.plot(Aplot, posterior_plot)
plt.title('{}'.format(Afits[freq]))

```

```python
Aplot = np.linspace(0,0.2)
phiplot = np.linspace(0,2*np.pi,200)
                 
theta,rad = np.meshgrid(phiplot, Aplot) #rectangular plot of polar data
X = theta
Y = rad

data2D = model.posterior(Y, freq, X)

fig = plt.figure()
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, polar='False')
ax.pcolormesh(X, Y, data2D) #X,Y & data2D must all be same dimensions

```

```python
plt.plot(Aplot, np.sum(data2D, axis=1))
```

```python
data
```

```python
(np.cos(3) - np.cos(3.000001)) / 0.000001
```

```python

```
