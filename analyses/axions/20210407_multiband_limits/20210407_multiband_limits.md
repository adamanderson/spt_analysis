---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3 (v3)
    language: python
    name: python3-v3
---

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import timefittools
from glob import glob
from imp import reload
reload(timefittools)
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import chi2
import time
import os
```

```python
sims_fname = '/home/adama/SPT/spt3g_software/scratch/kferguson/axion_oscillation/timeseries_analysis/merged_angles.pkl'
with open(sims_fname, 'rb') as f:
    d = pickle.load(f)
```

```python
bands = [90, 150]
obsids      = {band: np.array([obsid for obsid in d if band in d[obsid]])
               for band in bands}
angle_errs  = {band: np.array([np.std(d[obsid][band]['noise'])
                              for obsid in d if band in d[obsid]])
               for band in bands}
mean_angles = {band: np.array([np.mean(d[obsid][band]['noise'])
                               for obsid in d if band in d[obsid]])
               for band in bands}
deltat      = {band: np.array([d[obsid]['tstop'] - d[obsid]['tstart']
                               for obsid in d if band in d[obsid]])
               for band in bands}
```

```python
for jband, band in enumerate(bands):
    plt.figure(jband+1)
    plt.errorbar(obsids[band], mean_angles[band], angle_errs[band], linestyle='none')
```

```python
# make fake datasets
angles = {90:[], 150:[]}

for obsid in d:
    for band in angles:
        if band in d[obsid]:
            angles[band].append(np.random.choice(d[obsid][band]['noise']))
```

```python
for jband, band in enumerate(bands):
    plt.figure(jband+1)
    plt.errorbar(obsids[band], np.asarray(angles[band]) * 180/np.pi,
                 np.asarray(angle_errs[band]) * 180/np.pi, linestyle='none')
```

## Checking the high-frequency behavior of the model

```python
# make fake datasets
angles = {90:[], 150:[]}

for obsid in d:
    for band in angles:
        if band in d[obsid]:
            angles[band].append(np.random.choice(d[obsid][band]['noise']) * 180/np.pi)
```

```python
data = {'times':  obsids,
        'angles': angles,
        'errs':   angle_errs,
        'deltat': deltat}
model = timefittools.TimeDomainModel(data)
```

```python
angle_osc = model.time_domain_model(1, 0.1/(60*60*24), 0, data['times'][90], data['deltat'][90])
plt.plot(data['times'][90], angle_osc, '.', label='0.1 d$^{-1}$')

angle_osc = model.time_domain_model(1, 1/(60*60*24), 0, data['times'][90], data['deltat'][90])
plt.plot(data['times'][90], angle_osc, '.', label='1.0 d$^{-1}$')

angle_osc = model.time_domain_model(1, 2/(60*60*24), 0, data['times'][90], data['deltat'][90])
plt.plot(data['times'][90], angle_osc, '.', label='2.0 d$^{-1}$')

angle_osc = model.time_domain_model(1, 6/(60*60*24), 0, data['times'][90], data['deltat'][90])
plt.plot(data['times'][90], angle_osc, '.', label='6.0 d$^{-1}$')

plt.xlabel('obsid')
plt.ylabel('pol. oscillation [deg.]')
plt.legend()
```

```python
frequencies = np.logspace(-1, 1)
observed_angles = []
for freq in frequencies:
    angle_obs = model.time_domain_model(1, freq/(60*60*24), 0, data['times'][90], data['deltat'][90])
    observed_angles.append(np.max(np.abs(angle_obs)))
    
plt.figure(1)
plt.semilogx(frequencies, observed_angles)
plt.xlabel('oscillation frequency [d$^{-1}$]')
plt.ylabel('reconstructed amp. / true amp.')
plt.tight_layout()
plt.grid()
plt.savefig('oscillation_bias.png', dpi=200)
```

# Checking Limits

```python
fname_pattern = '/sptgrid/user/adama/20210407_multiband_limits/test_limits2/' + \
                'sim_limits_*_amp=0.00000_freq=0.01000*.pkl'
fnames = np.sort(glob(fname_pattern))
amp_data = {}
amp_ul_data = {}

for fname in fnames:
    print(fname)
    with open(fname, 'rb') as f:
        d = pickle.load(f)
        
    for freq, amp in d[0]['A_fit'].items():
        if freq not in amp_data:
            amp_data[freq] = []
        amp_data[freq].append(float(amp) * 180/np.pi)
        
    for freq, result in d[0]['A_upperlimit'].items():
        if freq not in amp_ul_data:
            amp_ul_data[freq] = []
        amp_ul_data[freq].append(result[0] * 180/np.pi)
```

```python
freqs = np.array(list(amp_ul_data.keys()))
up2sigma_uls = np.array([np.percentile(amp_ul_data[freq], 97.5) for freq in amp_ul_data])
up1sigma_uls = np.array([np.percentile(amp_ul_data[freq], 84) for freq in amp_ul_data])
down1sigma_uls = np.array([np.percentile(amp_ul_data[freq], 16) for freq in amp_ul_data])
down2sigma_uls = np.array([np.percentile(amp_ul_data[freq], 2.5) for freq in amp_ul_data])
median_uls = np.array([np.mean(amp_ul_data[freq]) for freq in amp_ul_data])

upper_limit_data = {'median':median_uls,
                    '2sigma_low':down2sigma_uls,
                    '1sigma_low':down1sigma_uls,
                    '1sigma_high':up1sigma_uls,
                    '2sigma_high':up2sigma_uls}
with open('upper_limit_data.pkl', 'wb') as f:
    pickle.dump(upper_limit_data, f)

plt.fill_between(freqs, down2sigma_uls, up2sigma_uls,
                 color=(0.95,0.95,0), label='$\pm 2\sigma$')
plt.fill_between(freqs, down1sigma_uls, up1sigma_uls,
                 color=(0,0.9,0), label='$\pm 1\sigma$')
plt.semilogx(freqs, median_uls, 'k--',
             label='median 95% CL upper limit')
plt.ylim([0., 0.35])
plt.xlabel('frequency [d$^{-1}$]')
plt.ylabel('$Q/U$ rotation angle [deg]')
plt.legend()
plt.tight_layout()
plt.savefig('expected_limit_v0.png', dpi=200)
```

```python
fname_pattern = '/sptgrid/user/adama/20210407_multiband_limits/test_limits/' + \
                'sim_limits_*_amp=0.00000_freq=0.01000*.pkl'
fnames = np.sort(glob(fname_pattern))

d_all = []
for fname in fnames[:10]:
    print(fname)
    with open(fname, 'rb') as f:
        d = pickle.load(f)
        d_all.append(d)
```

```python
for data in d_all:
    print(data[0]['data']['angles'][150])
    ul = [data[0]['A_upperlimit'][freq][0] for freq in data[0]['A_upperlimit']]
    print(ul[:10])
```

```python
data[0]['A_upperlimit']
```

## Manual Testing

```python
sims_fname = '/home/adama/SPT/spt3g_software/scratch/kferguson/axion_oscillation/timeseries_analysis/merged_angles.pkl'
with open(sims_fname, 'rb') as f:
    d = pickle.load(f)
```

```python
bands = [90, 150]
obsids      = {band: np.array([obsid for obsid in d if band in d[obsid]])
               for band in bands}
angle_errs  = {band: np.array([np.std(d[obsid][band]['noise'])
                              for obsid in d if band in d[obsid]])
               for band in bands}
mean_angles = {band: np.array([np.mean(d[obsid][band]['noise'])
                               for obsid in d if band in d[obsid]])
               for band in bands}
deltat      = {band: np.array([d[obsid]['tstop'] - d[obsid]['tstart']
                               for obsid in d if band in d[obsid]])
               for band in bands}
```

```python
# make fake datasets
for j in range(1):
    angles = {90:[], 150:[]}

    for obsid in d:
        for band in angles:
            if band in d[obsid]:
                angles[band].append(np.random.choice(d[obsid][band]['noise']))

    data = {'times':  obsids,
            'angles': angles,
            'errs':   angle_errs,
            'deltat': deltat}
    model = timefittools.TimeDomainModel(data)
                
    result = model.bestfit_bayesian(0.1/(60*60*24), 0.2*np.pi/180)
    print(result.x*180/np.pi)
    result = model.bestfit_bayesian(0.154654/(60*60*24), 0.2*np.pi/180)
    print(result.x*180/np.pi)
    result = model.bestfit_bayesian(0.73453/(60*60*24), 0.2*np.pi/180)
    print(result.x*180/np.pi)
```

```python
data = {'times':  obsids,
        'angles': angles,
        'errs':   angle_errs,
        'deltat': deltat}
model = timefittools.TimeDomainModel(data)
model_interp = timefittools.TimeDomainModel(data, interp=True, amp_range=(0,0.5*np.pi/180))
```

```python
A_plot = np.linspace(0, 0.2)
posterior_plot = np.array([model.posterior_marginal(A*np.pi/180, 0.1/(60*60*24)) for A in A_plot])

plt.plot(A_plot, -2*np.log(posterior_plot))
```

```python
result = model.bestfit_bayesian(0.1/(60*60*24), 0.2*np.pi/180)
print(result.x*180/np.pi)
```

```python
start_time = time.clock()
result = model.upper_limit_bayesian(0.1/(60*60*24), 0.95)
print(result[0]*180/np.pi)
print(time.clock() - start_time, "seconds")
```

```python
start_time = time.clock()
result = model_interp.upper_limit_bayesian(0.1/(60*60*24), 0.95)
print(result[0]*180/np.pi)
print(time.clock() - start_time, "seconds")
```

## Check background-only distributions

```python
fname = '/home/adama/SPT/spt_analysis/analyses/axions/20210407_multiband_limits/condor/sim_results.pkl'

with open(fname, 'rb') as f:
    bkg_data = pickle.load(f)
```

```python
fit_freq = 0.01
A_bayesian = np.array([bkg_data[j]['A_fit_bayesian'][fit_freq] for j in bkg_data])
A_ml       = np.array([bkg_data[j]['A_fit'][fit_freq].x[0] for j in bkg_data])
chi2       = np.array([bkg_data[j]['A_fit'][fit_freq].fun for j in bkg_data])
chi2_0     = np.array([bkg_data[j]['chi2(A=0)'][fit_freq] for j in bkg_data])
delta_chi2 = chi2_0 - chi2
```

```python
plt.hist(A_bayesian)
plt.hist(A_ml)
```

```python
_ = plt.hist(delta_chi2, density=True)
x = np.linspace(0,10)
plt.plot(x, chi2.pdf(x, df=1))
plt.plot(x, chi2.pdf(x, df=2))
plt.plot(x, chi2.pdf(x, df=3))
```

```python
plt.plot(A_bayesian, A_ml, '.')
```

## Scratch

```python
datadir = '/sptgrid/user/adama/20210407_multiband_limits/test_limits_bkg_only'
fnames = np.sort(glob(os.path.join(datadir, 'sim_bkg_only_fitfreq=0.01000-2.00000_amp=0.00000_freq=0.01000_*.pkl')))
```

```python
Afit = {}
Afit_bayesian = {}
chi2_0 = {}
delta_chi2 = {}

for fn in fnames[:1000]:
    with open(fn, 'rb') as f:
        d = pickle.load(f)

    for j in d:
        for freq in d[j]['A_fit']:
            if freq not in Afit:
                Afit[freq] = []
                Afit_bayesian[freq] = []
                chi2_0[freq] = []
                delta_chi2[freq] = []
            Afit[freq].append(d[j]['A_fit'][freq].x[0])
            Afit_bayesian[freq].append(d[j]['A_fit_bayesian'][freq])
            chi2_0[freq].append(d[j]['chi2(A=0)'][freq])
            delta_chi2[freq].append(d[j]['chi2(A=0)'][freq] - d[j]['A_fit'][freq].fun)
```

```python
_ = plt.hist(Afit[list(Afit.keys())[400]])
_ = plt.hist(Afit_bayesian[list(Afit.keys())[400]])
```

```python
_ = plt.hist(delta_chi2[list(Afit.keys())[400]],
             bins=np.linspace(0,12,25), density=True)
x = np.linspace(0,10)
plt.plot(x, chi2.pdf(x, df=1), label='$\chi^2($ndf$=1)$')
plt.plot(x, chi2.pdf(x, df=2), label='$\chi^2($ndf$=2)$')
plt.plot(x, chi2.pdf(x, df=3), label='$\chi^2($ndf$=3)$')
plt.legend()
```

```python
d[3].keys()
```

```python
bkg_data[0].keys()
```

```python
len(delta_chi2[list(Afit.keys())[400]])
```

```python
fn
```

```python
d[0].keys()
```

```python

```
