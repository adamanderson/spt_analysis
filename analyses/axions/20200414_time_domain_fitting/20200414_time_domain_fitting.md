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

# Testing Time-Domain Fitting
Let's do some preliminary tests of the time-domain fitting before we scale up the scripts.

```python
import numpy as np
import matplotlib.pyplot as plt
import timefittools
from importlib import reload
import pickle
from glob import glob
import os.path
from spt3g.std_processing import obsid_to_g3time
from spt3g import core
from datetime import datetime

reload(timefittools)
```

## Prototyping
Working up some code to put in a script to calculate upper limits.

```python
# make some fake data
pol_err = 1.5
nobs = 500
duration = 60*60*24*270
times = np.linspace(0, duration, nobs)
pol_angles = np.random.normal(loc=0, scale=1.5, size=nobs)
pol_angle_errs = np.array([pol_err for jobs in np.arange(nobs)])
data = {'times': times,
        'angles': pol_angles,
        'errs': pol_angle_errs}

plt.figure(figsize=(12,4))
plt.plot(times, pol_angles, marker='o', linewidth=0, markersize=4, color='C0')
plt.errorbar(times, pol_angles, yerr=pol_err, ls='none', color='C0')
plt.xlabel('observation time [sec]')
plt.ylabel('polarization rotation angle [deg]')
```

```python
# evaluate the 2D posterior for a specific frequency
freq = 1/(60*60*24*21)   # 1/(7 days)
n_amp_points = 50
n_phase_points = 50
phase_grid_2d, amp_grid_2d = np.meshgrid(np.linspace(0, 2*np.pi, n_phase_points),
                                         np.linspace(0, 1, n_amp_points))
amp_grid_1d = np.unique(amp_grid_2d)

posterior = np.zeros(n_amp_points * n_phase_points)
for jpoint, pair in enumerate(zip(np.hstack(phase_grid), np.hstack(amp_grid))):
    phase, amp = pair
    posterior[jpoint] = timefittools.posterior(amp, freq, phase, data)
posterior = np.reshape(posterior, newshape=(n_amp_points, n_phase_points))

posterior_marginal     = np.array([timefittools.posterior_marginal(amp, freq, data) \
                                   for amp in amp_grid_1d])
posterior_marginal_cdf = np.array([timefittools.posterior_marginal_cdf(amp, freq, data) \
                                   for amp in amp_grid_1d])
Acl = timefittools.upper_limit_bayesian(freq, data, 0.95, [1e-3, 1])
```

```python
plt.imshow(posterior, origin='lower', aspect='auto',
           extent=(np.min(phase_grid), np.max(phase_grid),
                   np.min(amp_grid), np.max(amp_grid)),)
```

```python
plt.plot(amp_grid_1d, posterior_marginal / np.max(posterior_marginal))
plt.plot(amp_grid_1d, posterior_marginal_cdf)
plt.plot([Acl, Acl], [0, 1], '--C2')
plt.plot([0, 1], [0.95, 0.95], '--C2')
plt.axis([0,1,0,1])
```

```python
plt.figure(figsize=(12,4))
plt.plot(times, pol_angles, marker='o', linewidth=0, markersize=4, color='C0')
plt.errorbar(times, pol_angles, yerr=pol_err, ls='none', color='C0')
plt.plot(times, timefittools.time_domain_model(Acl, freq, 0, times), color='C1')
plt.xlabel('observation time [sec]')
plt.ylabel('polarization rotation angle [deg]')
```

## Checking output from simulation script
Let's check the output of `simulate_time_domain.py`. This run simulated 1.5deg / observation pol angles for 500 observations over 270 days.

```python
sim_fnames = glob('/sptlocal/user/adama/axions/time_domain_fitting/timefit_test_full2/*.pkl')

upper_lims = {}
for fname in sim_fnames:
    try:
        with open(fname, 'rb') as f:
            ul_data = pickle.load(f)

        freqs = list(ul_data['results'][0]['upper limit'].keys())
        for freq in freqs:
            upper_lims[freq] = [ul_data['results'][jsim]['upper limit'][freq] \
                                for jsim in ul_data['results'].keys()]
    except:
        pass
```

```python
freqs = list(upper_lims.keys())
median_upper_lims = [np.median(upper_lims[freq]) for freq in freqs]

_ = plt.hist(median_upper_lims)
```

```python
plt.semilogx(freqs, median_upper_lims, 'o')
plt.xlabel('frequency [d$^{-1}$]')
plt.ylabel('95% CL upper limit on oscillation amplitude [deg]')
plt.ylim(0,0.5)
plt.tight_layout()
plt.savefig('amplitude_limit_1p5deg_err_500obs_270d.png', dpi=150)
```

## Change to use realistic observations times
Let's go find all the observation times from the real data in 2019 so that we can feed them into the sensitivity calculator.

```python
datapath = '/spt/data/bolodata/downsampled'
fields = ['ra0hdec-44.75', 'ra0hdec-52.25', 'ra0hdec-59.75', 'ra0hdec-67.25']

tstart = datetime(year=2019, month=3, day=21).timestamp()
tstop = datetime(year=2019, month=12, day=18).timestamp()

obsids = []
for field in fields:
    obs_paths = glob(os.path.join(datapath, field, '*'))
    field_obsids = np.array([float(os.path.basename(obs_path)) \
                             for obs_path in obs_paths])
    field_times = np.array([obsid_to_g3time(oid).time/core.G3Units.second \
                            for oid in field_obsids])
    obsids.extend(field_obsids[(field_times>tstart) & \
                               (field_times<tstop)])
obsids = np.sort(obsids)

with open('obsids_1500d_2019.pkl', 'wb') as f:
    pickle.dump(obsids, f)
```

Now let's look at some output from the grid simulation.

```python
sim_fnames = glob('/sptlocal/user/adama/axions/time_domain_fitting/timefit_2019_test/*.pkl')

upper_lims = {}
for fname in sim_fnames:
    with open(fname, 'rb') as f:
        ul_data = pickle.load(f)

    freqs = list(ul_data['results'][0]['upper limit'].keys())
    for freq in freqs:
        if freq not in upper_lims:
            upper_lims[freq] = []
        else:
            upper_lims[freq].extend([ul_data['results'][jsim]['upper limit'][freq] \
                                     for jsim in ul_data['results'].keys()])
    
freqs = list(upper_lims.keys())
median_upper_lims = [np.median(upper_lims[freq]) for freq in freqs]
```

```python
plt.semilogx(freqs, median_upper_lims, 'o')
plt.xlabel('frequency [d$^{-1}$]')
plt.ylabel('95% CL upper limit on oscillation amplitude [deg]')
plt.ylim(0,0.5)
plt.tight_layout()
plt.savefig('amplitude_limit_1p8deg_err_2019.png', dpi=150)
```

### Comparison against uniformly spaced observations

```python
sim_fnames = glob('/sptlocal/user/adama/axions/time_domain_fitting/timefit_2019_uniform_test/*.pkl')

upper_lims = {}
for fname in sim_fnames:
    with open(fname, 'rb') as f:
        ul_data = pickle.load(f)

    freqs = list(ul_data['results'][0]['upper limit'].keys())
    for freq in freqs:
        if freq not in upper_lims:
            upper_lims[freq] = []
        else:
            upper_lims[freq].extend([ul_data['results'][jsim]['upper limit'][freq] \
                                     for jsim in ul_data['results'].keys()])
    
freqs = list(upper_lims.keys())
median_upper_lims = [np.median(upper_lims[freq]) for freq in freqs]
```

```python
plt.semilogx(freqs, median_upper_lims, 'o')
plt.xlabel('frequency [d$^{-1}$]')
plt.ylabel('95% CL upper limit on oscillation amplitude [deg]')
plt.ylim(0,0.5)
plt.tight_layout()
plt.savefig('amplitude_limit_1p8deg_err_uniform_interval.png', dpi=150)
```

## Forecast Plot

```python
sim_fnames = glob('/sptlocal/user/adama/axions/time_domain_fitting/timefit_2019_test/*.pkl')

upper_lims = {}
for fname in sim_fnames:
    with open(fname, 'rb') as f:
        ul_data = pickle.load(f)

    freqs = list(ul_data['results'][0]['upper limit'].keys())
    for freq in freqs:
        if freq not in upper_lims:
            upper_lims[freq] = []
        else:
            upper_lims[freq].extend([ul_data['results'][jsim]['upper limit'][freq] \
                                     for jsim in ul_data['results'].keys()])
    
freqs = np.array(list(upper_lims.keys()))
m_axion_2019 = 2*np.pi*freqs/(60*60*24) * 6.528e-16
median_upper_lims_2019 = np.array([np.median(upper_lims[freq]) for freq in freqs])
```

```python
m_axion = np.logspace(-23, -18, 100)

# CAST solar axion limit
g_limit_cast = 6.6e-11 # from 1705.02290
plt.loglog(m_axion, g_limit_cast*np.ones(m_axion.shape))
plt.fill_between(m_axion, g_limit_cast*np.ones(m_axion.shape), 1, alpha=0.2)
plt.text(2e-19, 8e-11, s='CAST', color='C0')

# Planck "washout" limit
g_limit_planck = 9.6e-13 * (m_axion / 1e-21) # from equation (73) of 1903.02666
plt.loglog(m_axion, g_limit_planck)
plt.fill_between(m_axion, g_limit_planck, 1, alpha=0.2)
plt.text(1.3e-23, 2e-13, s='Planck "washout"',
             rotation=33, color='C1')

# small-scale structure (limit is very approximate)
m_limit_sss = 1e-22 # from 1610.08297
plt.loglog([m_limit_sss, m_limit_sss], [1e-14, 1e-9], '--', color='0.5')
plt.text(6e-23, 4e-11, s='small-scale structure', rotation=90)

# 3G
g_limit_2019 = median_upper_lims_2019 * (2*np.pi / 360) * 2 / 2.1e9 * \
                    (m_axion_2019/1e-21)
plt.loglog(m_axion_2019, g_limit_2019, '-C3')
plt.text(3.5e-21, 1e-10, s='SPT-3G 2019 (150 GHz) forecast',
             rotation=33, ha='center', color='C3')

plt.loglog(m_axion_2019, g_limit_2019 / np.sqrt(10), '--C3')
plt.text(5e-21, 2e-11, s='SPT-3G 5-year (95 + 150 GHz) forecast',
             rotation=33, ha='center', color='C3')

plt.axis([1e-23, 1e-18, 1e-14, 1e-9])
plt.xlabel('$m_a$ [eV]')
plt.ylabel('$g_{a\gamma}$ [GeV$^{-1}$]')
plt.tight_layout()
plt.savefig('axion_forecast.png', dpi=200)
```

```python


```
