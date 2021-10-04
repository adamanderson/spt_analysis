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
import numpy as np
import matplotlib.pyplot as plt
import pickle
from timefittools import TimeDomainModel
from spt3g import core
```

```python
with open('obsids_1500d_2019_byfield.pkl', 'rb') as f:
    obsids = pickle.load(f)
    for field in obsids:
        obsids[field] = np.array(obsids[field])
```

```python
obsids.keys()
```

```python
for jfield, field in enumerate(list(obsids.keys())):
    freq = 1. / (60*60*24 * 2)
    angles = 0.1*np.pi/180*np.sin(2*np.pi*obsids[field]*freq) + \
            np.random.normal(loc=0, scale=0.01*np.pi/180, size=len(obsids[field]))
    
    data = {'times': {150: obsids[field]},
            'angles': {150: angles},
            'errs': {150: 0.1*np.pi/180*np.ones(len(angles))}}
    model = TimeDomainModel(data)
    result = model.minimize_chi2(freq)
    print(result.x)
    
    plt.figure(jfield+1, figsize=(12,4))
    plt.plot(obsids[field], angles, '.-')
    plt.plot()
    plt.title(field)
```

```python
for jfield, field in enumerate(list(obsids.keys())):
    freq = 1. / (60*60*24 * 2)
    angles = np.random.normal(loc=0, scale=0.01*np.pi/180, size=len(obsids[field]))
    
    data = {'times': {150: obsids[field]},
            'angles': {150: angles},
            'errs': {150: 0.1*np.pi/180*np.ones(len(angles))}}
    model = TimeDomainModel(data)
    result = model.minimize_chi2(freq)
    print(result.x)
    
    plt.figure(jfield+1, figsize=(12,4))
    plt.plot(obsids[field], angles, '.-')
    plt.plot()
    plt.title(field)
```

```python
Afit = {field:[] for field in obsids}
for jexp in range(10000):
    for jfield, field in enumerate(list(obsids.keys())):
        freq = 1. / (60*60*24 * 2)
        angles = np.random.normal(loc=0, scale=0.01*np.pi/180, size=len(obsids[field]))

        data = {'times': {150: obsids[field]},
                'angles': {150: angles},
                'errs': {150: 0.1*np.pi/180*np.ones(len(angles))}}
        model = TimeDomainModel(data)
        result = model.minimize_chi2(freq)
        Afit[field].append(result.x[0])
        
```

```python
for jfield, field in enumerate(list(obsids.keys())):
    plt.hist(Afit[field], histtype='step', bins=np.linspace(0,6e-5,51), label=field)
plt.legend()
plt.title('freq = {:.2f} day$^{{-1}}$'.format(freq*(60*60*24)))
plt.xlabel('amplitude')
```

## DC Offset Issues?

```python
from scipy.signal import lombscargle
```

```python
freqs = np.linspace(0.0001,2,1000)
periodogram = lombscargle(obsids['ra0hdec-44.75'] / (60*60*24),
            np.random.normal(loc=0, scale=1, size=len(obsids['ra0hdec-44.75'])),
            freqs)
```

```python
plt.plot(freqs, periodogram)
```

```python
freqs = np.linspace(0.0001,10,1000)
plt.figure(figsize=(10,8))
for jfield, field in enumerate(obsids.keys()):
    plt.subplot(2,2,jfield+1)
    freq = 1. / (60*60*24 * 3)
    angles = 1. * np.ones(len(obsids[field])) #0.1*np.pi/180*np.sin(2*np.pi*obsids[field]*freq)
    periodogram = lombscargle(obsids[field] / (60*60*24),
                              angles, freqs)
    plt.plot(freqs / (2*np.pi), np.sqrt(periodogram))
    plt.xlabel('frequency [1/d]')
    plt.ylabel('sqrt(L-S periodogram) [arb.]')
    plt.title('constant DC offset for {}'.format(field))
    plt.tight_layout()
plt.savefig('dc_offsets.png', dpi=200)
```

## Jackknife angle distributions

```python
ls /sptlocal/user/kferguson/updated_noise_jackknife_amp_dists_v2/collated_*
```

```python
fname = '/sptlocal/user/kferguson/updated_noise_jackknife_amp_dists_v2/collated_split_best_fit_amps_150GHz_ra0hdec-44.75xra0hdec-52.25.g3.gz'
d = list(core.G3File(fname))
```

```python
print(d[0])
```

```python
plt.hist(d[0]["AmpDiffs"]['0.0155'])
```

## Window functions

```python
from scipy.signal import lombscargle
```

```python
angles = {}
angles_noise = {}
for jfield, field in enumerate(list(obsids.keys())):
    freq = 1. / (60*60*24 * 20)
    noise = np.random.normal(loc=0, scale=0.01*np.pi/180, size=len(obsids[field]))
    angles[field] = 0.05*np.pi/180*np.sin(2*np.pi*obsids[field]*freq) + noise
    angles_noise[field] = noise
    
    data = {'times': {150: obsids[field]},
            'angles': {150: angles[field]},
            'errs': {150: 0.1*np.pi/180*np.ones(len(angles[field]))}}
    model = TimeDomainModel(data)
    result = model.minimize_chi2(freq)
    print(result.x)
    
    plt.figure(jfield+1, figsize=(12,4))
    plt.plot(obsids[field], angles[field], '.-')
    plt.plot()
    plt.title(field)
```

```python
freqs = np.linspace(0.0001,10,1000)
plt.figure(figsize=(10,8))
for jfield, field in enumerate(obsids.keys()):
    periodogram = lombscargle(obsids[field] / (60*60*24),
                              angles[field], freqs)
    plt.plot(freqs / (2*np.pi), np.sqrt(periodogram))
    plt.xlabel('frequency [1/d]')
    plt.ylabel('sqrt(L-S periodogram) [arb.]')
    plt.title('constant DC offset for {}'.format(field))
    plt.tight_layout()
```

```python
freqs = np.linspace(0.0001,10,1000)
plt.figure(figsize=(12,12))
for jfield, field1 in enumerate(obsids.keys()):
    periodogram1 = lombscargle(obsids[field1] / (60*60*24),
                              angles[field1], freqs)
    noise_periodogram1 = lombscargle(obsids[field1] / (60*60*24),
                              angles_noise[field1], freqs)
    for kfield, field2 in enumerate(obsids.keys()):
        if jfield > kfield:
            plt.subplot(4,4,jfield*4 + kfield+1)
            periodogram2 = lombscargle(obsids[field2] / (60*60*24),
                                       angles[field2], freqs)
            noise_periodogram2 = lombscargle(obsids[field2] / (60*60*24),
                                  angles_noise[field2], freqs)
            plt.plot(freqs / (2*np.pi), np.sqrt(periodogram1) - np.sqrt(periodogram2),
                     label='signal injected')
            plt.plot(freqs / (2*np.pi), np.sqrt(noise_periodogram1) - np.sqrt(noise_periodogram2),
                     label='noise-only')
            plt.xlabel('frequency [1/d]')
            plt.ylabel('sqrt(L-S periodogram) [arb.]')
            plt.title('{} - {}'.format(field1, field2))
            plt.legend()
plt.tight_layout()
plt.savefig('lombscargle_difference.png', dpi=200)
```

```python

```
