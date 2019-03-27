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

# Noise Plots for Tijmen for S4 Meeting
This is a super-quick note to generate a few plots for Tijmen's fMUX overview at the CMB-S4 collaboration meeting. He just wanted the comparison of on-sky noise and readout noise from our noise measurements at the horizon.

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
```

```python
fname_noise_intrans     = '/big_scratch/pydfmux_output/20181224/' \
                          '20181224_015850_measure_noise/data/8592_BOLOS_INFO_AND_MAPPING.pkl'
fname_noise_normalVbias = '/big_scratch/pydfmux_output/20181225/' \
                          '20181225_073517_measure_noise/data/9999_BOLOS_INFO_AND_MAPPING.pkl'
fname_noise_smallVbias  = '/big_scratch/pydfmux_output/20181225/' \
                          '20181225_090623_measure_noise/data/11200_BOLOS_INFO_AND_MAPPING.pkl'
```

```python
with open(fname_noise_intrans, 'rb') as f:
    noise_intrans = pickle.load(f, encoding='latin1')
```

```python
with open(fname_noise_normalVbias, 'rb') as f:
    noise_normalVbias = pickle.load(f, encoding='latin1')
```

```python
with open(fname_noise_smallVbias, 'rb') as f:
    noise_smallVbias = pickle.load(f, encoding='latin1')
```

```python
nei_intrans        = np.array([noise_intrans[bolo]['noise']['i_phase']['median_noise']
                               for bolo in noise_intrans])
wafers_intrans     = np.array([noise_intrans[bolo]['wafer_name']
                               for bolo in noise_intrans])
bands_intrans      = np.array([float(noise_intrans[bolo]['physical_name'].split('.')[1])
                               for bolo in noise_intrans])
nei_normalVbias    = np.array([noise_normalVbias[bolo]['noise']['i_phase']['median_noise']
                               for bolo in noise_normalVbias])
wafers_normalVbias = np.array([noise_normalVbias[bolo]['wafer_name']
                               for bolo in noise_normalVbias])
bands_normalVbias  = np.array([float(noise_normalVbias[bolo]['physical_name'].split('.')[1])
                               for bolo in noise_normalVbias])
nei_smallVbias     = np.array([noise_smallVbias[bolo]['noise']['i_phase']['median_noise']
                               for bolo in noise_smallVbias])
wafers_smallVbias  = np.array([noise_smallVbias[bolo]['wafer_name']
                               for bolo in noise_smallVbias])
bands_smallVbias   = np.array([float(noise_smallVbias[bolo]['physical_name'].split('.')[1])
                               for bolo in noise_smallVbias])
wafer_list  = np.unique(wafers)
```

```python
plt.rc('axes', linewidth=1.5)
for jwafer, wafer in enumerate(wafer_list):
    plt.figure(jwafer, figsize=(12,3.5))
    for jband, band in enumerate([90, 150, 220]):
        plt.subplot(1,3,jband+1)
        plt.title('{} GHz'.format(band))
        plt.hist(nei_intrans[(wafers_intrans==wafer) & (bands_intrans==band)],
                 bins=np.linspace(0,50,31), linewidth=1.5,
                 histtype='step', label='in-transition')
        plt.hist(nei_normalVbias[(wafers_normalVbias==wafer) & (bands_normalVbias==band)],
                 bins=np.linspace(0,50,31), linewidth=1.5,
                 histtype='step', label='readout only')
        plt.xlabel('current noise [pA/rtHz]')
        plt.ylabel('bolometers')
        if band == 220:
            plt.legend(frameon=False)
        plt.suptitle(wafer)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('nei_{}.png'.format(wafer), dpi=200)
```

```python
plt.rc('axes', linewidth=1.5)
plt.figure(jwafer, figsize=(12,3.5))
for jband, band in enumerate([90, 150, 220]):
    plt.subplot(1,3,jband+1)
    plt.title('{} GHz'.format(band))
    plt.hist(nei_intrans[(bands_intrans==band)],
             bins=np.linspace(0,50,31), linewidth=1.5,
             histtype='step', label='in-transition (el~45d)')
    plt.hist(nei_normalVbias[(bands_normalVbias==band)],
             bins=np.linspace(0,50,31), linewidth=1.5,
             histtype='step', label='normal Vbias (horizon)')
    plt.hist(nei_smallVbias[(bands_smallVbias==band)],
             bins=np.linspace(0,50,31), linewidth=1.5,
             histtype='step', label='small Vbias (horizon)', linestyle='--')
    plt.xlabel('current noise [pA/rtHz]')
    plt.ylabel('bolometers')
    if band == 220:
        plt.legend(frameon=False)
    plt.suptitle('all wafers')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('nei_all.png', dpi=200)
```

```python
plt.rc
```

```python

```
