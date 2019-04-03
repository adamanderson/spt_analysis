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

# Analyzing Grid Output from Bias Simulations
Let's analyze the output from grid simulations of the detector nonlinearity.

```python
from spt3g import core, mapmaker, calibration
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os.path
import pickle

%matplotlib inline

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
```

## Comparison of biased vs. unbiased spectra
Let's look at the `simbias2` run which compares all maps without cosmology fitting.

```python
dirname = '/spt/user/adama/simbias2/'
filenames = glob(os.path.join(dirname, '*', '*pkl'))
```

```python
data = {}
for fname in filenames:
    nsky = int(os.path.basename(fname).split('_')[1].split('.')[0])
    with open(fname, 'rb') as f:
        data[nsky] = pickle.load(f)
```

```python
for spectrum in ['TT', 'EE', 'TE', 'TB', 'EB', 'BB']:
    plt.figure(figsize=(12,6))
    all_cls_median = {}
    all_cls_up1sigma = {}
    all_cls_down1sigma = {}
    all_cls_ratio_median = {}
    all_cls_ratio_up1sigma = {}
    all_cls_ratio_down1sigma = {}
    for bias_mag in [0.0, 1.0, 2.0, 3.0]:
        ells = data[0][bias_mag]['cls']['ell']

        cls_median = []
        cls_up1sigma = []
        cls_down1sigma = []
        cls_ratio_median = []
        cls_ratio_up1sigma = []
        cls_ratio_down1sigma = []
        for ell in ells:
            cls = [data[nsky][bias_mag]['cls'][spectrum][ells==ell] for nsky in data]
            cls_median.append(np.median(cls))
            cls_up1sigma.append(np.percentile(cls, 84))
            cls_down1sigma.append(np.percentile(cls, 16))
            
            cls_ratio = [data[nsky][bias_mag]['cls'][spectrum][ells==ell] / \
                         data[nsky][0.0]['cls'][spectrum][ells==ell] for nsky in data]
            cls_ratio_median.append(np.median(cls_ratio))
            cls_ratio_up1sigma.append(np.percentile(cls_ratio, 84))
            cls_ratio_down1sigma.append(np.percentile(cls_ratio, 16))
            
        cls_median = np.array(cls_median)
        cls_up1sigma = np.array(cls_up1sigma)
        cls_down1sigma = np.array(cls_down1sigma)
        
        cls_ratio_median = np.array(cls_ratio_median)
        cls_ratio_up1sigma = np.array(cls_ratio_up1sigma)
        cls_ratio_down1sigma = np.array(cls_ratio_down1sigma)
        
        all_cls_ratio_median[bias_mag] = cls_ratio_median
        all_cls_ratio_up1sigma[bias_mag] = cls_ratio_up1sigma
        all_cls_ratio_down1sigma[bias_mag] = cls_ratio_down1sigma
        
        if spectrum in ['TE', 'EB', 'TB']:
            plt.plot(ells, cls_median * ells*(ells+1) / (2.*np.pi),
                     label='{}% bias / deg'.format(bias_mag))
        else:
            plt.semilogy(ells, cls_median * ells*(ells+1) / (2.*np.pi),
                         label='{}% bias / deg'.format(bias_mag))
    plt.title(spectrum) 
    plt.grid()
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$ [$\mu$K$^2$]')
    plt.tight_layout()
    plt.legend()
    plt.savefig('spectrum_{}bias.png'.format(spectrum), dpi=200)

    plt.figure(figsize=(12,6))
    for jmag, bias_mag in enumerate([0.0, 1.0, 2.0, 3.0]):
        for jsky in data:
            plt.plot(ells,
                     data[jsky][bias_mag]['cls'][spectrum] / data[jsky][0.0]['cls'][spectrum],
                     linewidth=0.2, color='C{}'.format(jmag), alpha=0.3)
        plt.fill_between(ells,
                         all_cls_ratio_down1sigma[bias_mag],
                         all_cls_ratio_up1sigma[bias_mag], alpha=0.3,
                         color='C{}'.format(jmag))
        plt.plot(ells, all_cls_ratio_median[bias_mag],
                 color='C{}'.format(jmag),
                     label='{}% bias / deg'.format(bias_mag))
    if spectrum == 'TE':
        plt.ylim([0.9, 1.5])
    if spectrum in ['EB', 'TB']:
        plt.ylim([0, 2])
    plt.title(spectrum) 
    plt.grid()
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$(biased) / $D_\ell$(no bias)')
    plt.tight_layout()
    plt.legend()
    plt.savefig('ratio_{}bias.png'.format(spectrum), dpi=200)
#     plt.xlim([0, 500])
```

## Comparison of biased vs. unbiased parameters

```python
dirname = '/spt/user/adama/simbias5/'
filenames = glob(os.path.join(dirname, '*', '*pkl'))

data_params = {}
for fname in filenames:
    print(fname)
    nsky = int(os.path.basename(fname).split('_')[1].split('.')[0])
    try:
        with open(fname, 'rb') as f:
            data_params[nsky] = pickle.load(f)
    except:
        print('Failed to load file.')
```

```python
len(data_params)
```

```python
data_params[108][1.0]['fit']

xlims = [[60, 70], [2.0e-2, 2.4e-2], [1.1e-1, 1.3e-1]]
param_names = ['$H_0$', '$\Omega_b h^2$', '$\Omega_c h^2$']
for jparam in [0, 1, 2]:
    plt.figure()
    for bias_mag in [0.0, 1.0, 2.0, 3.0]:
        params = [data_params[jsky][bias_mag]['fit'].x[jparam] for jsky in data_params]
        plt.hist(params, bins=np.linspace(xlims[jparam][0], xlims[jparam][1], 21), histtype='step',
                 label='{:.1f}% / deg'.format(bias_mag))
    plt.xlabel(param_names[jparam])
    plt.legend()
```

```python
param_names = ['$H_0$', '$\Omega_b h^2$', '$\Omega_c h^2$']
plt.figure(figsize=(12,4))
for jparam in [0, 1, 2]:
    plt.subplot(1,3,jparam+1)
    for bias_mag in [1.0, 3.0]:
        bias = [(data_params[jsky][bias_mag]['fit'].x[jparam] - \
                   data_params[jsky][0.0]['fit'].x[jparam]) / data_params[jsky][0.0]['fit'].x[jparam] \
                   for jsky in data_params]
        plt.hist(100*np.array(bias), bins=np.linspace(-5, 5, 21), histtype='step',
                 label='{:.1f}% / deg'.format(bias_mag))
    plt.title(param_names[jparam])
    plt.xlabel('% shift vs. no gain variation')
    plt.legend() 
plt.tight_layout()
plt.savefig('bias_param_shift_v0.png', dpi=200)
```

## Comparison of 1 vs. 2 arcmin resolution
Let's check the TT bias with 1 vs. 2 arcmin resolution. This is **not** a resolution effect. Seems to be an effect of the way the fake skies are generated.

```python
dirname_1arcmin = '/spt/user/adama/simbias_1arcmin/'
filenames_1arcmin = glob(os.path.join(dirname_1arcmin, '*', '*pkl'))

dirname_2arcmin = '/spt/user/adama/simbias2/'
filenames_2arcmin = glob(os.path.join(dirname_2arcmin, '*', '*pkl'))
```

```python
data_1arcmin = {}
for fname in filenames_1arcmin:
    nsky = int(os.path.basename(fname).split('_')[2].split('.')[0])
    with open(fname, 'rb') as f:
        data_1arcmin[nsky] = pickle.load(f)
        
data_2arcmin = {}
for fname in filenames_2arcmin:
    nsky = int(os.path.basename(fname).split('_')[1].split('.')[0])
    with open(fname, 'rb') as f:
        data_2arcmin[nsky] = pickle.load(f)
```

```python
all_cls_median_1arcmin = {}
for bias_mag in [0.0, 1.0, 2.0, 3.0]:
    ells = data_1arcmin[0][bias_mag]['cls']['ell']
    
    cls_median = []
    cls_up1sigma = []
    cls_down1sigma = []
    for ell in ells:
        cls = [data_1arcmin[nsky][bias_mag]['cls']['TT'][ells==ell] for nsky in data_1arcmin]
        cls_median.append(np.median(cls))
        cls_up1sigma.append(np.percentile(cls, 84))
        cls_down1sigma.append(np.percentile(cls, 16))
    cls_median = np.array(cls_median)
    cls_up1sigma = np.array(cls_up1sigma)
    cls_down1sigma = np.array(cls_down1sigma)
    all_cls_median_1arcmin[bias_mag] = cls_median
    
all_cls_median_2arcmin = {}
for bias_mag in [0.0, 1.0, 2.0, 3.0]:
    ells = data_2arcmin[0][bias_mag]['cls']['ell']
    
    cls_median = []
    cls_up1sigma = []
    cls_down1sigma = []
    for ell in ells:
        cls = [data_2arcmin[nsky][bias_mag]['cls']['TT'][ells==ell] for nsky in data_2arcmin]
        cls_median.append(np.median(cls))
        cls_up1sigma.append(np.percentile(cls, 84))
        cls_down1sigma.append(np.percentile(cls, 16))
    cls_median = np.array(cls_median)
    cls_up1sigma = np.array(cls_up1sigma)
    cls_down1sigma = np.array(cls_down1sigma)
    all_cls_median_2arcmin[bias_mag] = cls_median
    
plt.figure(figsize=(12,6))
for jmag, bias_mag in enumerate([0.0, 1.0, 2.0, 3.0]):
    plt.plot(ells, all_cls_median_1arcmin[bias_mag] / all_cls_median_1arcmin[0.0],
             color='C{}'.format(jmag), label='{}% bias / deg; 1 arcmin res.'.format(bias_mag))
    plt.plot(ells, all_cls_median_2arcmin[bias_mag] / all_cls_median_2arcmin[0.0],
             '--', color='C{}'.format(jmag),
             label='{}% bias / deg; 2 arcmin res.'.format(bias_mag))
plt.grid()
plt.title('TT') 
plt.grid()
plt.xlabel('$\ell$')
plt.ylabel('$D_\ell$(biased) / $D_\ell$(no bias)')
plt.tight_layout()
plt.legend()
plt.savefig('ratio_{}bias_1vs2arcmin_res.png'.format(spectrum), dpi=200)
```

```python
0.0573/8 *60
```

```python
# load Jason's bandpowers and bandpower uncertainties
with open('3g_sensitivities.pkl', 'rb') as f:
    d_bandpowers = pickle.load(f, encoding='latin1')

# bin the bandpowers and bandpower uncertainties
dell = 50

Dl_binned = dict()
Dl_errors = dict()
Dl_cov = dict()
for spectrum in ['TT', 'EE', 'BB']:
    theory_spectrum = d_bandpowers['theory'][spectrum]
    dcl = theory_spectrum * d_bandpowers['150'][spectrum]
```

```python
for spectrum in ['TT', 'EE', 'BB']:
    plt.figure(figsize=(12,6))
    for jmag, bias_mag in enumerate([0.0, 1.0, 2.0, 3.0]):
        ratio = data[jsky][bias_mag]['cls'][spectrum] / data[jsky][0.0]['cls'][spectrum]
        plt.plot(ells, ratio / np.mean(ratio[(ells>1000) & (ells<2000)]),
                 color='C{}'.format(jmag),
                 label='{}% bias / deg; 2 arcmin res.'.format(bias_mag))
    plt.plot(d_bandpowers['theory']['ell'],
             1 + d_bandpowers['150'][spectrum]/2/np.sqrt(dell), 'k--',
             label='bandpower error (150 GHz, $\Delta \ell = 50$)')
    plt.plot(d_bandpowers['theory']['ell'],
             1 - d_bandpowers['150'][spectrum]/2/np.sqrt(dell), 'k--')
    plt.grid()
    plt.title(spectrum) 
    plt.grid()
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$(biased) / $D_\ell$(no bias)\n(normalized to $\ell \in$ (1000, 2000))')
    plt.axis([0, 4000, 0.9, 1.1])
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig('ratio_normalized_{}_bandpower.png'.format(spectrum), dpi=200)
```

```python
ells
```

## Plots of bias
For presentation purposes, let's make a plot of the bias amplitude as a function of elevation. Let's just lift this from the script used for grid processing.

```python
cal_decs = np.linspace(-71, -41, 4+1)
decs = np.linspace(-71, -41, 1000)
bias_param = np.ones(1000)
for j in range(len(bias_param)):
    bias_param[j] = 1 + 0.02 * (np.min(cal_decs[cal_decs>=decs[j]]) - decs[j])
```

```python
cal_decs
```

```python
plt.plot(decs, bias_param)
plt.xlabel('dec [deg]')
plt.ylabel('relative gain')
plt.title('slice of gain variation in dec (2% / deg)')
plt.savefig('gain_variation_slice.png', dpi=200)
```

```python

```
