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

# Flat Sky Simulations of Nonlinearity Bias
During the analysis call of 1 April 2019, there were two main comments on my note simulating detector nonlinearity:

1. Tom mentioned that my simulations convolve projection effects, repixelization effects, and the detector nonlinearity. It's not clear which of these effects contribute most to the distortion that we see in the power spectrum.
1. Christian pointed out that the repixelization that I perform to go from healpix to flatsky is only approximate because the pixel centers in the two projections are not identical. This could induce both small-scale distortion, as well as large-scale distortion because pixel offsets are coherent on larger scales.

Let's try to deal with the first of these comments, but redoing the bias simulations in flatsky. This requies a bit of rejiggering to use some flatsky simulation code that Tom ported from SPTpol / SPT-SZ.

```python
import numpy as np
import matplotlib.pyplot as plt
import camb
from spt3g import core, coordinateutils
from spt3g.simulations.quick_flatsky_routines import cmb_flatsky, make_ellgrid, cl_flatsky
from spt3g.mapspectra.map_analysis import calculateCls
from spt3g import mapspectra
import pickle

%matplotlib inline
```

First, we need to generate some $C_l$ from camb for a plausible cosmology.

```python
pars = camb.CAMBparams()

# This function sets up CosmoMC-like settings,
# with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.87, ombh2=0.022277, omch2=0.11843, mnu=0.06, omk=0, tau=0.0666)
pars.set_for_lmax(10000, lens_potential_accuracy=0)
pars.InitPower.set_params(As=2.141e-9, ns=0.9683, r=0)

#calculate results for these parameters
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=10000)
totCL=powers['total']
```

```python
plt.plot(totCL[:,3])
```

```python
cldict = {'ell': np.arange(totCL.shape[0]),
          'cl': {'TT': totCL[:,0],
                 'EE': totCL[:,1],
                 'BB': totCL[:,2],
                 'TE': totCL[:,3]}}
```

```python
map_res = 2.*core.G3Units.arcmin
map_width = 100.*core.G3Units.deg
map_height = 30.*core.G3Units.deg
map_nwidth = int(map_width / map_res)
map_nheight = int(map_height / map_res)

sim_map = cmb_flatsky(cldict,
                     ngrid = map_nwidth,
                     reso_rad = map_res / core.G3Units.rad,
                     seed=-1,
                     use_seed=False,
                     update_seed=False,
                     rng=-1, use_rng=False,
                     return_rng=False,
                     gradcurl=False,
                     T_only=False)
```

```python
plt.figure(figsize=(12,5))
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map[jmap,:,:])
```

```python
plt.figure(figsize=(12,5))
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map[jmap,:map_nheight,:])
```

```python
sim_map_truncated = sim_map[:,:map_nheight,:]
cl_reco = cl_flatsky(sim_map_truncated,
           reso_arcmin=map_res/core.G3Units.arcmin,
           delta_ell=None,
           ellvec=None,
           apod_mask=None,
           hann=True,
           mask=None,
           gradcurl_in=False,
           trans_func_grid=None,
           tfthresh=0.5,
           return_dl=False,
           verbose=False)
```

```python
for spectrum in cldict['cl'].keys():
    plt.figure()
    plt.plot(cldict['ell'], cldict['cl'][spectrum])
    plt.plot(cl_reco['ell'], cl_reco['cl'][spectrum], '.')
    plt.xlim([0,3000])
    plt.title(spectrum)
```

## Test case, square map

```python
bias_per_deg = 0.02
n_subfields = 4

sim_map = cmb_flatsky(cldict,
                     ngrid = map_nheight,
                     reso_rad = map_res / core.G3Units.rad,
                     seed=-1,
                     use_seed=False,
                     update_seed=False,
                     rng=-1, use_rng=False,
                     return_rng=False,
                     gradcurl=False,
                     T_only=False)

sim_map_biased = sim_map.copy()
for jspectrum in range(3):
    for jfield in range(4):
        bias = np.linspace(1, 1+bias_per_deg*map_nheight / n_subfields * map_res/core.G3Units.deg,
                           map_nheight / n_subfields)
        for jrow in range(int(map_nheight / n_subfields)):
            sim_map_biased[jspectrum, jrow + int(jfield*map_nheight / n_subfields),:] = \
                bias[jrow] * sim_map[jspectrum, jrow + int(jfield*map_nheight / n_subfields),:]
```

```python
plt.figure(figsize=(13,5))
names = ['T', 'Q', 'U']
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map[jmap,:,:])
    plt.title(names[jmap])
plt.savefig('flatsky_sim_30x30d.png', dpi=200)
```

```python
plt.figure(figsize=(14,5))
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map_biased[jmap,:,:] / sim_map[jmap,:,:])
# plt.colorbar()
# plt.tight_layout()
plt.savefig('flatsky_bias_30x30d.png', dpi=200)
```

```python
cl_reco_bias = cl_flatsky(sim_map_biased,
           reso_arcmin=1.0,
           delta_ell=None,
           ellvec=None,
           apod_mask=None,
           hann=False,
           mask=None,
           gradcurl_in=False,
           trans_func_grid=None,
           tfthresh=0.5,
           return_dl=False,
           verbose=False)
```

```python
for spectrum in cldict['cl'].keys():
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.plot(cldict['ell'], cldict['cl'][spectrum])
    plt.plot(cl_reco_bias['ell'], cl_reco_bias['cl'][spectrum], '.')
    plt.xlim([0,3000])
    plt.title(spectrum)
    
    plt.subplot(1,2,2)
    cl_ratio = cl_reco_bias['cl'][spectrum] / cl_reco['cl'][spectrum]
    plt.plot(cl_reco_bias['ell'], cl_ratio, '.')
    plt.xlim([0,10000])
    plt.ylim([0.8, 1.4])
    plt.title(spectrum)
```

## High statistics simulations, square map

```python
def sim_n_cls(cldict, nskies, bias_per_deg = 0.02, n_subfields = 4,
                  map_nside=int(30*60), map_res = 1.*core.G3Units.arcmin):
    cl_sim = {}
    for jsky in range(nskies):
        print('Simulating sky #{}'.format(jsky))
        sim_map = cmb_flatsky(cldict,
                             ngrid = map_nside,
                             reso_rad = map_res / core.G3Units.rad,
                             seed=-1,
                             use_seed=False,
                             update_seed=False,
                             rng=-1, use_rng=False,
                             return_rng=False,
                             gradcurl=False,
                             T_only=False)

        sim_map_biased = sim_map.copy()
        for jspectrum in range(3):
            for jfield in range(4):
                bias = np.linspace(1, 1+bias_per_deg*map_nside / n_subfields * map_res/core.G3Units.deg,
                                   map_nside / n_subfields)
                for jrow in range(int(map_nside / n_subfields)):
                    sim_map_biased[jspectrum, jrow + int(jfield*map_nside / n_subfields),:] = \
                        bias[jrow] * sim_map[jspectrum, jrow + int(jfield*map_nside / n_subfields),:]

        cl_sim[jsky] = cl_flatsky(sim_map_biased,
                   reso_arcmin=map_res / core.G3Units.arcmin,
                   delta_ell=40,
                   ellvec=None,
                   apod_mask=None,
                   hann=False,
                   mask=None,
                   gradcurl_in=False,
                   trans_func_grid=None,
                   tfthresh=0.5,
                   return_dl=False,
                   verbose=False)
        
    return cl_sim
```

```python
map_nside = int(30*60)
map_res = 1.*core.G3Units.arcmin

bias_list = [0, 0.01, 0.02, 0.03]
cls = {}
for bias in bias_list:
    cls[bias] = sim_n_cls(cldict, 500, bias_per_deg = bias, n_subfields = 4,
                          map_nside = map_nheight, map_res = map_res)
    with open('cls_sim_flatsky.pkl', 'wb') as f:
        pickle.dump(cls, f)
```

```python
mean_spectra = {}
for spectrum in ['TT', 'EE', 'TE', 'BB', 'EB', 'TB']:
    mean_spectra[spectrum] = {}
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    for bias in cls:
        mean_spectrum = np.zeros(len(cls[bias][0]['ell']))
        for jsky in cls[bias]:
            mean_spectrum += cls[bias][jsky]['cl'][spectrum]
        mean_spectrum /= len(list(cls[bias].keys()))
        mean_spectra[spectrum][bias] = mean_spectrum
        
        plt.plot(cls[bias][0]['ell'], mean_spectrum,
                 label='{} % bias / deg'.format(bias*100))
    plt.xlim([0, 10000])
    plt.title(spectrum)
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$ [$\mu$K$^2$]')
    plt.legend()
    plt.grid()
        
    plt.subplot(1,2,2)
    for bias in cls:
        ell = cls[bias][0]['ell']
        normalization = np.median(mean_spectra[spectrum][bias][(ell>1000) & (ell<5000)] / \
                                mean_spectra[spectrum][0.0][(ell>1000) & (ell<5000)])
        plt.plot(cls[bias][0]['ell'],
                 mean_spectra[spectrum][bias] / mean_spectra[spectrum][0.0],
                 label='{} % bias / deg'.format(bias*100))
    plt.axis([0, 10000, 0.85, 1.4])
    plt.title(spectrum)
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$(biased) / $D_\ell$(no bias)')
    plt.legend()
    plt.grid()
    plt.savefig('flatsky_spectrum_{}bias.png'.format(spectrum), dpi=120)
```

```python
# load Jason's bandpowers and bandpower uncertainties
with open('3g_sensitivities.pkl', 'rb') as f:
    d_bandpowers = pickle.load(f, encoding='latin1')

# bin the bandpowers and bandpower uncertainties
dell = 40

Dl_binned = dict()
Dl_errors = dict()
Dl_cov = dict()
for spectrum in ['TT', 'EE', 'BB']:
    theory_spectrum = d_bandpowers['theory'][spectrum]
    dcl = theory_spectrum * d_bandpowers['150'][spectrum]
```

```python
ell_norm_range = [1000, 5000]

for spectrum in ['TT', 'EE', 'BB']:
    plt.figure(figsize=(12,6))

    for bias in mean_spectra[spectrum]:
        normalization = np.median(mean_spectra[spectrum][bias][(ell>ell_norm_range[0]) & (ell<ell_norm_range[1])] / \
                                mean_spectra[spectrum][0.0][(ell>ell_norm_range[0]) & (ell<ell_norm_range[1])])
        plt.plot(cls[bias][0]['ell'],
                 mean_spectra[spectrum][bias] / mean_spectra[spectrum][0.0] / normalization,
                 label='{} % bias / deg'.format(bias*100))

    plt.plot(d_bandpowers['theory']['ell'],
             1 + d_bandpowers['150'][spectrum]/2/np.sqrt(dell), 'k--',
             label='bandpower error (150 GHz, $\Delta \ell = {}$)'.format(dell))
    plt.plot(d_bandpowers['theory']['ell'],
             1 - d_bandpowers['150'][spectrum]/2/np.sqrt(dell), 'k--')
    plt.grid()
    plt.title(spectrum) 
    plt.grid()
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$(biased) / $D_\ell$(no bias)' + \
               '\n(normalized to $\ell \in$ ({}, {}))'.format(ell_norm_range[0], ell_norm_range[1]))
    plt.axis([0, 4000, 0.9, 1.1])
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig('flatsky_ratio_normalized_{}_bandpower.png'.format(spectrum), dpi=200)
```

## Test case, real map size

```python
sim_map_wide = cmb_flatsky(cldict,
                     ngrid = map_nwidth,
                     reso_rad = map_res / core.G3Units.rad,
                     seed=-1,
                     use_seed=False,
                     update_seed=False,
                     rng=-1, use_rng=False,
                     return_rng=False,
                     gradcurl=False,
                     T_only=False)

sim_map_wide_biased = sim_map_wide.copy()
for jspectrum in range(3):
    for jfield in range(4):
        bias = np.linspace(1, 1+bias_per_deg*map_nheight / n_subfields * map_res/core.G3Units.deg,
                           map_nheight / n_subfields)
        for jrow in range(int(map_nheight / n_subfields)):
            sim_map_wide_biased[jspectrum, jrow + int(jfield*map_nheight / n_subfields),:] = \
                bias[jrow] * sim_map_wide[jspectrum, jrow + int(jfield*map_nheight / n_subfields),:]
```

```python
plt.figure(figsize=(13,5))
names = ['T', 'Q', 'U']
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map_wide[jmap,:map_nheight,:map_nwidth])
    plt.title(names[jmap])
# plt.savefig('flatsky_sim_30x30d.png', dpi=200)
```

```python
plt.figure(figsize=(13,5))
names = ['T', 'Q', 'U']
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map_wide_biased[jmap,:map_nheight,:map_nwidth] / \
               sim_map_wide[jmap,:map_nheight,:map_nwidth])
    plt.title(names[jmap])
# plt.savefig('flatsky_sim_30x30d.png', dpi=200)
```

```python
sim_map_wide_truncated = sim_map_wide[:,:map_nheight,:]
cl_reco_wide = cl_flatsky(sim_map_wide_truncated,
           reso_arcmin=map_res/core.G3Units.arcmin,
           delta_ell=None,
           ellvec=None,
           apod_mask=None,
           hann=True,
           mask=None,
           gradcurl_in=False,
           trans_func_grid=None,
           tfthresh=0.5,
           return_dl=False,
           verbose=False)
```

```python
for spectrum in cldict['cl'].keys():
    plt.figure()
    plt.plot(cldict['ell'], cldict['cl'][spectrum])
    plt.plot(cl_reco_wide['ell'], cl_reco_wide['cl'][spectrum], '.')
    plt.xlim([0,3000])
    plt.title(spectrum)
```

```python
d_bandpowers['theory']['ell']
```

```python

```
