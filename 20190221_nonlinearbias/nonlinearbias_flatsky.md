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
from spt3g import core
from spt3g.simulations.quick_flatsky_routines import cmb_flatsky, make_ellgrid, cl_flatsky

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
map_nside = int(30*60)
map_res = 1.*core.G3Units.arcmin

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
```

```python
plt.figure(figsize=(12,5))
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(result[jmap,:,:])
```

```python
cl_reco = cl_flatsky(sim_map,
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
cl_reco
```

```python
for spectrum in cldict['cl'].keys():
    plt.figure()
    plt.plot(cldict['ell'], cldict['cl'][spectrum])
    plt.plot(cl_reco['ell'], cl_reco['cl'][spectrum], '.')
    plt.xlim([0,3000])
    plt.title(spectrum)
```

## Simulating bias

```python
bias_per_deg = 0.02
n_subfields = 4

sim_map_biased = sim_map.copy()
for jspectrum in range(3):
    for jfield in range(4):
        bias = np.linspace(1, 1+bias_per_deg*map_nside / n_subfields * map_res/core.G3Units.deg,
                           map_nside / n_subfields)
        for jrow in range(int(map_nside / n_subfields)):
            sim_map_biased[jspectrum, jrow + int(jfield*map_nside / n_subfields),:] = \
                bias[jrow] * sim_map[jspectrum, jrow + int(jfield*map_nside / n_subfields),:]
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
    plt.imshow(sim_map_biased[jmap,:,:])
```

```python
plt.figure(figsize=(12,5))
for jmap in range(3):
    plt.subplot(1,3,jmap+1)
    plt.imshow(sim_map_biased[jmap,:,:] / sim_map[jmap,:,:])
plt.colorbar()
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

```python

```
