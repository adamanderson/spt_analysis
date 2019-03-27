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

```python
from spt3g import core, mapmaker, coordinateutils, mapspectra
from spt3g.mapmaker.mapmakerutils import load_spt3g_map
from spt3g.mapspectra.map_analysis import calculateCls
import matplotlib.pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import os.path
from scipy.optimize import minimize
import pickle
import pdb

%matplotlib inline

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
```

Some simulation paths provided by Daniel Dutcher:
* FITS files with lenspix maps:
 * /spt/user/arahlin/lenspix_maps/
* $a_{lm}$ from Kimmy:
 * /spt/user/arahlin/lenspix_alms/
* Script that Sasha used to generate the lenspix maps:
 * spt3g_software/scratch/arahlin/healpix/make_lenspix_maps.py
 
Let's inspect the result of the simulation FITS file after we load it using `load_spt3g_map`.

```python
sim_map_dir = '/spt/user/arahlin/lenspix_maps/'
fname_map = 'lensed_cmb_lmax7000_nside8192_interp0.3_method1_pol_1_sim_65_lensed_map.fits'
map_sim = load_spt3g_map(os.path.join(sim_map_dir, fname_map))
```

Let's take a look at one of Daniel Dutcher's simulation "stub" files. He has generated a bunch of these in directories like `/spt/user/ddutcher/{source}/{mapmaking setting string}/{obsid}/simstub_*.g3`. The stub files are basically the same as the raw scan files, except that the bolometer timestream data has been removed. This means that the size of the stub file is about 15x smaller than the size of the raw data file.

```python
stubfile = '/spt/user/ddutcher/ra0hdec-44.75/sim0_noW201wafCMpoly19mhpf300/' + \
           'sim0_noW201wafCMpoly19mhpf300_51916176.g3'
d_stub = [fr for fr in core.G3File(stubfile)]
```

```python
mapfile = '/spt/user/ddutcher/ra0hdec-44.75/sim0_noW201wafCMpoly19mhpf300/' + \
           'sim0_noW201wafCMpoly19mhpf300_51916176.g3'
d_map = [fr for fr in core.G3File(mapfile)]
```

```python
# make the map frame
Tmap = coordinateutils.FlatSkyMap(4000, 1500, 2.0*core.G3Units.arcmin,
                                  proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                  alpha_center=0*core.G3Units.deg,
                                  delta_center=-60*core.G3Units.deg)
Qmap = coordinateutils.FlatSkyMap(4000, 1500, 2.0*core.G3Units.arcmin,
                                  proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                  alpha_center=0*core.G3Units.deg,
                                  delta_center=-60*core.G3Units.deg)
Umap = coordinateutils.FlatSkyMap(4000, 1500, 2.0*core.G3Units.arcmin,
                                  proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                  alpha_center=0*core.G3Units.deg,
                                  delta_center=-60*core.G3Units.deg)

# inject the curved map values into the flat map
ra = np.zeros((4000, 1500))
dec = np.zeros((4000, 1500))
for xpx in range(4000):
    for ypx in range(1500):
        coords = Tmap.pixel_to_angle(xpx, ypx)
        npx = map_sim['T'].angle_to_pixel(coords[0], coords[1])
        ra[xpx, ypx] = coords[0]
        dec[xpx, ypx] = coords[1]
        if npx < map_sim['T'].shape[0]:
            Tmap[ypx, xpx] = map_sim['T'][npx]
            Qmap[ypx, xpx] = map_sim['Q'][npx]
            Umap[ypx, xpx] = map_sim['U'][npx]
            
# construct the weights
weights = core.G3SkyMapWeights(Tmap, weight_type=core.WeightType.Wpol)
for xpx in range(weights.shape[0]):
    for ypx in range(weights.shape[1]):
        if Tmap[xpx, ypx] != 0:
            weights[xpx, ypx] = np.eye(3)
            
# fill the map frame
map_fr = core.G3Frame(core.G3FrameType.Map)
map_fr['T'] = Tmap
map_fr['Q'] = Qmap
map_fr['U'] = Umap
map_fr['Wpol'] = weights
```

```python
# construct the apodization mask
apod = mapspectra.apodmask.makeBorderApodization(
           map_fr['Wpol'], apod_type='cos',
           radius_arcmin=15.,zero_border_arcmin=10,
           smooth_weights_arcmin=5)
```

```python
# calculate power spectra
cls_data = calculateCls(map_fr, apod_mask=apod, t_only=False, delta_ell=40)
```

```python
# plot the T, Q, U maps
plt.figure(figsize=(12,4))
plt.imshow(np.array(Tmap))
plt.colorbar()

plt.figure(figsize=(12,4))
plt.imshow(np.array(Qmap))
plt.colorbar()

plt.figure(figsize=(12,4))
plt.imshow(np.array(Umap))
plt.colorbar()
```

```python
# plot the apodization mask
plt.figure(figsize=(12,4))
plt.imshow(apod)
plt.colorbar()
```

```python
# plot a zoom in of the T map just make sure it looks reasonable
plt.figure(figsize=(8,8))
plt.imshow(np.array(Tmap))
plt.axis([2000, 2100, 800, 900])
```

```python
# and finally plot the power spectra
plt.figure(figsize=(10,10))
for jspectrum, spectrum in enumerate(['TT', 'EE', 'BB', 'TE']):
    plt.subplot(2,2,jspectrum+1)
    plt.plot(cls_data['ell'],
             cls_data[spectrum]*cls_data['ell']*(cls_data['ell']+1) / (2*np.pi), '.-')
    plt.xlim([50,2500])
    plt.title(spectrum)
```

## Bandpower uncertainties from Jason's forecasts

```python
# load Jason's bandpowers and bandpower uncertainties
with open('3g_sensitivities.pkl', 'rb') as f:
    d_bandpowers = pickle.load(f, encoding='latin1')
```

```python
# bin the bandpowers and bandpower uncertainties
dell = cls_data['ell'][1] - cls_data['ell'][0]

Dl_binned = dict()
Dl_errors = dict()
Dl_cov = dict()
for spectrum in ['TT', 'EE', 'BB']:
    theory_spectrum = d_bandpowers['theory'][spectrum]
    dcl = theory_spectrum * d_bandpowers['150'][spectrum]

    Dl_binned[spectrum] = np.array([np.sum(theory_spectrum[(ells >= ell_center - (dell/2.)) & \
                                                           (ells < ell_center + (dell/2.))]) / dell \
                                    for ell_center in cls_data['ell']])
    Dl_errors[spectrum] = np.array([np.sqrt(np.sum(dcl[(ells >= ell_center - (dell/2.)) & \
                                                       (ells < ell_center + (dell/2.))]**2.0)) / dell \
                                    for ell_center in cls_data['ell']])
    Dl_cov[spectrum] = Dl_errors[spectrum]**2.
```

```python
plt.figure(figsize=(12,6))
for jspectrum, spectrum in enumerate(['TT', 'EE', 'BB']):
#     plt.subplot(3,1,jspectrum+1)
    plt.errorbar(cls_data['ell'][cls_data['ell']<2000],
                 Dl_binned[spectrum][cls_data['ell']<2000],
                 Dl_errors[spectrum][cls_data['ell']<2000], linestyle='None')
#     plt.title(spectrum)
plt.gca().set_yscale('log')
```

```python
plt.figure(figsize=(12,6))
for jspectrum, spectrum in enumerate(['TT', 'EE', 'BB']):
    theory_spectrum = d_bandpowers['theory'][spectrum]
    dcl = theory_spectrum * d_bandpowers['150'][spectrum]
    plt.errorbar(d_bandpowers['theory']['ell'],
                 theory_spectrum,
                 dcl / np.sqrt(40), linestyle='None')
plt.gca().set_yscale('log')
plt.grid()
plt.axis([50, 2000, 1e-4, 1e4])
```

```python
plt.figure(figsize=(12,6))
for jspectrum, spectrum in enumerate(['TT', 'EE', 'BB']):
    plt.plot(cls_data['ell'][cls_data['ell']<2000],
             Dl_errors[spectrum][cls_data['ell']<2000] / \
             Dl_binned[spectrum][cls_data['ell']<2000], 'o-',
             label='{}'.format(spectrum))
plt.grid()
plt.axis([50, 2000, 0, 0.25])
plt.legend()
plt.xlabel('$\ell$')
plt.ylabel('bandpower error / bandpower')
plt.tight_layout()
plt.savefig('jason_forecast_sigmaDloverDl.png', dpi=200)
```

## Fitting cosmology
Let's now fit the noise-free simulations to the biased simulations.

```python
def knox_errors(ells, cls, fsky, noise):
    sigma_beam = 1.2 / np.sqrt(8*np.log(2)) * np.pi / 10800.
    dcls = np.sqrt(2. / ((2.*ells + 1) * fsky)) * \
                (cls + (noise * np.pi / 10800)**2. * np.exp(ells**2 * sigma_beam**2))
    return dcls
```

```python
def neg2LogL(x, cls_data):
    camb_index = {'TT':0, 'EE':1, 'BB':3, 'TE': 2}
    fsky = 1500 / (4*np.pi / ((2*np.pi / 360)**2))
    Tnoise = {150: 2.2} # uK arcmin
    
    print(x)
    H0 = x[0]
    ombh2 = x[1]
    omch2 = x[2]
    
    if H0 > 80 or H0 < 50 or ombh2 > 0.03 or ombh2 < 0.012 or omch2 > 0.15 or omch2 < 0.07:
        return 1e9

    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings,
    # with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=0.0666)
    camb.set_params(lmax=5000)
    pars.InitPower.set_params(As=2.141e-9, ns=0.9683, r=0)

    #calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=powers['total']

    ells_theory = np.arange(totCL.shape[0])
    ells_theory_binned = np.intersect1d(ells_theory, cls_data['ell'])
    ells_data = cls_data['ell'][np.isin(cls_data['ell'], ells_theory_binned)]
    chi2 = 0
    for spectrum in ['TT', 'EE']:
        dls_theory_binned = totCL[:,camb_index[spectrum]] \
                                 [np.isin(ells_theory, ells_theory_binned)]
        cls_theory_binned = dls_theory_binned / (ells_theory_binned*(ells_theory_binned+1) / (2*np.pi))
        cls_data_binned = cls_data[spectrum][np.isin(cls_data['ell'], ells_theory_binned)]
        residual = cls_theory_binned - cls_data_binned 
        
        if spectrum == 'TT':
            noise = Tnoise[150]
        else:
            noise = Tnoise[150] * np.sqrt(2.)
        cl_cov = knox_errors(ells_theory_binned, cls_theory_binned, fsky, noise)**2
        
        chi2 += np.sum(residual**2. / cl_cov)
    print(chi2)
    
    return chi2
```

```python
res = minimize(neg2LogL, [67.87, 0.022277, 0.11843], args=(cls_data), method='powell',
               options={'xtol': 1e-6, 'disp': True})
```

```python
res
```

```python
res = minimize(neg2LogL, [65.87, 0.023277, 0.11843], args=(cls_data), method='powell',
               options={'xtol': 1e-6, 'disp': True})
```

```python
res
```

## Checking the Knox formula

```python
fsky = 1500 / (4*np.pi / ((2*np.pi / 360)**2))
Tnoise = {150: 15} # uK arcmin

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.87, ombh2=0.022277, omch2=0.11843, mnu=0.06, omk=0, tau=0.0666)
camb.set_params(lmax=5000)
pars.InitPower.set_params(As=2.141e-9, ns=0.9683, r=0)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL=powers['total']

ells_knox = np.arange(totCL.shape[0])
dls_knox = totCL[:,1]
cls_knox = dls_knox / ((ells_knox*(ells_knox+1)) / (2.*np.pi))

cl_errors_knox = knox_errors(ells_knox, cls_knox, fsky, Tnoise[150])
```

```python
# plot results; in reasonable agreement with Daniel's 150 GHz figure on the wiki here:
# https://pole.uchicago.edu/spt3g/index.php/File:Y1_EE_sensitivity.png
plt.figure()
plt.semilogy(ells_knox, cl_errors_knox / cls_knox)
plt.ylim([0.1, 10])
plt.grid()
```

```python

```
