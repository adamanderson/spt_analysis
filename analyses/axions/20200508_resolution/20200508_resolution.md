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

# Effect of Resolution on Pol Angle

```python
from spt3g.simulations import quick_flatsky_routines as qfr
from spt3g.todfilter.dftutils import lowpass_func, highpass_func
import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower

from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
```

```python
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

#calculate results for these parameters
results = camb.get_results(pars)
```

```python
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
cl_dict = {'ell': np.arange(totCL.shape[0])}
cl_dict['cl'] = {'TT': totCL[:,0] / (cl_dict['ell'] * (cl_dict['ell'] + 1)) * 2*np.pi,
                 'EE': totCL[:,1] / (cl_dict['ell'] * (cl_dict['ell'] + 1)) * 2*np.pi,
                 'BB': totCL[:,2] / (cl_dict['ell'] * (cl_dict['ell'] + 1)) * 2*np.pi,
                 'TE': totCL[:,3] / (cl_dict['ell'] * (cl_dict['ell'] + 1)) * 2*np.pi}
for pol in cl_dict['cl']:
    cl_dict['cl'][pol][0] = 0
```

```python
# simulation settings
res_per_pixel = np.pi/180/60
pixels_per_side = 500
```

```python
# experiment settings
spt_beam_fwhm = np.pi/180/60*1.2
spt_beam_sigma = spt_beam_fwhm / np.sqrt(8*np.log(2))
bicep_beam_sigma = np.pi/180 * 0.21
bicep_beam_fwhm = bicep_beam_sigma * np.sqrt(8*np.log(2))
```

## Including a gaussian beam
Let's consider a gaussian beam. The analytic form of this is
\begin{equation}
B_\ell = \exp(-\sigma^2 \ell (\ell+1)),
\end{equation}
where the usual FWHM resolution is related to $\sigma$ by
\begin{equation}
\sigma = \frac{\textrm{FWHM}}{\sqrt{8\log 2}}.
\end{equation}

```python
def B_ell_gaussian(ell, beam_fwhm):
    '''
    Parameters:
    -----------
    ell : array-like
        Multipoles.
    beam_fwhm : float
        Full-width half max of the desired beam, in arcminutes.
    
    Returns:
    --------
    B_ell : array-like
        Spherical harmonic transform of gaussian beam.
    '''
    beam_sigma = np.pi/180/60*beam_fwhm / np.sqrt(8*np.log(2))
    B_ell = np.exp(-1*beam_sigma**2 * ell * (ell+1))
    return B_ell

def cl_beam_corr(beam_fwhm):
    cl_dict_corr = {'ell': cl_dict['ell']}
    cl_dict_corr['cl'] = {pol: cl_dict['cl'][pol] * B_ell_gaussian(cl_dict['ell'], beam_fwhm) \
                          for pol in cl_dict['cl']}
    return cl_dict_corr
```

### Effect of Gaussian beam on $C_\ell$

```python
plt.plot(cl_dict['ell'], B_ell_gaussian(cl_dict['ell'], 1.2), label='1.2 arcmin beam')
plt.plot(cl_dict['ell'], B_ell_gaussian(cl_dict['ell'], 0.21*60*np.sqrt(8*np.log(2))), label='15 arcmin beam')
plt.legend()
plt.xlabel('$\ell$')
plt.ylabel('$B_{\ell}$')
plt.tight_layout()
```

```python
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi),
         label='no beam')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2),
         label='SPT 150 GHz: 1.2 arcmin (FWHM)')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], bicep_beam_fwhm * 180/np.pi * 60),
         label='Keck 150 GHz: 0.21 deg ($\sigma$) (1904.01640)')
plt.xlim([10, 2500])
plt.xlabel('$\ell$')
plt.ylabel('$\mathcal{D}_{\ell}^{EE}$ [$\mu$K$^2$]')

plt.subplot(1, 2, 2)
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'],
         label='no beam')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
                         B_ell_gaussian(cl_dict['ell'], 1.2),
         label='SPT 150 GHz: 1.2 arcmin (FWHM)')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
                         B_ell_gaussian(cl_dict['ell'], 0.21*60*np.sqrt(8*np.log(2))),
         label='Keck 150 GHz: 0.21 deg ($\sigma$) (1904.01640)')
plt.axis([10, 2500, 0, 1e-3])
plt.xlabel('$\ell$')
plt.ylabel('$C_{\ell}^{EE}$ [$\mu$K$^2$]')
plt.legend()

plt.tight_layout()
plt.savefig('emode_spectrum_with_beam.png', dpi=200)
```

```python
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * (2*cl_dict['ell']+1) / (2*np.pi))
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * (2*cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2))
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * (2*cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], bicep_beam_fwhm * 180/np.pi * 60))
plt.xlim([10, 2500])
plt.xlabel('$\ell$')
plt.ylabel('$\mathcal{D}_{\ell}$')
```

```python
np.sum(cl_dict['cl']['EE'] * (2*cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2)) / \
np.sum(cl_dict['cl']['EE'] * (2*cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 30))
```

### Applying a gaussian beam

```python
cl_dict_bicep_beam = cl_beam_corr(bicep_beam_fwhm * 180/np.pi * 60)
fake_sky_bicep_beam = qfr.cmb_flatsky(cl_dict_bicep_beam, ngrid=pixels_per_side,
                                      reso_rad=res_per_pixel, seed=1, use_seed=True)

cl_dict_1p2arcmin_beam = cl_beam_corr(1.2)
fake_sky_1p2arcmin_beam = qfr.cmb_flatsky(cl_dict_1p2arcmin_beam, ngrid=pixels_per_side,
                                          reso_rad=res_per_pixel, seed=1, use_seed=True)
```

```python
pol_titles = ['T', 'Q', 'U']

color_range = [[-350, 350], [-15, 15], [-15, 15]]
plt.figure(1, figsize=(15,4))
for jpol in range(3):
    plt.subplot(1,3,1+jpol)
    plt.imshow(fake_sky_bicep_beam[jpol, :, :],
               vmin=color_range[jpol][0], vmax=color_range[jpol][1])
    plt.colorbar()
    plt.title(pol_titles[jpol])
plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.suptitle('Keck 150 GHz resolution')
plt.savefig('simmap_keck_res.png', dpi=200)

plt.figure(2, figsize=(15,4))
for jpol in range(3):
    plt.subplot(1,3,1+jpol)
    plt.imshow(fake_sky_1p2arcmin_beam[jpol, :, :],
               vmin=color_range[jpol][0], vmax=color_range[jpol][1])
    plt.colorbar()
    plt.title(pol_titles[jpol])
plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.suptitle('SPT 150 GHz resolution')
plt.savefig('simmap_spt3g_res.png', dpi=200)


```

## Low-pass Filter
To investigate the effect of changing the pixelization in the map, let's implement a low-pass filter.

```python
plt.plot(cl_dict['ell'], lowpass_func(cl_dict['ell'], 2700))
```

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi))
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2), 'C1')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2) * \
                         lowpass_func(cl_dict['ell'], 2700), 'C1--')
plt.xlim([10, 2500])
plt.xlabel('$\ell$')
plt.ylabel('$\mathcal{D}_{\ell}$')

plt.subplot(1, 2, 2)
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'])
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
                         B_ell_gaussian(cl_dict['ell'], 1.2), 'C1')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
                         B_ell_gaussian(cl_dict['ell'], 1.2) * \
                         lowpass_func(cl_dict['ell'], 2700), 'C1--')
plt.axis([10, 2500, 0, 1e-3])
plt.xlabel('$\ell$')
plt.ylabel('$C_{\ell}$')

plt.tight_layout()
```

```python
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * (2*cl_dict['ell'] + 1))
# plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
#                          B_ell_gaussian(cl_dict['ell'], 1.2), 'C1')
# plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
#                          B_ell_gaussian(cl_dict['ell'], 1.2) * \
#                          lowpass_func(cl_dict['ell'], 2700), 'C1--')
plt.axis([1500, 3000, 0, 0.7])
plt.xlabel('$\ell$')
plt.ylabel('$(2\ell + 1)C_{\ell}$')

plt.tight_layout()
```

## High-pass Filter
Daniel's maps, which we are currently using, have an $\ell>300$ high-pass filter applied. Let's simulate the effect of this.

```python
plt.plot(cl_dict['ell'], highpass_func(cl_dict['ell'], 300))
```

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi))
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2), 'C1')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * cl_dict['ell'] * (cl_dict['ell']+1) / (2*np.pi) * \
                         B_ell_gaussian(cl_dict['ell'], 1.2) * \
                         highpass_func(cl_dict['ell'], 300), 'C1--')
plt.xlim([10, 2500])
plt.xlabel('$\ell$')
plt.ylabel('$\mathcal{D}_{\ell}$')

plt.subplot(1, 2, 2)
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'])
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
                         B_ell_gaussian(cl_dict['ell'], 1.2), 'C1')
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'] * \
                         B_ell_gaussian(cl_dict['ell'], 1.2) * \
                         highpass_func(cl_dict['ell'], 300), 'C1--')
plt.axis([10, 2500, 0, 1e-3])
plt.xlabel('$\ell$')
plt.ylabel('$C_{\ell}$')

plt.tight_layout()
```

## Including Noise

```python
# map noise level
map_noise = 5 # [uK * arcmin]
pixel_noise = map_noise * np.pi/(180*60) / res_per_pixel # [uK * rad / rad = uK]
```

```python
# generate random map noise, assuming uncorrelated T, Q, U (obviously wrong)
instrument_noise = np.random.normal(0, map_noise, size=(3, pixels_per_side,pixels_per_side))
instrument_noise[1:,:,:] = np.sqrt(2) * instrument_noise[1:,:,:]
```

```python
sim_map_bicep_beam = fake_sky_bicep_beam + instrument_noise
sim_map_1p2arcmin_beam = fake_sky_1p2arcmin_beam + instrument_noise
```

```python
plt.figure(1, figsize=(15,5))
for jpol in range(3):
    plt.subplot(1,3,1+jpol)
    plt.imshow(sim_map_bicep_beam[jpol, :, :])
    plt.colorbar()
    plt.title(pol_titles[jpol])
    
plt.figure(2, figsize=(15,5))
for jpol in range(3):
    plt.subplot(1,3,1+jpol)
    plt.imshow(sim_map_1p2arcmin_beam[jpol, :, :])
    plt.colorbar()
    plt.title(pol_titles[jpol])
```

### Cross-checking the instrumental noise

```python
cl_spectrum = cl_beam_corr(1.2)
for pol in cl_spectrum['cl']:
    cl_spectrum['cl'][pol] = cl_spectrum['cl'][pol] * \
                             lowpass_func(cl_dict['ell'], 2700) * \
                             highpass_func(cl_dict['ell'], 300)
fake_sky_test = qfr.cmb_flatsky(cl_spectrum, ngrid=pixels_per_side, reso_rad=res_per_pixel)
```

```python
plt.figure(1)
plt.imshow(fake_sky_test[0,:,:])
plt.colorbar()
```

## Computing the polarization angle

```python
def chi2_15arcmin_beam(angle):
    return np.sum((fake_sky_15arcmin_beam[1,:,:] - sim_map_15arcmin_beam[1,:,:] + \
                       angle*fake_sky_15arcmin_beam[2,:,:])**2) + \
           np.sum((fake_sky_15arcmin_beam[2,:,:] - sim_map_15arcmin_beam[2,:,:] - \
                       angle*fake_sky_15arcmin_beam[1,:,:])**2)

def chi2_1p2arcmin_beam(angle):
    return np.sum((fake_sky_1p2arcmin_beam[1,:,:] - sim_map_1p2arcmin_beam[1,:,:] + \
                       angle*fake_sky_1p2arcmin_beam[2,:,:])**2) + \
           np.sum((fake_sky_1p2arcmin_beam[2,:,:] - sim_map_1p2arcmin_beam[2,:,:] - \
                       angle*fake_sky_1p2arcmin_beam[1,:,:])**2)
```

```python
result = minimize(chi2_15arcmin_beam, 0.1)
print(result.x[0] * 180/np.pi)

result = minimize(chi2_1p2arcmin_beam, 0.1)
print(result.x[0] * 180/np.pi)
```

## High-Statistics Simulations

```python
def pol_angle_sims(num_sims, res_fwhm_arcmin, map_noise, seed=None, highpass_ell=None, lowpass_ell=None,
                   vary_map=True):
    cl_dict_beam_corr = cl_beam_corr(res_fwhm_arcmin)
    if highpass_ell is not None:
        for pol in cl_spectrum['cl']:
            cl_dict_beam_corr['cl'][pol] = cl_dict_beam_corr['cl'][pol] * \
                                           highpass_func(cl_dict['ell'], highpass_ell)
    if lowpass_ell is not None:
        for pol in cl_spectrum['cl']:
            cl_dict_beam_corr['cl'][pol] = cl_dict_beam_corr['cl'][pol] * \
                                           lowpass_func(cl_dict['ell'], lowpass_ell)
                        
    optional_args = {}
    if seed is not None:
        optional_args['use_seed'] = True
        optional_args['seed'] = seed
        
    if not vary_map:
        fake_sky_beam_corr = qfr.cmb_flatsky(cl_dict_beam_corr, ngrid=pixels_per_side, reso_rad=res_per_pixel,
                                             **optional_args)

    results = []
    for jsim in range(num_sims):
        if vary_map:
            fake_sky_beam_corr = qfr.cmb_flatsky(cl_dict_beam_corr, ngrid=pixels_per_side, reso_rad=res_per_pixel,
                                                 **optional_args)
        
        instrument_noise = np.random.normal(0, map_noise, size=(3, pixels_per_side,pixels_per_side))
        instrument_noise[1:,:,:] = np.sqrt(2) * instrument_noise[1:,:,:]
        sim_map_beam_corr = fake_sky_beam_corr + instrument_noise

        def chi2(angle):
            return np.sum((fake_sky_beam_corr[1,:,:] - sim_map_beam_corr[1,:,:] + \
                               angle*fake_sky_beam_corr[2,:,:])**2) + \
                   np.sum((fake_sky_beam_corr[2,:,:] - sim_map_beam_corr[2,:,:] - \
                               angle*fake_sky_beam_corr[1,:,:])**2)

        result = minimize(chi2, 0.01)
        results.append(result)
    
    fit_angles = np.array([result.x[0] for result in results]) * 180 / np.pi
    
    return fit_angles
```

### Beam-size simulations

```python
res_sim = np.linspace(1, 60, 10)
fit_angles = {}
for res in res_sim:
    print('simulating sigma = {:.1f} arcmin'.format(res))
    fit_angles[res] = {}
    for jsky in range(20):
        fit_angles[res][jsky] = pol_angle_sims(num_sims=20, res_fwhm_arcmin=res, map_noise=5, seed=jsky)
```

```python
# analytic estimate of polarization angle sensitivity from mode-counting
res_plot = np.linspace(1, 60, 100)
sum_of_modes = []
for res in res_plot:
    cl_dict_beam_corr = cl_beam_corr(res)
    sum_of_modes.append(np.sum(cl_dict_beam_corr['cl']['EE'] * (2*cl_dict_beam_corr['ell'] + 1)))
sum_of_modes = np.array(sum_of_modes)
pol_angle_sensitivity = np.sqrt(1./sum_of_modes)
pol_angle_rel_sensitivity = pol_angle_sensitivity / pol_angle_sensitivity[0]
```

```python
fit_angle_means = []
for res in fit_angles:
    fit_angle_stds = [np.std(fit_angles[res][jsky]) for jsky in fit_angles[res]]
    fit_angle_means.append(np.mean(fit_angle_stds))
fit_angle_means = np.array(fit_angle_means)

plt.plot(res_sim, fit_angle_means / fit_angle_means[0], 'o')
plt.plot([1.2, 1.2], [0,10], '--',
         label='SPT 150 GHz: 1.2 arcmin (FWHM)')
plt.plot([0.21*60*np.sqrt(8*np.log(2)), 0.21*60*np.sqrt(8*np.log(2))], [0,10], '--',
         label='Keck 150 GHz: 0.21 deg ($\sigma$) (1904.01640)')
plt.plot(res_plot, pol_angle_rel_sensitivity,
         label='expectation from mode-counting')
plt.ylim([0,10])
plt.legend()
plt.xlabel('beam resolution (FWHM) [arcmin]')
plt.ylabel('sensitivity / area / map depth\nrelative to 1 arcmin FWHM beam')
plt.tight_layout()
plt.grid()
plt.savefig('pol_err_resolution.png', dpi=200)
```

### Filtering simulations

```python
fit_angles_no_filter = pol_angle_sims(num_sims=100, res_fwhm_arcmin=1.2, map_noise=5)
```

```python
fit_angles_with_filter = pol_angle_sims(num_sims=100, res_fwhm_arcmin=1.2, map_noise=5,
                                        highpass_ell=300, lowpass_ell=2700)
```

```python
np.std(fit_angles_no_filter) / np.std(fit_angles_with_filter)
```

```python
rng = np.random.RandomState(12)
```

```python
rng.normal()
```

```python
180*15
```

```python
fit_angles_5uKarcmin = pol_angle_sims(num_sims=100, res_fwhm_arcmin=1.2, map_noise=5)
fit_angles_15uKarcmin = pol_angle_sims(num_sims=100, res_fwhm_arcmin=1.2, map_noise=15)
```

```python
_ = plt.hist(fit_angles_5uKarcmin, bins=np.linspace(-1,1,20),
             histtype='step')
_ = plt.hist(fit_angles_15uKarcmin, bins=np.linspace(-1,1,20),
             histtype='step')
```

```python
cl_dict = {'ell': np.arange(10000),
           'cl': {'TT': 1e-3*np.ones(10000)}}

fake_sky = qfr.cmb_flatsky(cl_dict, ngrid=int(40*60/1),
                           reso_rad=np.pi/180/60 * 1, seed=1, use_seed=True)
result = qfr.cl_flatsky(fake_sky, reso_arcmin=1, ellvec=np.linspace(0,20000,200))
```

```python
result.keys()
```

```python
plt.plot(result['ell'], result['cl']['TT'])
# plt.axis([0, 4000, 1e-1, 1e3])
```

```python
180*60/2
```

```python

```
