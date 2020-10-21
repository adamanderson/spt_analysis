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

# Wiener Filtering Tests
In a certain sense, our $\chi^2$ estimator is a matched filter for the polarization rotation signal. But as defined in real-space, given computational constraints, it is only optimal (in the sense of being the minimum-variance estimator) if the noise is actually white. Since we know that we have some degree of low-frequency noise, it is natural to wonder whether pre-filtering with a Wiener-type filter may be more optimal. We explore this with some toy flat-sky simulations.

```python
import numpy as np
import matplotlib.pyplot as plt
from spt3g.simulations import quick_flatsky_routines as qfr
import camb
from camb import model, initialpower
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq, ifftshift
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.colors as colors
```

## Generate $C_\ell$ from CAMB
That's all we're doing here in this section.

```python
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(10000, lens_potential_accuracy=0);

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
plt.plot(cl_dict['ell'], cl_dict['cl']['EE'])
```

## Global settings

```python
# simulation settings
res_per_pixel = np.pi/180/60
pixels_per_side = int(10*np.pi/180 / res_per_pixel)
```

```python
# map noise level
map_noise = 10 # [uK * arcmin]
pixel_noise = map_noise * np.pi/(180*60) / res_per_pixel # [uK * rad / rad = uK]
```

## Instrument noise and map checks

```python
# generate random map noise, assuming uncorrelated T, Q, U (obviously wrong)
instrument_noise = np.random.normal(0, map_noise, size=(3, pixels_per_side,pixels_per_side))
instrument_noise[1:,:,:] = np.sqrt(2) * instrument_noise[1:,:,:]
```

```python
fake_sky = qfr.cmb_flatsky(cl_dict, ngrid=pixels_per_side,
                           reso_rad=res_per_pixel, seed=1, use_seed=True)
total_map = fake_sky + instrument_noise
```

```python
pol_titles = ['T', 'Q', 'U']

color_range = [[-350, 350], [-15, 15], [-15, 15]]
plt.figure(1, figsize=(15,4))
for jpol in range(3):
    plt.subplot(1,3,1+jpol)
    plt.imshow(fake_sky[jpol, :, :],
               vmin=color_range[jpol][0], vmax=color_range[jpol][1])
    plt.colorbar()
    plt.title(pol_titles[jpol])
plt.tight_layout()
# plt.subplots_adjust(top=0.86)

plt.figure(2, figsize=(15,4))
for jpol in range(3):
    plt.subplot(1,3,1+jpol)
    plt.imshow(fake_sky[jpol, :, :] + instrument_noise[jpol, :, :],
               vmin=color_range[jpol][0], vmax=color_range[jpol][1])
    plt.colorbar()
    plt.title(pol_titles[jpol])
plt.tight_layout()
```

## Create the Wiener filter

```python
wiener_ell = cl_dict['ell']
Sonly = cl_dict['cl']['EE']
SplusN = cl_dict['cl']['EE'] + (pixel_noise * np.pi / 180 / 60)**2
wiener_ratio = Sonly / SplusN

wiener_ratio[0] = 1
wiener_ratio[-1] = 0
wiener_ell[0] = 0
wiener_ell[-1] = 20000

wiener_filter = interp1d(wiener_ell, wiener_ratio)
```

```python
plt.loglog(SplusN_ell[(SplusN_ell>50) & (SplusN_ell<8000)],
           SplusN[(SplusN_ell>50) & (SplusN_ell<8000)])
plt.loglog(Sonly_ell[(Sonly_ell>50) & (Sonly_ell<8000)],
           Sonly[(Sonly_ell>50) & (Sonly_ell<8000)])
```

```python
# Simple Wiener filter
plt.subplot(2,1,1)
ell_interp = np.linspace(50,15000,1000)
plt.plot(ell_interp, wiener_filter(ell_plot))

plt.subplot(2,1,2)
plt.semilogy(ell_interp, wiener_filter(ell_plot))
```

## Plotting E modes

```python
side_angles = np.linspace(0, pixels_per_side*res_per_pixel, pixels_per_side)
xangle, yangle = np.meshgrid(side_angles, side_angles)
```

```python
ff = fftshift(fftfreq(pixels_per_side, d=res_per_pixel))
ff_x, ff_y = np.meshgrid(ff, ff)

q_ft = fftshift(fft2(fake_sky[1,:,:]))
u_ft = fftshift(fft2(fake_sky[2,:,:]))
emodes_s = q_ft * np.cos(2*np.arctan2(ff_y, ff_x)) + u_ft * np.sin(2*np.arctan2(ff_y, ff_x))

q_ft = fftshift(fft2(total_map[1,:,:]))
u_ft = fftshift(fft2(total_map[2,:,:]))
emodes_sn = q_ft * np.cos(2*np.arctan2(ff_y, ff_x)) + u_ft * np.sin(2*np.arctan2(ff_y, ff_x))
```

```python
plt.figure(1, figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(np.abs(q_ft))
plt.title('Q')
plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(np.abs(u_ft))
plt.title('U')
plt.axis([250, 350, 250, 350])
plt.colorbar()
```

```python
plt.figure(1, figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(np.cos(2*np.arctan2(ff_y, ff_x)))
plt.title('$\cos 2\phi_\ell$')
plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(np.sin(2*np.arctan2(ff_y, ff_x)))
plt.title('$\sin 2\phi_\ell$')
plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(2*np.pi*np.sqrt(ff_x**2 + ff_y**2), vmin=0, vmax=3000)
# plt.title('$\sin 2\phi_\ell$')
plt.axis([250, 350, 250, 350])
plt.colorbar()
```

```python
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.imshow(np.abs(emodes_s))
plt.axis([250, 350, 250, 350])
# plt.axis([150, 450, 150, 450])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(np.abs(emodes_sn))
plt.axis([250, 350, 250, 350])
# plt.axis([150, 450, 150, 450])
plt.colorbar()
```

## Apply Wiener Filter
Let's try actually applying the Wiener filter...

```python
plt.figure(1)
ell_plot = np.linspace(0,5000,1000)
plt.plot(ell_plot, wiener_filter(ell_plot))

plt.figure(2)
plt.imshow(wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2)))
plt.colorbar()
```

```python
q_ft = fftshift(fft2(total_map[1,:,:]))
u_ft = fftshift(fft2(total_map[2,:,:]))

q_ft_filtered_shifted = q_ft * wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2))
u_ft_filtered_shifted = u_ft * wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2))

q_ft_filtered = ifftshift(q_ft_filtered_shifted)
u_ft_filtered = ifftshift(u_ft_filtered_shifted)

q_map_filtered = ifft2(q_ft_filtered)
u_map_filtered = ifft2(u_ft_filtered)
```

```python
# Some Fourier-domain plots
plt.figure(1, figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(np.abs(q_ft))
plt.title('Q')
plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(np.abs(u_ft))
plt.title('U')
plt.axis([250, 350, 250, 350])
plt.colorbar()


plt.figure(2, figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(np.abs(q_ft_filtered_shifted))
plt.title('Q')
plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(np.abs(u_ft_filtered_shifted))
plt.title('U')
plt.axis([250, 350, 250, 350])
plt.colorbar()


plt.figure(3, figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(np.abs(q_ft_filtered))
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(np.abs(u_ft_filtered))
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()
```

```python
# Some real-space plots
plt.figure(1, figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(np.real_if_close(q_map_filtered))
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(np.real_if_close(u_map_filtered))
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(total_map[1,:,:])
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(total_map[2,:,:])
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()
```

## Simulate rotation angle estimator with and without Wiener filter
When we apply the Wiener filter, the map noise is no longer white (flat in frequency), thus our perfect $\chi^2$ ceases to be the optimal maximum-likelihood estimator of the polarization angle.

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


def cl_beam_corr(beam_fwhm, cl_dict):
    cl_dict_corr = {'ell': cl_dict['ell']}
    cl_dict_corr['cl'] = {pol: cl_dict['cl'][pol] * B_ell_gaussian(cl_dict['ell'], beam_fwhm) \
                          for pol in cl_dict['cl']}
    return cl_dict_corr


def create_wiener(cl_dict, pixel_noise):
    wiener_ell = cl_dict['ell']
    Sonly = cl_dict['cl']['EE']
    SplusN = cl_dict['cl']['EE'] + (pixel_noise * np.pi / 180 / 60)**2
    wiener_ratio = Sonly / SplusN

    wiener_ratio[0] = 1
    wiener_ratio[-1] = 0
    wiener_ell[0] = 0
    wiener_ell[-1] = 20000

    wiener_filter = interp1d(wiener_ell, wiener_ratio)
    
    return wiener_filter


def filter_map(maps, filter_func):
    ff = fftshift(fftfreq(pixels_per_side, d=res_per_pixel))
    ff_x, ff_y = np.meshgrid(ff, ff)
    
    q_ft = fftshift(fft2(maps[1,:,:]))
    u_ft = fftshift(fft2(maps[2,:,:]))

    q_ft_filtered_shifted = q_ft * wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2))
    u_ft_filtered_shifted = u_ft * wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2))

    q_ft_filtered = ifftshift(q_ft_filtered_shifted)
    u_ft_filtered = ifftshift(u_ft_filtered_shifted)

    new_maps = np.zeros(maps.shape)
    new_maps[0,:,:] = maps[0,:,:]
    new_maps[1,:,:] = np.real_if_close(ifft2(q_ft_filtered))
    new_maps[2,:,:] = np.real_if_close(ifft2(u_ft_filtered))
    
    return new_maps

    
def pol_angle_sims(num_sims, cl_dict, res_fwhm_arcmin, map_noise, res_per_pixel, pixels_per_side,
                   seed=None, highpass_ell=None, lowpass_ell=None, vary_map=True, do_wiener_filt=False,
                   save_example=False):    
    cl_dict_beam_corr = cl_beam_corr(res_fwhm_arcmin, cl_dict)
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
        
    if do_wiener_filt:
        wiener_filter = create_wiener(cl_dict, pixel_noise = map_noise / res_per_pixel)
        
    if not vary_map:
        fake_sky_beam_corr = qfr.cmb_flatsky(cl_dict_beam_corr, ngrid=pixels_per_side,
                                             reso_rad=res_per_pixel, **optional_args)
        if do_wiener_filt:
            fake_sky_beam_corr_true = filter_map(fake_sky_beam_corr, wiener_filter)
        else:
            fake_sky_beam_corr_true = fake_sky_beam_corr
        

    results = []
    for jsim in range(num_sims):
        if vary_map:
            fake_sky_beam_corr = qfr.cmb_flatsky(cl_dict_beam_corr, ngrid=pixels_per_side,
                                                 reso_rad=res_per_pixel, **optional_args)
            if do_wiener_filt:
                fake_sky_beam_corr_true = filter_map(fake_sky_beam_corr, wiener_filter)
            else:
                fake_sky_beam_corr_true = fake_sky_beam_corr
        
        instrument_noise = np.random.normal(0, map_noise, size=(3, pixels_per_side, pixels_per_side))
        instrument_noise[1:,:,:] = np.sqrt(2) * instrument_noise[1:,:,:]
        sim_map_beam_corr = fake_sky_beam_corr + instrument_noise

        if do_wiener_filt:
            sim_map_beam_corr = filter_map(sim_map_beam_corr, wiener_filter)
            
        
        def chi2(angle):
            return np.sum((sim_map_beam_corr[1,:,:] - fake_sky_beam_corr_true[1,:,:] + \
                               angle*fake_sky_beam_corr_true[2,:,:])**2) + \
                   np.sum((sim_map_beam_corr[2,:,:] - fake_sky_beam_corr_true[2,:,:] - \
                               angle*fake_sky_beam_corr_true[1,:,:])**2)

        result = minimize(chi2, 0.01)
        results.append(result)
    
    fit_angles = np.array([result.x[0] for result in results]) * 180 / np.pi
    
    if save_example:
        return fit_angles, sim_map_beam_corr, fake_sky_beam_corr_true
    else:
        return fit_angles
```

```python
n_sims = 1000
fit_angles        = pol_angle_sims(num_sims=n_sims, cl_dict=cl_dict, res_fwhm_arcmin=1.0,
                                   res_per_pixel=res_per_pixel, pixels_per_side=pixels_per_side,
                                   map_noise=150, seed=1, vary_map=False)
fit_angles_wiener, sample_map, template \
                  = pol_angle_sims(num_sims=n_sims, cl_dict=cl_dict, res_fwhm_arcmin=1.0,
                                   res_per_pixel=res_per_pixel, pixels_per_side=pixels_per_side,
                                   map_noise=150, seed=1, vary_map=False,
                                   do_wiener_filt=True, save_example=True)
```

```python
plt.figure(1)
_ = plt.hist(fit_angles, bins=np.linspace(-15,15,21),
             histtype='step',
             label='no filter ($\sigma = ${:.2f}$^\circ$)'.format(np.std(fit_angles)))
_ = plt.hist(fit_angles_wiener, bins=np.linspace(-15,15,21),
             histtype='step',
             label='with Wiener filter ($\sigma = ${:.2f}$^\circ$)'.format(np.std(fit_angles_wiener)))
plt.legend()
plt.xlabel('estimated polarization angle [deg]')
plt.ylabel('realizations')
plt.tight_layout()
```

```python
# Some real-space plots
plt.figure(1, figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(sample_map[1,:,:])
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(sample_map[2,:,:])
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(template[1,:,:])
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(template[2,:,:])
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()
```

## Some presentation plots
The foregoing sections were a bit tortuous, since I wrote them while learning some details. Let's generate the additional presentation plots that we need below, using the machinery we defined in the preceding section.

```python
plt.figure(1)
ell_plot = np.linspace(0,5000,1000)
plt.plot(ell_plot, wiener_filter(ell_plot))
plt.xlabel('multipole $\ell$')
plt.ylabel('Wiener filter $\mathcal{W}_\ell$')
plt.tight_layout()
plt.savefig('wiener_filter_emodes.png', dpi=200)

plt.figure(2, figsize=(6, 5))
plt.imshow(wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2)),
           extent=[2*np.pi*np.min(ff_x), 2*np.pi*np.max(ff_x),
                   2*np.pi*np.min(ff_y), 2*np.pi*np.max(ff_y)])
plt.xlabel('$\ell_x$')
plt.ylabel('$\ell_y$')
plt.axis([-5000, 5000, -5000, 5000])
plt.colorbar()
plt.title('Wiener filter $\mathcal{W}_\ell$')
plt.tight_layout()
plt.savefig('wiener_filter_emodes_2d.png', dpi=200)
```

```python
ff = fftshift(fftfreq(pixels_per_side, d=res_per_pixel))
ff_x, ff_y = np.meshgrid(ff, ff)

q_ft = fftshift(fft2(total_map[1,:,:]))
u_ft = fftshift(fft2(total_map[2,:,:]))

q_ft_filtered_shifted = q_ft * wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2))
u_ft_filtered_shifted = u_ft * wiener_filter(2*np.pi*np.sqrt(ff_x**2 + ff_y**2))

q_ft_filtered = ifftshift(q_ft_filtered_shifted)
u_ft_filtered = ifftshift(u_ft_filtered_shifted)

q_map_filtered = ifft2(q_ft_filtered)
u_map_filtered = ifft2(u_ft_filtered)

emodes = q_ft * np.cos(2*np.arctan2(ff_y, ff_x)) + u_ft * np.sin(2*np.arctan2(ff_y, ff_x))


plt.figure(1, figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(np.abs(q_ft),
           extent=[2*np.pi*np.min(ff_x), 2*np.pi*np.max(ff_x),
                   2*np.pi*np.min(ff_y), 2*np.pi*np.max(ff_y)])
plt.title('Q')
plt.axis([-3000, 3000, -3000, 3000])
plt.xlabel('$\ell_x$')
plt.ylabel('$\ell_y$')
# plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(np.abs(u_ft),
           extent=[2*np.pi*np.min(ff_x), 2*np.pi*np.max(ff_x),
                   2*np.pi*np.min(ff_y), 2*np.pi*np.max(ff_y)])
plt.title('U')
plt.axis([-3000, 3000, -3000, 3000])
plt.xlabel('$\ell_x$')
plt.ylabel('$\ell_y$')
# plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.abs(emodes),
           extent=[2*np.pi*np.min(ff_x), 2*np.pi*np.max(ff_x),
                   2*np.pi*np.min(ff_y), 2*np.pi*np.max(ff_y)])
plt.title('E modes')
plt.axis([-3000, 3000, -3000, 3000])
plt.xlabel('$\ell_x$')
plt.ylabel('$\ell_y$')
# plt.colorbar()

plt.tight_layout()
plt.savefig('flatsky_fourier2d_withnoise.png', dpi=200)
```

```python
# Some real-space plots
plt.figure(1, figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(sample_map[1,:,:], vmin=-150, vmax=150)
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(sample_map[2,:,:], vmin=-150, vmax=150)
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()
plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.suptitle('Wiener filtered maps ($N_T = 150\mu$K arcmin)')

plt.savefig('flatsky_wienerfiltered_qu.png', dpi=200)
```

```python
# Some real-space plots
plt.figure(1, figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(template[1,:,:], vmin=-17, vmax=17)
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(template[2,:,:], vmin=-17, vmax=17)
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()
plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.suptitle('E-mode maps (noise-free)')

plt.savefig('flatsky_template_qu.png', dpi=200)
```

```python
# Some real-space plots
instrument_noise = np.random.normal(0, 150, size=(3, pixels_per_side, pixels_per_side))
instrument_noise[1:,:,:] = np.sqrt(2) * instrument_noise[1:,:,:]

plt.figure(1, figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(template[1,:,:] + instrument_noise[1,:,:], vmin=-1000, vmax=1000)
plt.title('Q')
# plt.axis([250, 350, 250, 350])
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(template[2,:,:] + instrument_noise[2,:,:], vmin=-1000, vmax=1000)
plt.title('U')
# plt.axis([250, 350, 250, 350])
plt.colorbar()
plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.suptitle('E-mode maps ($N_T = 150\mu$K arcmin)')

plt.savefig('flatsky_template_with_noise_qu.png', dpi=200)
```

```python
plt.figure(1)
_ = plt.hist(fit_angles, bins=np.linspace(-15,15,21),
             histtype='step',
             label='no filter ($\sigma = ${:.2f}$^\circ$)'.format(np.std(fit_angles)))
_ = plt.hist(fit_angles_wiener, bins=np.linspace(-15,15,21),
             histtype='step',
             label='with Wiener filter ($\sigma = ${:.2f}$^\circ$)'.format(np.std(fit_angles_wiener)))
plt.legend()
plt.xlabel('estimated polarization angle [deg]')
plt.ylabel('realizations')
plt.tight_layout()
plt.savefig('fit_angles_with_wiener.png', dpi=200)
```

```python

```
