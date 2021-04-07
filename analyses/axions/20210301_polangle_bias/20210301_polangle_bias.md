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
from spt3g.simulations.quick_flatsky_routines import *
import camb
from scipy.interpolate import interp1d
```

```python
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(10000, lens_potential_accuracy=0)
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
```

```python
#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
print(totCL.shape)
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ls = np.arange(totCL.shape[0]) + 1
fig, ax = plt.subplots(2,2, figsize = (12,12))
ax[0,0].semilogy(ls,totCL[:,0], color='k')
ax[0,0].semilogy(ls,unlensedCL[:,0], color='r')
ax[0,0].set_title('TT')
ax[0,1].semilogy(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0]);
ax[0,1].set_title(r'$\Delta TT$')
ax[1,0].semilogy(ls,totCL[:,1], color='k')
ax[1,0].semilogy(ls,unlensedCL[:,1], color='r')
ax[1,0].set_title(r'$EE$')
ax[1,1].semilogy(ls,totCL[:,3], color='k')
ax[1,1].semilogy(ls,unlensedCL[:,3], color='r')
ax[1,1].set_title(r'$TE$');
# for ax in ax.reshape(-1): ax.set_xlim([2,2500]);
```

```python
cldict = {'ell': ls,
          'cl': {'TT':totCL[:,0] * 2*np.pi / (ls*(ls+1)),
                 'EE':totCL[:,1] * 2*np.pi / (ls*(ls+1)),
                 'BB':totCL[:,2] * 2*np.pi / (ls*(ls+1)),
                 'TE':totCL[:,3] * 2*np.pi / (ls*(ls+1)),}}
sim_map = cmb_flatsky(cldict, ngrid=1200, reso_rad=np.pi/(180*60))
```

```python
plt.imshow(sim_map[0])
```

```python
sim_cl = cl_flatsky(sim_map, reso_arcmin=1, delta_ell=10)
```

```python
sim_cl['cl'].keys()
plt.plot(sim_cl['ell'], sim_cl['cl']['EE'] * (sim_cl['ell']*(sim_cl['ell']+1)) / (2*np.pi))
plt.xlim([0, 2500])
```

```python
map_noise = 5 # uK-arcmin
res_arcmin = 1.0
pixel_noise = map_noise * np.pi/10800 / (np.pi/10800*res_arcmin)

def generate_noise_map(map_noise, res_armin):
    pixel_noise = map_noise * np.pi/10800 / (np.pi/10800*res_arcmin)
    noise_map = np.random.normal(loc=0, scale=pixel_noise, size=sim_map.shape)
    noise_map[1,:] = np.sqrt(2) * noise_map[1,:]    # polarized noise is sqrt(2) higher
    noise_map[2,:] = np.sqrt(2) * noise_map[2,:]    # polarized noise is sqrt(2) higher
    return noise_map
```

```python
# generate white noise
noise_map = np.random.normal(loc=0, scale=pixel_noise, size=sim_map.shape)
noise_map[1,:] = np.sqrt(2) * noise_map[1,:]    # polarized noise is sqrt(2) higher
noise_map[2,:] = np.sqrt(2) * noise_map[2,:]    # polarized noise is sqrt(2) higher
noise_cl = cl_flatsky(noise_map, reso_arcmin=res_arcmin, delta_ell=10)

sim_cl['cl'].keys()
plt.plot(noise_cl['ell'], np.sqrt(noise_cl['cl']['TT']) / (np.pi/10800))
plt.xlim([0, 2500])
```

```python
noise_cl = cl_flatsky(noise_map, reso_arcmin=res_arcmin, delta_ell=10)

plt.figure(figsize=(12,4))
jpol = 0
for pol in ['TT', 'EE', 'BB']:
    plt.subplot(1,3,jpol+1)
    ell_plot = noise_cl['ell']
    plt.semilogy(ell_plot, np.sqrt(noise_cl['cl'][pol]) / (np.pi/10800))
    plt.title(pol)
    jpol += 1
plt.tight_layout()
```

## Filter setup


In this idealized setting, both signal and noise are isotropic, and there is no transfer function, so our filter comes from pure theory and the assumed noise levels.

```python
def filter_1d(Cls, map_noise):
    ls = np.arange(Cls.shape[0]) + 1
    iso_filter_cl = {'T': Cls[:,0] / (Cls[:,0] + ((map_noise*np.pi/10800)**2)*ls*(ls+1)/(2*np.pi)),
                     'P': Cls[:,1] / (Cls[:,1] + ((map_noise*np.sqrt(2)*np.pi/10800)**2)*ls*(ls+1)/(2*np.pi))}
    iso_filter_cl_interp = {pol: interp1d(ls, iso_filter_cl[pol], fill_value=0, bounds_error=False) \
                            for pol in iso_filter_cl}
    return iso_filter_cl_interp

def filter_map(map_to_filter, filter_func):
    data_ft = np.fft.fft2(map_to_filter)
    ell_grid = make_ellgrid(map_to_filter.shape[1], np.pi/180 * 1.0/60.)
    filter_grid = filter_func(ell_grid)
    filtered_map = np.real(np.fft.ifft2(filter_grid * data_ft))
    return filtered_map

def filter_map_2d(map_to_filter, filter_func):
    fft_of_map = np.fft.fft2(map_to_filter)
    filtered_map = np.real(np.fft.ifft2(fft_of_map * filter_func)) # imaginary part should be rounding error
    return filtered_map

```

```python
iso_filter_cl_interp = filter_1d(totCL, 5.0)
noise_map = generate_noise_map(map_noise=5.0, res_armin=1.0)
sim_plus_noise_map = sim_map + noise_map
```

```python


plt.plot(ls, iso_filter_cl_interp['T'](ls), label='T')
plt.plot(ls, iso_filter_cl_interp['P'](ls), label='E')
plt.legend()
plt.xlabel('$\ell$')
plt.ylabel('filter response')
plt.tight_layout()
plt.savefig('filter_1d.png', dpi=200)
```

```python
data_ft = np.fft.fft2(sim_plus_noise_map)
ell_grid = make_ellgrid(sim_map.shape[1], np.pi/180 * 1.0/60.)
filter_grid = iso_filter_cl_interp['P'](ell_grid)
filtered_map = np.real(np.fft.ifft2(filter_grid * data_ft))
```

```python
plt.figure(1, figsize=(12,8))
labels = ['temperature', 'Stokes Q', 'Stokes U']
for jpol in range(len(sim_plus_noise_map)):
    plt.subplot(2,3,jpol+1)
    plt.imshow(sim_plus_noise_map[jpol,:])
    plt.title(labels[jpol])

for jpol in range(len(sim_plus_noise_map)):
    plt.subplot(2,3,jpol+4)
    plt.imshow(filtered_map[jpol,:])
    plt.title('{} (filtered)'.format(labels[jpol]))
plt.tight_layout()

plt.savefig('filtered_map_example_1d.png', dpi=200)
```

## Fitting polarization angle

```python
def calculate_rho(data_map, template_map):
    q = data_map[1,:,:]
    u = data_map[2,:,:]
    qc = template_map[1,:,:]
    uc = template_map[2,:,:]
    
    num = uc*(qc-q) + qc*(u-uc)
    den = uc**2 + qc**2
    
    rho = np.sum(num) / np.sum(den)
    
    return rho
```

### 1. Bias with no signal injected, perfect template, 100$\mu$K-arcmin noise

```python
nsims = 1000
pol_angles_test1 = np.zeros(nsims)
for jsim in range(nsims):
    noise_map = generate_noise_map(map_noise=100.0, res_armin=1.0)
    sim_plus_noise_map = sim_map + noise_map
    pol_angles_test1[jsim] = calculate_rho(sim_plus_noise_map, sim_map) * 180/np.pi
```

```python
_ = plt.hist(pol_angles_test1, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test1),
                                             np.std(pol_angles_test1) / \
                                                 np.sqrt(len(pol_angles_test1))))
plt.xlabel('pol angle [deg]')
plt.tight_layout()
plt.savefig('test1_polangle.png', dpi=200)
```

### 2. Bias with signal injected, perfect template, 100$\mu$K-arcmin noise

```python
nsims = 1000
rotation_angle = 2.0 * np.pi/180
pol_angles_test2 = np.zeros(nsims)
for jsim in range(nsims):
    noise_map = generate_noise_map(map_noise=100.0, res_armin=1.0)
    
    rotated_map = np.zeros(sim_map.shape)
    rotated_map[1,:,:] = sim_map[1,:,:] - rotation_angle*sim_map[2,:,:]
    rotated_map[2,:,:] = sim_map[2,:,:] + rotation_angle*sim_map[1,:,:]
    
    sim_plus_noise_map = rotated_map + noise_map
    pol_angles_test2[jsim] = calculate_rho(sim_plus_noise_map, sim_map) * 180/np.pi
```

```python
_ = plt.hist(pol_angles_test2, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test2),
                                             np.std(pol_angles_test2) / \
                                                 np.sqrt(len(pol_angles_test2))))
plt.xlabel('pol angle [deg]')
plt.tight_layout()
plt.savefig('test2_polangle.png', dpi=200)
```

### 3. Bias with signal injected, noisy template (5$\mu$K-arcmin), 100$\mu$K-arcmin noise

```python
nsims = 1000
rotation_angle = 2.0 * np.pi/180
pol_angles_test3 = np.zeros(nsims)

template_noise_map = generate_noise_map(map_noise=5.0, res_armin=1.0)
template_map = template_noise_map + sim_map

for jsim in range(nsims):
    noise_map = generate_noise_map(map_noise=100.0, res_armin=1.0)
    
    rotated_map = np.zeros(sim_map.shape)
    rotated_map[1,:,:] = sim_map[1,:,:] - rotation_angle*sim_map[2,:,:]
    rotated_map[2,:,:] = sim_map[2,:,:] + rotation_angle*sim_map[1,:,:]
    
    sim_plus_noise_map = rotated_map + noise_map
    pol_angles_test3[jsim] = calculate_rho(sim_plus_noise_map, template_map) * 180/np.pi
```

```python
_ = plt.hist(pol_angles_test3, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test3),
                                             np.std(pol_angles_test3) / \
                                                 np.sqrt(len(pol_angles_test3))))
plt.xlabel('pol angle [deg]')
plt.tight_layout()
plt.savefig('test3_polangle.png', dpi=200)
```

Below is the same set of simulations as above, but with multiplicative bias correction and overlaid with test 2 (perfect template). Note that we have eliminated the bias, at the expense of variance.

```python
_ = plt.hist(pol_angles_test3 * 2 / np.mean(pol_angles_test3),
             bins=np.linspace(-8,8,41),
             label='noisy template, bias-corrected',
             histtype='step')
_ = plt.hist(pol_angles_test2,
             bins=np.linspace(-8,8,41),
             label='perfect template',
             histtype='step')
plt.legend()
plt.tight_layout()
```

### 4. Signal injected, noisy template (5$\mu$K-arcmin) + azimuthally symmetric filter, 100$\mu$K-arcmin noise

```python
iso_filter_cl_interp = filter_1d(totCL, 5.0)

template_noise_map = generate_noise_map(map_noise=5.0, res_armin=1.0)
template_map = template_noise_map + sim_map

filtered_template_map = filter_map(template_map, iso_filter_cl_interp['P'])
```

```python
plt.figure(figsize=(12,4))
pols = ['T', 'Q', 'U']
for jpol in range(3):
    plt.subplot(1,3,jpol+1)
    plt.imshow(template_map[jpol,:,:])
    plt.title(pols[jpol])
plt.tight_layout()

plt.figure(figsize=(12,4))
pols = ['T', 'Q', 'U']
for jpol in range(3):
    plt.subplot(1,3,jpol+1)
    plt.imshow(filtered_template_map[jpol,:,:])
    plt.title('{} filtered'.format(pols[jpol]))
plt.tight_layout()
```

```python
nsims = 1000
rotation_angle = 2.0 * np.pi/180
pol_angles_test4 = np.zeros(nsims)

for jsim in range(nsims):
    noise_map = generate_noise_map(map_noise=100.0, res_armin=1.0)
    
    rotated_map = np.zeros(sim_map.shape)
    rotated_map[1,:,:] = sim_map[1,:,:] - rotation_angle*sim_map[2,:,:]
    rotated_map[2,:,:] = sim_map[2,:,:] + rotation_angle*sim_map[1,:,:]
    
    sim_plus_noise_map = rotated_map + noise_map
    pol_angles_test4[jsim] = calculate_rho(sim_plus_noise_map, filtered_template_map) * 180/np.pi
```

```python
_ = plt.hist(pol_angles_test4, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test4),
                                             np.std(pol_angles_test4) / \
                                                 np.sqrt(len(pol_angles_test4))))
plt.xlabel('pol angle [deg]')
plt.tight_layout()
plt.savefig('test4_polangle.png', dpi=200)
```

## 2D filter
Let's make a 2D filter. Is the filter constructed out of the power spectrum or the amplitude of the Fourier Transform??

```python
ell_x = np.fft.fftfreq(sim_map.shape[1], d=1/(360*60))
ell_y = np.fft.fftfreq(sim_map.shape[1], d=1/(360*60))
ell_x_2d, ell_y_2d = np.meshgrid(ell_x, ell_y)
phi_ell = np.arctan2(ell_y_2d, ell_x_2d)
ell = np.sqrt(ell_x_2d**2 + ell_y_2d**2)

template_noise_map = generate_noise_map(map_noise=5.0, res_armin=1.0)
ft_template_noise_map = np.fft.fft2(template_noise_map)

cldict = {'ell': ls,
          'cl': {'TT':totCL[:,0] * 2*np.pi / (ls*(ls+1)),
                 'EE':totCL[:,1] * 2*np.pi / (ls*(ls+1)),
                 'BB':totCL[:,2] * 2*np.pi / (ls*(ls+1)),
                 'TE':totCL[:,3] * 2*np.pi / (ls*(ls+1)),}}
# sim_map = cmb_flatsky(cldict, ngrid=1200, reso_rad=np.pi/(180*60))
# psd_sim_map = np.abs(np.fft.fft2(sim_map))**2
tt_interp = interp1d(cldict['ell'], cldict['cl']['TT'], bounds_error=False, fill_value='extrapolate')
ee_interp = interp1d(cldict['ell'], cldict['cl']['EE'], bounds_error=False, fill_value='extrapolate')
t_signal = tt_interp(ell)
q_signal = ee_interp(ell) * np.cos(2*phi_ell)**2
u_signal = ee_interp(ell) * np.sin(2*phi_ell)**2
signal_psd = np.array([t_signal, q_signal, u_signal])

white_noise_level = np.mean(np.abs(ft_template_noise_map)) * (res_arcmin * np.pi/(180*60) / ft_noise_map.shape[1])
psd_noise = np.ones(psd_sim_map.shape) * white_noise_level**2

wiener_filter_2d = signal_psd / (signal_psd + psd_noise)
```

```python
plt.figure(figsize=(15,4))
pols = ['T', 'Q', 'U']
for jpol in np.arange(3):
    plt.subplot(1,3,jpol+1)
    plt.imshow(np.fft.fftshift(np.abs(wiener_filter_2d[jpol,:])), vmin=0, vmax=1,
               extent=(np.min(ell_x), np.max(ell_x),
                       np.min(ell_y), np.max(ell_y)))
    plt.axis([-5000,5000,-5000,5000])
    plt.colorbar()
    plt.xlabel('$\ell_x$')
    plt.ylabel('$\ell_y$')
    plt.title('{} filter'.format(pols[jpol]))
plt.tight_layout()
plt.savefig('filter_2d.png', dpi=200)
```

### 5. Signal injected, noisy template (5$\mu$K-arcmin) + 2D filter, 100$\mu$K-arcmin noise

```python
sim_map = cmb_flatsky(cldict, ngrid=1200, reso_rad=np.pi/(180*60))
template_noise_map = generate_noise_map(map_noise=5.0, res_armin=1.0)
template_map = template_noise_map + sim_map

filtered_template_map = filter_map_2d(template_map, wiener_filter_2d)
```

```python
plt.figure(figsize=(12,8))
pols = ['T', 'Q', 'U']
for jpol in range(3):
    plt.subplot(2,3,jpol+1)
    plt.imshow(template_map[jpol,:,:])
    plt.title('Stokes {}'.format(pols[jpol]))
plt.tight_layout()

pols = ['T', 'Q', 'U']
for jpol in range(3):
    plt.subplot(2,3,jpol+4)
    plt.imshow(filtered_template_map[jpol,:,:])
    plt.title('Stokes {} (filtered)'.format(pols[jpol]))
plt.tight_layout()

plt.savefig('filtered_map_example_2d.png', dpi=200)
```

```python
nsims = 1000
rotation_angle = 2.0 * np.pi/180
pol_angles_test5 = np.zeros(nsims)

for jsim in range(nsims):
    noise_map = generate_noise_map(map_noise=100.0, res_armin=1.0)
    
    rotated_map = np.zeros(sim_map.shape)
    rotated_map[1,:,:] = sim_map[1,:,:] - rotation_angle*sim_map[2,:,:]
    rotated_map[2,:,:] = sim_map[2,:,:] + rotation_angle*sim_map[1,:,:]
    
    sim_plus_noise_map = rotated_map + noise_map
    pol_angles_test5[jsim] = calculate_rho(sim_plus_noise_map, filtered_template_map) * 180/np.pi
```

```python
_ = plt.hist(pol_angles_test5, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test5),
                                             np.std(pol_angles_test5) / \
                                                 np.sqrt(len(pol_angles_test5))))
plt.xlabel('pol angle [deg]')
plt.tight_layout()
plt.savefig('test5_polangle.png', dpi=200)
```

```python
_ = plt.hist(pol_angles_test5, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test5),
                                             np.std(pol_angles_test5) / \
                                                 np.sqrt(len(pol_angles_test5))))
plt.tight_layout()
```

```python
_ = plt.hist(pol_angles_test5, bins=np.linspace(-5,5,41))
plt.title('mean: {:.4f} $\pm$ {:.4f}'.format(np.mean(pol_angles_test5),
                                             np.std(pol_angles_test5) / \
                                                 np.sqrt(len(pol_angles_test5))))
plt.tight_layout()
```

### Comparisons

```python
plt.figure(figsize=(6,5))
corrected_test5 = pol_angles_test5 * 2 / np.mean(pol_angles_test5)
_ = plt.hist(corrected_test5,
             bins=np.linspace(-8,8,41),
             label='noisy filtered (2D) template, bias-corrected (mean = {:.4f} $\pm$ {:.4f})'.format(\
                     np.mean(corrected_test5), np.std(corrected_test5) / np.sqrt(len(corrected_test5))),
             histtype='step')
corrected_test4 = pol_angles_test4 * 2 / np.mean(pol_angles_test4)
_ = plt.hist(corrected_test4,
             bins=np.linspace(-8,8,41),
             label='noisy filtered (1D) template, bias-corrected (mean = {:.4f} $\pm$ {:.4f})'.format(\
                     np.mean(corrected_test4), np.std(corrected_test4) / np.sqrt(len(corrected_test4))),
             histtype='step')
corrected_test3 = pol_angles_test3 * 2 / np.mean(pol_angles_test3)
_ = plt.hist(corrected_test3,
             bins=np.linspace(-8,8,41),
             label='noisy template, bias-corrected (mean = {:.4f} $\pm$ {:.4f})'.format(\
                     np.mean(corrected_test3), np.std(corrected_test3) / np.sqrt(len(corrected_test3))),
             histtype='step')
_ = plt.hist(pol_angles_test2,
             bins=np.linspace(-8,8,41),
             label='perfect template (mean = {:.4f} $\pm$ {:.4f})'.format(\
                     np.mean(pol_angles_test2), np.std(pol_angles_test2) / np.sqrt(len(pol_angles_test2))),
             histtype='step')
# plt.legend()
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0)
plt.xlabel('pol. angle [deg]')
# plt.title('2 deg signal injected')
plt.tight_layout()
plt.savefig('pol_angle_tests.png', dpi=200)
```

```python
corrected_test5 = pol_angles_test5 * 2 / np.mean(pol_angles_test5)
_ = plt.hist(corrected_test5,
             bins=np.linspace(-8,8,41),
             label='noisy filtered (2D) template, bias-corrected (mean = {:.4f} $\pm$ {:.4f})'.format(\
                     np.mean(corrected_test5), np.std(corrected_test5) / np.sqrt(len(corrected_test5))),
             histtype='step')
corrected_test4 = pol_angles_test4 * 2 / np.mean(pol_angles_test4)
_ = plt.hist(corrected_test4,
             bins=np.linspace(-8,8,41),
             label='noisy filtered (1D) template, bias-corrected (mean = {:.4f} $\pm$ {:.4f})'.format(\
                     np.mean(corrected_test4), np.std(corrected_test4) / np.sqrt(len(corrected_test4))),
             histtype='step')
```

```python

```
