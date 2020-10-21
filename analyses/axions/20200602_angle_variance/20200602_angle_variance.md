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

# Studies of Pol Angle Variance in Maps
Various aspects of Kyle Ferguson's note [1] confused me, so I decided to do my own quick investigation of the pol angle estimator variance to help generate some ideas for him to look into. For consistency, I will use the tools that he has coded up for map-space calculations (also they are nicely written and clear).

```python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os.path
from glob import glob
from spt3g import core, maps
from spt3g.mapspectra import map_analysis
import pandas as pd
import re
from importlib import reload
from spt3g.std_processing.mapmakers import master_field_coadder
import axion_utils
import pickle
from scipy.stats import norm
from scipy.signal import periodogram

reload(master_field_coadder)
reload(map_analysis)
```

```python
fieldnames = ['ra0hdec-44.75', 'ra0hdec-52.25', 'ra0hdec-59.75', 'ra0hdec-67.25']
year_dir = 'y1_ee_20190811'
maps_dirs = [split_dir \
             for field in fieldnames \
             for split_dir in glob(os.path.join('/spt/user/ddutcher/', field, year_dir, '*'))]

obs_dirs = np.hstack([glob(os.path.join(dirname, '*')) for dirname in maps_dirs])
obsids = [int(os.path.basename(dirname)) for dirname in obs_dirs]
map_fnames = []
for dirname in maps_dirs:
    split_name = dirname.split('/')[6]
    map_fnames.extend(glob(os.path.join(dirname, '*', split_name + '*g3.gz')))
```

```python
map_fnames_dict = {}
for fname in map_fnames:
    result = re.search('maps_(.*?).g3.gz', fname)
    obsid = int(result.group(1))
    
    if obsid not in map_fnames_dict:
        map_fnames_dict[obsid] = {}
    
    result = re.search('_(.*?)GHz_', fname.split('/')[6])
    band = int(result.group(1))
    
    if band not in map_fnames_dict[obsid]:
        map_fnames_dict[obsid][band] = []
        
    map_fnames_dict[obsid][band].append(fname)
```

## Simulations from Kyle

```python
map_sims_dir = '/sptlocal/user/kferguson/mock_observed_cmb_maps_simstubs_from_data/'
map_sims_fnames = glob(os.path.join(map_sims_dir, '*'))

map_sims_fnames_dict = {}
for fname in map_sims_fnames:
    result = re.search('mock_observed_sim_(.*?)_(.*?)GHz', fname)
    obsid = int(result.group(1))
    band = int(result.group(2))
    
    if obsid not in map_sims_fnames_dict:
        map_sims_fnames_dict[obsid] = {}
        
    map_sims_fnames_dict[obsid][band] = fname
```

## Calculated angles
We have the filenames mangled above. Let's coadd Daniel's splits into a single map, inspect, and then use the simstubs from Kyle to compute the polarization angle.

```python
master_field_coadder.run(input_files=map_fnames_dict[54667094][150], log_file='test2.log',
                         flag_bad_maps=False, combine_left_right=True, map_ids=["150GHz"])
```

```python
data_frame = list(core.G3File('coadded_maps.g3'))[0]
```

```python
for fr in core.G3File(map_sims_fnames_dict[54667094][150]):
    if fr.type == core.G3FrameType.Map and fr['Id'] == '150GHz':
        sim_frame = fr
```

```python
rho, rho_err = axion_utils.calculate_rho(data_frame, sim_frame, freq_factor=8.7, return_err=True)
```

```python
rho_plot = np.linspace(-10,10,21) * np.pi / 180
chi2_plot = [axion_utils.calculate_chi2(data_frame, sim_frame, freq_factor=8.7,
                                        rho=rho, use_t=False) \
             for rho in rho_plot]
```

```python
plt.plot(rho_plot*180/np.pi, chi2_plot)
```

```python
data_frame['T'].res / core.G3Units.arcmin
```

```python
map_analysis.calculateNoise(data_frame)
```

```python
map_analysis.calculateNoise(sim_frame)
```

```python
result = map_analysis.calculate_powerspectra(data_frame, delta_l=25, qu_eb='qu')
plt.plot(spectrum.lbins[:-1], np.sqrt(result['UU'].get_cl()) / (core.G3Units.arcmin * core.G3Units.uK))

result = map_analysis.calculate_powerspectra(sim_frame, delta_l=25, qu_eb='qu')
plt.plot(spectrum.lbins[:-1], np.sqrt(result['QQ']) / (core.G3Units.arcmin * core.G3Units.uK))
plt.xlim([6000, 8000])
plt.grid()
```

## Angles from All Observations

```python
with open('angle_fit_data_all.pkl', 'rb') as f:
    angle_data = pickle.load(f)
```

```python
all_angles     = {band: np.array([angle_data['angles_fit'][obsid][band] for obsid in angle_data['angles_fit']])\
                  for band in [90, 150, 220]}
all_angle_errs = {band: np.array([angle_data['angles_err'][obsid][band] for obsid in angle_data['angles_err']])\
                  for band in [90, 150, 220]}
```

```python
plt.figure(1, figsize=(12, 4))
for jband, band in enumerate([90, 150, 220]):
    plt.subplot(1, 3, jband+1)
    plt.hist(all_angles[band]*180/np.pi, bins=np.linspace(-20,20,41))
    plt.title('std = {:.2f}'.format(np.std(all_angles[band]*180/np.pi)))
plt.tight_layout()

plt.figure(2, figsize=(12, 4))
for jband, band in enumerate([90, 150, 220]):
    plt.subplot(1, 3, jband+1)
    plt.hist(all_angle_errs[band]*180/np.pi, bins=np.linspace(0,20,41))
#     plt.title('std = {:.2f}'.format(np.std(all_angles[band]*180/np.pi)))
plt.tight_layout()
```

```python
plt.figure(1, figsize=(12, 4))
for jband, band in enumerate([90, 150, 220]):
    plt.subplot(1, 3, jband+1)
    bins = np.linspace(-2,2,41)
    plt.hist(all_angles[band] / all_angle_errs[band],
             bins=bins,
             normed=True)
    plt.plot(bins, norm.pdf(bins, loc=0, scale=1))
#     plt.title('std = {:.2f}'.format(np.std(all_angles[band]*180/np.pi)))
```

## Pixel variance in a single map

```python
real_data = list(core.G3File('/sptlocal/user/kferguson/full_daniel_maps/ra0hdec-59.75_150GHz_51294774_map.g3.gz'))[0]
```

```python
maps.RemoveWeights(real_data)
```

```python
q_arr = np.array(real_data['Q'])
q_weight = np.array(real_data['Wpol'].QQ)

q_arr_finite = q_arr[np.isfinite(q_arr)]
q_weight_finite = q_weight[np.isfinite(q_arr)]
```

```python
q_normed = q_arr_finite * np.sqrt(q_weight_finite)

_ = plt.hist(q_normed,
             bins=np.linspace(-10, 10, 51),
             density=True)
plt.plot(np.linspace(-10,10),
         norm.pdf(np.linspace(-10,10), loc=0, scale=np.std(q_normed)))
```

```python
np.std(q_normed)
```

### Scaling factor by band and field

```python
coadd_dir = '/sptlocal/user/kferguson/full_daniel_maps/'
coadd_fnames = glob(os.path.join(coadd_dir, '*.g3.gz'))

for fname in coadd_fnames[:5]:
    result = re.match('(.*?)_(.*?)GHz_(.*?)_map.g3.gz', os.path.basename(fname))
    field  = result.group(1)
    band   = int(result.group(2))
    obsid  = int(result.group(3))
    
    real_data = list(core.G3File(fname))[0]
    maps.RemoveWeights(real_data)

    q_arr = np.array(real_data['Q'])
    q_weight = np.array(real_data['Wpol'].QQ)

    q_arr_finite = q_arr[np.isfinite(q_arr)]
    q_weight_finite = q_weight[np.isfinite(q_arr)]

    print('{} {} {}'.format(field, obsid, band))
    print(np.var(q_arr_finite * np.sqrt(q_weight_finite)))
```

```python
with open('weights_var_factors.pkl', 'rb') as f:
    d = pickle.load(f)
```

## Changing Map Settings

```python
map_filenames = ['53556941_2arcmin_default.g3.gz', '53556941_2arcmin_LPF_10000.g3.gz',
                 '53556941_4arcmin_default.g3.gz']

for jfname, fname in enumerate(map_filenames):
    print(fname)
    #d = list(core.G3File('53556941_2arcmin_default.g3.gz'))[5]
    d = list(core.G3File(fname))[5]
    maps.RemoveWeights(d)

#     print(d)

    q_arr = np.array(d['Q'])
    q_weight = np.array(d['Wpol'].QQ)

    q_arr_finite = q_arr[np.isfinite(q_arr)]
    q_weight_finite = q_weight[np.isfinite(q_arr)]

    var = np.var(q_arr_finite * np.sqrt(q_weight_finite))
    print(var)
    
    plt.figure(1)
    plt.hist(q_arr_finite * np.sqrt(q_weight_finite),
             bins=np.linspace(-20, 20, 41), histtype='step', density=True)
    
    plt.figure(10)
    plt.hist(q_arr_finite, bins=np.linspace(-1, 1, 41), histtype='step', density=True)
    
    plt.figure(20)
    plt.hist(q_weight_finite, bins=np.linspace(0, 500, 41), histtype='step', density=True)
    
```

## Checking Gaussianity
We calculated KS test statistic p-values for each of the fields in a separate script `calc_scale_factor.py`. Let's just plot up the results here.

```python
with open('weights_var_factors.pkl', 'rb') as f:
    data_kstest = pickle.load(f)

ks_pvalue = {}
for field in data_kstest['ks_result']:
    ks_pvalue[field] = {90:[], 150:[], 220:[]}
    for obsid in data_kstest['ks_result'][field]:
        for band in ks_pvalue[field]:
            ks_pvalue[field][band].append(data_kstest['ks_result'][field][obsid][band].pvalue)
```

```python
plt.figure(1, figsize=(10,8))

for jfield, field in enumerate(ks_pvalue):
    plt.subplot(2, 2, jfield+1)
    plt.grid()
    for jband, band in enumerate(ks_pvalue[field]):
        plt.hist(ks_pvalue[field][band], bins=np.linspace(0,1,15),
                 histtype='step', label='{} GHz'.format(band))
    plt.title('field: {}'.format(field))
    plt.xlim([0, 1])
    plt.xlabel('KS $p$-value')
    plt.ylabel('observations')
plt.tight_layout()
plt.legend()
plt.savefig('KS_test_pvalues.png', dpi=150)
```

```python
plt.figure(1, figsize=(10,8))

for jfield, field in enumerate(ks_pvalue):
    plt.subplot(2, 2, jfield+1)
    plt.grid()
    for jband, band in enumerate(ks_pvalue[field]):
        plt.hist(ks_pvalue[field][band], bins=np.logspace(-10,0,21),
                 histtype='step', label='{} GHz'.format(band))
    plt.title('field: {}'.format(field))
#     plt.xlim([0, 0.1])
    plt.gca().set_xscale('log')
    plt.xlabel('KS $p$-value')
    plt.ylabel('observations')
plt.tight_layout()
plt.legend()
plt.savefig('KS_test_pvalues_zoom.png', dpi=150)
```

## Checking variance

```python
with open('weights_var_factors_backup.pkl', 'rb') as f:
    data_kstest = pickle.load(f)

std_value = {}
for field in data_kstest['std_factor']:
    std_value[field] = {90:[], 150:[], 220:[]}
    for obsid in data_kstest['std_factor'][field]:
        for band in std_value[field]:
            std_value[field][band].append(data_kstest['std_factor'][field][obsid][band])
```

```python
plt.figure(1, figsize=(10,8))

for jfield, field in enumerate(std_value):
    plt.subplot(2, 2, jfield+1)
    plt.grid()
    for jband, band in enumerate(std_value[field]):
        plt.hist(std_value[field][band], bins=np.linspace(1.75, 3.5 ,41),
                 histtype='step', label='{} GHz'.format(band))
    plt.title('field: {}'.format(field))
    plt.xlim([1.75, 3.5])
    plt.xlabel('std$[Q_i \\times \sqrt{W_{qqi}}]$')
    plt.ylabel('observations')
plt.tight_layout()
plt.legend()
plt.savefig('std_pixel_values.png', dpi=150)
```

## Changing the map settings systematically with Kyle's maps
Based on some arguments above, I convinced myself that the standard deviations would behave more consistently if I used a higher low-pass filter. Kyle has made maps with a variety of mapmaking settings that we can use to test this hypothesis.

```python

```

```python
plt.figure(1, figsize=(10,8))

lpf_list = [3300, 5000, 6600, 8000]
for lpf in lpf_list:
    fname = 'weights_var_factors_2019_4_res_300_hpf_{}_lpf_lrcoadd.pkl'.format(lpf)
    with open(fname, 'rb') as f:
        data_kstest = pickle.load(f)

    std_value = {}
    for field in data_kstest['std_factor']:
        std_value[field] = {90:[], 150:[], 220:[]}
        for obsid in data_kstest['std_factor'][field]:
            for band in std_value[field]:
                std_value[field][band].append(data_kstest['std_factor'][field][obsid][band])

    for jfield, field in enumerate(np.sort(list(std_value.keys()))):
        plt.subplot(2, 2, jfield+1)
        plt.grid()
        std_plot = np.hstack([std_value[field][band] for band in std_value[field]])

        plt.hist(std_plot, bins=np.linspace(2.5, 4.0 ,41),
                 histtype='step', label='LPF = {}'.format(lpf))
        plt.title('field: {}'.format(field))
        plt.xlim([2.5, 4.0])
        plt.xlabel('std$[Q_i \\times \sqrt{W_{qqi}}]$')
        plt.ylabel('observations')
        plt.legend()
plt.tight_layout()
plt.savefig('std_pixel_values_varyLPF.png', dpi=150)
```

## References
[1] - https://pole.uchicago.edu/spt3g/index.php/Estimation_of_per-observation_angle_uncertainties


## Scratch

```python
std_value
```

```python
data_kstest['ks_result'].keys()
```

```python
3.56**2
```

```python

```
