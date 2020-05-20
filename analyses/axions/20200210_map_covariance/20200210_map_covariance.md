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

# Map-Space Covariance
Let's estimate the map-space covariance from a single map, assuming that the spatial correlations in the map are isotropic. We use Daniel's 2018 E mode maps.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from spt3g import core, calibration, mapmaker
from spt3g.mapmaker.mapmakerutils import remove_weight
from spt3g.mapspectra.map_analysis import apply_weight
from glob import glob
import pickle
import os.path
```

```python
fname_maps = glob('/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/' + \
                  'high_150GHz_left_maps/*/high_150GHz_left_maps_*.g3.gz')
```

```python
print(fname_maps[0])
mapdata = list(core.G3File(fname_maps[0]))[-1]
coadddata = list(core.G3File('/spt/user/ddutcher/coadds/20190917_full_150GHz.g3.gz'))[0]
```

```python
map_noweight = core.G3Frame(core.G3FrameType.Map)
coadd_noweight = core.G3Frame(core.G3FrameType.Map)

map_noweight['T'], map_noweight['Q'], map_noweight['U']       = remove_weight(mapdata['T'], mapdata['Q'],
                                                                              mapdata['U'], mapdata['Wpol'])
map_noweight['Wpol'] = mapdata['Wpol']

coadd_noweight['T'], coadd_noweight['Q'], coadd_noweight['U'] = remove_weight(coadddata['T'], coadddata['Q'],
                                                                              coadddata['U'], coadddata['Wpol'])
coadd_noweight['Wpol'] = coadddata['Wpol']
```

In the foregoing analysis, we scan over pixels in the map and construct a cutout centered on each one. If the cutout does not have any nan pixels, then we use it to estimate one sample of the pixel-pixel covariance. We then average all the samples together to form the estimate of a local block in the pixel-pixel covariance.

```python
cutout_dim = 1*core.G3Units.deg
cutout_size = int(cutout_dim / map_noweight['Q'].res)
if cutout_size % 2:
    cutout_size += 1

# unweighted coadd from unweighted observation, then add weights back in
map_subtracted = core.G3Frame(core.G3FrameType.Map)
for stokes in ['T', 'Q', 'U']:
    map_subtracted[stokes] = map_noweight[stokes] - coadd_noweight[stokes]
map_subtracted['Wpol'] = map_noweight['Wpol']
# map_subtracted_weight = apply_weight(map_subtracted)
#Q_arr = np.array(map_subtracted_weight['Q'])
Q_arr = np.array(map_subtracted['Q'])

cutout_running = np.zeros((cutout_size, cutout_size))

for i in range(Q_arr.shape[0] - cutout_size):
    for j in range(Q_arr.shape[1] - cutout_size):
        cutout = Q_arr[i:(i+cutout_size), j:(j+cutout_size)]
        cutout_center = cutout[int((cutout_size-1) / 2),
                               int((cutout_size-1) / 2)]
        if np.all(np.isfinite(cutout)):
            cutout_running += cutout * cutout_center
```

```python
plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.imshow(cutout_running / cutout_running.max(),
           norm=colors.SymLogNorm(linthresh=0.001, linscale=1, vmin=-1, vmax=1))
plt.colorbar()
plt.xlabel('x pixel [2 arcmin/pixel]')
plt.ylabel('y pixel [2 arcmin/pixel]')
plt.title('log scale')

plt.subplot(1,2,1)
plt.imshow(cutout_running / cutout_running.max(), vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel('x pixel [2 arcmin/pixel]')
plt.ylabel('y pixel [2 arcmin/pixel]')
plt.title('linear scale')

plt.tight_layout()

plt.savefig('figures/pixelpixel_cov_q_2arcmin.png', dpi=200)
```

```python
_ = plt.hist(np.hstack(cutout_running / cutout_running.max()),
             bins=np.linspace(-1,1,101))
plt.gca().set_yscale('log')
plt.ylabel('pixels')
plt.xlabel('covariance')
plt.tight_layout()

plt.savefig('figures/pixelpixel_cov_q_hist_2arcmin.png', dpi=200)
```

There is clearly significant off-diagonal content in the pixel-pixel covariance matrix, although only in the nearest-neighbor pixels. There are two potential ways of dealing with this issue:
1. **Decrease resolution -** Decreasing resolution effectively throws away power at high multipoles by averaging over fluctuations. This is bad, but the vast majority of the power in the E-mode power spectrum is at $\ell > 3000$, which corresponds roughly to 4 arcmin pixels. We could therefore degrade our resolution by a factor of 2x without appreciable loss of signal.
2. **Invert the covariance -** This is annoying because the covariance matrix does not have a nice simple form due to the remapping of a 2D map into a 1D array. Nevertheless, the matrix is extremely sparse, so there may be some hope here... Not really. Each map has order $10^6$ pixels, so the covariance matrix has order $10^{12}$ entries. It may be possible to evaluate the inverse, but certainly not for every observation.
3. **Divide map -** All regions of the map are, in principle, equivalent, so we could split the map into sections small enough to have an invertible covariance. This would be equivalent to high-pass filtering the data, which also discards modes.



## Decrease Resolution
Let's proceed with the first (and by far the simplest) of these options. We rebin the map to have 4 arcmin pixels, and then recompute the map-space covariance, hoping that it is diagonal. Note that the approach of [1605.08633](https://arxiv.org/pdf/1605.08633.pdf) is to do exactly this. Their map-space analysis bins coarsely and then assumes that the pixel-pixel covariance is diagonal.

```python
plt.figure(figsize=(15,10))
plt.imshow(mapdata['Q'].rebin(2))
```

Rebinning works, so now let's recompute the pixel-pixel covariance for this observation using maps rebinned to 4 arcmin.

```python
mapdata = list(core.G3File(fname_maps[0]))[-1]
coadddata = list(core.G3File('/spt/user/ddutcher/coadds/20190917_full_150GHz.g3.gz'))[0]

mapdata_rebin = core.G3Frame(core.G3FrameType.Map)
coadddata_rebin = core.G3Frame(core.G3FrameType.Map)

for stokes in ['T', 'Q', 'U', 'Wpol']:
    mapdata_rebin[stokes] = mapdata[stokes].rebin(2)
    coadddata_rebin[stokes] = coadddata[stokes].rebin(2)
```

```python
map_rebin_noweight = core.G3Frame(core.G3FrameType.Map)
coadd_rebin_noweight = core.G3Frame(core.G3FrameType.Map)

map_rebin_noweight['T'], map_rebin_noweight['Q'], map_rebin_noweight['U'] = \
    remove_weight(mapdata_rebin['T'], mapdata_rebin['Q'],
                  mapdata_rebin['U'], mapdata_rebin['Wpol'])
map_rebin_noweight['Wpol'] = mapdata_rebin['Wpol']

coadd_rebin_noweight['T'], coadd_rebin_noweight['Q'], coadd_rebin_noweight['U'] = \
    remove_weight(coadddata_rebin['T'], coadddata_rebin['Q'],
                  coadddata_rebin['U'], coadddata_rebin['Wpol'])
coadd_rebin_noweight['Wpol'] = coadddata_rebin['Wpol']
```

```python
map_noweight_rebin = core.G3Frame(core.G3FrameType.Map)


cutout_dim = 1*core.G3Units.deg
cutout_size = int(cutout_dim / map_rebin_noweight['Q'].res)
if cutout_size % 2:
    cutout_size += 1

# unweighted coadd from unweighted observation, then add weights back in
map_subtracted = core.G3Frame(core.G3FrameType.Map)
for stokes in ['T', 'Q', 'U']:
    map_subtracted[stokes] = map_rebin_noweight[stokes] - coadd_rebin_noweight[stokes]
# map_subtracted['Wpol'] = map_rebin_noweight['Wpol']
# map_subtracted_weight = apply_weight(map_subtracted)
Q_arr = np.array(map_subtracted['Q'])

cutout_running = np.zeros((cutout_size, cutout_size))

for i in range(Q_arr.shape[0] - cutout_size):
    for j in range(Q_arr.shape[1] - cutout_size):
        cutout = Q_arr[i:(i+cutout_size), j:(j+cutout_size)]
        cutout_center = cutout[int((cutout_size-1) / 2),
                               int((cutout_size-1) / 2)]
        if np.all(np.isfinite(cutout)):
            cutout_running += cutout * cutout_center
```

```python
plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.imshow(cutout_running / cutout_running.max(),
           norm=colors.SymLogNorm(linthresh=0.001, linscale=1, vmin=-1, vmax=1))
plt.colorbar()
plt.xlabel('x pixel [2 arcmin/pixel]')
plt.ylabel('y pixel [2 arcmin/pixel]')
plt.title('log scale')

plt.subplot(1,2,1)
plt.imshow(cutout_running / cutout_running.max(), vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel('x pixel [2 arcmin/pixel]')
plt.ylabel('y pixel [2 arcmin/pixel]')
plt.title('linear scale')

plt.tight_layout()

plt.savefig('figures/pixelpixel_cov_q_4arcmin.png', dpi=200)
```

```python
_ = plt.hist(np.hstack(cutout_running / cutout_running.max()),
             bins=np.linspace(-0.1,0.1, 51))
plt.gca().set_yscale('log')
plt.ylabel('pixels')
plt.xlabel('covariance')
plt.tight_layout()

plt.savefig('figures/pixelpixel_cov_q_hist_4arcmin.png', dpi=200)
```

Based on the above, we can conclude that off-diagonal elements are generally <5% of the main pixel. It isn't totally clear to me whether this is small enough, or even what "enough" actually means, but given this, we should not expect enormous deviations from $\chi^2$ behavior, assuming these statistics are isotropic over the map.


## Run analysis over all observations and Stokes parameters
In the sections above, we ran the analysis only over a single observation in Q. To get a more representative assessment of the data, we calculate the covariance for all observations in Daniel's 2018 EE analysis. 

```python
cov_fnames = glob('/spt/user/adama/covariance_test1/*pkl')
```

```python
for fn in cov_fnames:
    with open(fn, 'rb') as f:
        d = pickle.load(f)
        
    for stokes in ['T', 'Q', 'U']:
        plt.figure()
        plt.imshow(d[stokes])
        plt.colorbar()
        plt.title('_'.join(os.path.splitext(os.path.basename(fn))[0].split('_')[:-1]))
        plt.tight_layout()
        plt.savefig('figures/covariance/cov_{}_{}.png'.format(stokes, d['filename'].split('.')[0]), dpi=200)
        plt.close()
```

## Integrating out variables from a multivariate gaussian
... is as simple as taking the relevant block of the covariance matrix.

```python
from numpy.random import multivariate_normal
from scipy.stats import norm

cov1 = np.array([[1,0.9,0], [0.9,1,0], [0,0,1]])
sample1 = multivariate_normal([0,0,0], cov1, 10000)

cov2 = np.array([[1,0], [0,1]])
sample2 = multivariate_normal([0,0], cov2, 10000)

plt.figure(1)
_ = plt.hist(sample1[:,1], bins=np.linspace(-5,5,101), normed=True)
_ = plt.hist(sample2[:,0], bins=np.linspace(-5,5,101), normed=True)
x = np.linspace(-5,5,101)
plt.plot(x, norm.pdf(x,0,1))
```

```python

```
