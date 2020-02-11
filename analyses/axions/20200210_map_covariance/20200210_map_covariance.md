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
from glob import glob
```

```python
fname_maps = glob('/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/' + \
                  'high_150GHz_left_maps/*/high_150GHz_left_maps_*.g3.gz')
```

```python
mapdata = list(core.G3File(fname_maps[0]))[-1]
```

```python
T_noweight, Q_noweight, U_noweight = remove_weight(mapdata['T'], mapdata['Q'], mapdata['U'], mapdata['Wpol'])
```

```python
plt.figure(figsize=(15,10))
plt.imshow(mapdata['Q'])

plt.figure(figsize=(15,10))
plt.imshow(Q_noweight)
```

In the foregoing analysis, we scan over pixels in the map and construct a cutout centered on each one. If the cutout does not have any nan pixels, then we use it to estimate one sample of the pixel-pixel covariance. We then average all the samples together to form the estimate of a local block in the pixel-pixel covariance.

```python
cutout_dim = 1*core.G3Units.deg
cutout_size = int(cutout_dim / Q_noweight.res)
if cutout_size % 2:
    cutout_size += 1

Q_arr = np.array(mapdata['Q'])
cutout_running = np.zeros((cutout_size, cutout_size))

for i in range(Q_arr.shape[0] - cutout_size):
    for j in range(Q_arr.shape[1] - cutout_size):
        cutout = Q_arr[i:(i+cutout_size), j:(j+cutout_size)]
        cutout_center = cutout[int((cutout_size-1) / 2),
                               int((cutout_size-1) / 2)]
#         if np.all(np.isfinite(cutout)):
#             cutout_running += cutout * cutout_center
        if np.all(cutout != 0):
            cutout_running += cutout * cutout_center
```

```python
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cutout_running / cutout_running.max(),
           norm=colors.SymLogNorm(linthresh=0.001, linscale=1, vmin=-1, vmax=1))
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(cutout_running / cutout_running.max(), vmin=-1, vmax=1)
plt.colorbar()
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
Q_rebinned = mapdata['Q'].rebin(2)

cutout_dim = 1*core.G3Units.deg
cutout_size = int(cutout_dim / Q_rebinned.res)
if cutout_size % 2:
    cutout_size += 1

Q_arr = np.array(Q_rebinned)
cutout_running = np.zeros((cutout_size, cutout_size))

for i in range(Q_arr.shape[0] - cutout_size):
    for j in range(Q_arr.shape[1] - cutout_size):
        cutout = Q_arr[i:(i+cutout_size), j:(j+cutout_size)]
        cutout_center = cutout[int((cutout_size-1) / 2),
                               int((cutout_size-1) / 2)]
#         if np.all(np.isfinite(cutout)):
#             cutout_running += cutout * cutout_center
        if np.all(cutout != 0):
            cutout_running += cutout * cutout_center
```

```python
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cutout_running / cutout_running.max(),
           norm=colors.SymLogNorm(linthresh=0.001, linscale=1, vmin=-1, vmax=1))
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(cutout_running / cutout_running.max(), vmin=-1, vmax=1)
plt.colorbar()
```

```python
np.sort(np.abs(np.hstack(cutout_running / cutout_running.max())))[-10:]
```

Based on the above, we can conclude that off-diagonal elements are generally <5% of the main pixel. It isn't totally clear to me whether this is small enough, or even what "enough" actually means, but given this, we should not expect enormous deviations from $\chi^2$ behavior, assuming these statistics are isotropic over the map.

```python

```
