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

%matplotlib inline
```

Let's inspect the result of the simulation FITS file after we load it using `load_spt3g_map`.

```python
fname_map = 'lensed_cmb_lmax7000_nside8192_interp0.3_method1_pol_1_sim_65_lensed_map.fits'
map_sim = load_spt3g_map(fname_map)
```

```python
m = map_sim['T']
```

```python
m.angle_to_pixel(0.3, -0.663406)
```

Let's take a look at one of Daniel Dutcher's simulation "stub" files. He has generated a bunch of these in directories like `/spt/user/ddutcher/{source}/{mapmaking setting string}/{obsid}/simstub_*.g3`. The stub files are basically the same as the raw scan files, except that the bolometer timestream data has been removed. This means that the size of the stub file is about 15x smaller than the size of the raw data file.

```python
stubfile = '/spt/user/ddutcher/ra0hdec-44.75/noW201_wafCM_poly19_mhpf300_lr/54667094/' + \
           'simstub_noW201_wafCM_poly19_mhpf300_lr_54667094.g3'
d_stub = [fr for fr in core.G3File(stubfile)]
```

```python
print(d_stub[10])
```

```python
mapfile = '/spt/user/ddutcher/ra0hdec-44.75/noW201_wafCM_poly19_mhpf300_lr/54667094/' + \
          'noW201_wafCM_poly19_mhpf300_lr_54667094.g3'
d_map = [fr for fr in core.G3File(mapfile)]
```

```python
print(d_map[6])
```

```python
bsmap = d_map[6]['T'] #d_map[0]['T']
```

```python
bsmap.pixel_to_angle(1000,1000)
```

```python
Tmap = coordinateutils.FlatSkyMap(4000, 1500, 2.0*core.G3Units.arcmin,
                                  proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                  alpha_center=0*core.G3Units.deg,
                                  delta_center=-60*core.G3Units.deg)
ra = np.zeros((4000, 1500))
dec = np.zeros((4000, 1500))
for xpx in range(4000):
    for ypx in range(1500):
        coords = Tmap.pixel_to_angle(xpx, ypx)
        npx = m.angle_to_pixel(coords[0], coords[1])
        ra[xpx, ypx] = coords[0]
        dec[xpx, ypx] = coords[1]
        if npx < m.shape[0]:
            Tmap[ypx, xpx] = m[npx]
```

```python
plt.figure(figsize=(12,12))
plt.imshow(np.array(Tmap))
```

```python
plt.figure(figsize=(8,8))
plt.imshow(np.array(Tmap))
plt.axis([2000, 2100, 800, 900])
```

```python
weights = core.G3SkyMapWeights(Tmap, weight_type=core.WeightType.Wunpol)
for xpx in range(weights.shape[0]):
    for ypx in range(weights.shape[1]):
        if Tmap[xpx, ypx] != 0:
            weights[xpx, ypx] = np.eye(3)
```

```python
map_fr = core.G3Frame(core.G3FrameType.Map)
map_fr['T'] = Tmap
map_fr['Wunpol'] = weights
```

```python
apod = mapspectra.apodmask.makeBorderApodization(
           map_fr['Wunpol'], apod_type='cos',
           radius_arcmin=15.,zero_border_arcmin=10,
           smooth_weights_arcmin=5)
plt.imshow(apod)
plt.colorbar()
```

```python
cls = calculateCls(map_fr, apod_mask=apod, t_only=True, delta_ell=40)
```

```python
plt.plot(cls['ell'], cls['TT']*cls['ell']*(cls['ell']+1) / (2*np.pi), '.')
plt.xlim([50,2500])
```

```python
np.min(Tmap)
```

```python
dec = np.zeros((4000, 1500))

for x in range(4000):
    for y in range(1500):
        _, dec[x, y] = Tmap.pixel_to_angle(x, y)
```

```python
np.array(Tmap!=0

```

```python

```
