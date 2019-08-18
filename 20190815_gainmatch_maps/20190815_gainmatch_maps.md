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

# Low-Ell Gain-Matching Maps


The purpose of this note is to test out the improvement in low-$\ell$ noise performance from doing gain-matching.

```python
from spt3g import core, calibration, mapmaker
from spt3g.mapspectra import map_analysis
import numpy as np
import matplotlib.pyplot as plt
import pickle
```

## Some notes on map runs
In the course of writing this note, I did several runs of mapmaking. These are documented below:

* **Test 1:** Used settings in `low_ell_test1.yaml`. Used observations in `obsids_to_process_test1.txt`. Same as Daniel's mapmaking parameters, but no high-pass filter applied and a low-pass of $\ell=6000$ instead of $\ell=6600$. No left-right split. The command executed on the grid by condor is (for obsid 82517444):
# ```
$PWD/code/env-shell.sh python master_field_mapmaker.py 82517444.g3 0000.g3 0001.g3 0002.g3 0003.g3 0004.g3 -o 1500d_maps_ra0hdec-67.25_82517444.g3.gz -z --config-file low_ell_test1.yaml 
# ```

* **Test 2:** Fully the same settings as Daniel (`low_ell_test2.yaml`). Includes a left-right split.
# ```
$PWD/code/env-shell.sh python master_field_mapmaker.py 82517444.g3 0000.g3 0001.g3 0002.g3 0003.g3 0004.g3 -o 1500d_maps_ra0hdec-67.25_82517444.g3.gz -z --config-file low_ell_test2.yaml --split-left-right both
# ```


## Playing with Maps
Let's start out by looking at a map run with no masked high-pass filter. This should give very similar results to Jessica's "low-ell" note. Let's start out by just making some plots to make sure that things look okay visually.

```python
d = list(core.G3File('/spt/user/adama/20190815_1500d_maps_20190801_to_20190815_test2/'
                     'downsampled/1500d_maps_coadded.g3'))
for fr in d:
    print(fr)
```

```python
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(d[1]["U"] - d[4]["U"], vmin=-2000, vmax=2000)
plt.axis([1000, 1200, 600, 800])
plt.title('U noise (L-R)')

plt.subplot(1,2,2)
plt.imshow(d[1]["U"] + d[4]["U"], vmin=-2000, vmax=2000)
plt.axis([1000, 1200, 600, 800])
plt.title('U signal (L+R)')


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(d[1]["T"] - d[4]["T"], vmin=-10000, vmax=10000)
plt.axis([1000, 1200, 600, 800])
plt.title('T noise (L-R)')

plt.subplot(1,2,2)
plt.imshow((d[1]["T"] + d[4]["T"]), vmin=-10000, vmax=10000)
plt.axis([1000, 1200, 600, 800])
plt.title('T signal (L+R)')
```

Construct sum and difference maps and take power spectra.

```python
frame_diff = core.G3Frame(core.G3FrameType.Map)
frequency = '90GHz'
frame_diff['Id'] = frequency
for stokes in ['T', 'Q', 'U']:
    frame_diff[stokes] = d[0][stokes] + -1*d[3][stokes]
frame_diff['Wpol'] = d[0]['Wpol'] + d[3]['Wpol']
```

```python
print(frame_diff)
```

```python
with open('/spt/user/ddutcher/masks/3band_res2_bundle_total_mask.pkl', 'rb') as f:
    apod_mask = pickle.load(f)
```

```python
plt.figure(figsize=(8,6))
plt.imshow(apod_mask)
```

```python
spectra = map_analysis.calculate_powerspectra(d[0], l_min=20, l_max=5000,
                                              delta_l=25, apod_mask=apod_mask)
```

```python
plt.figure(figsize=(10,6))
for pol in ['TT', 'EE']:
    ms = spectra[pol]
    plt.loglog(ms.bin_centers, np.array(ms.get_dl()) / \
                 (core.G3Units.microkelvin)**2)
# plt.xlim([0,1000])
```

```python
import numpy as np
from spt3g import core, coordinateutils
from spt3g.mapspectra import basicmaputils, map_analysis
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum1DDict

dx = 40 * core.G3Units.arcmin
dy = dx
tmap = (
    np.random.standard_normal([75, 113])
    * 1
    * core.G3Units.uK
    * core.G3Units.arcmin
    / (np.sqrt(dx * dy))
)
tmap = coordinateutils.FlatSkyMap(
    tmap,
    dx,
    proj=coordinateutils.MapProjection.Proj5,
    alpha_center=0.0,
    delta_center=-57.5 * core.G3Units.deg,
    coord_ref=core.MapCoordReference.Equatorial,
    is_weighted=False,
)
maps = {"T": tmap, "Q": tmap, "U": tmap}

cls = map_analysis.calculate_powerspectra(maps, flatten=True, apod_mask=None,
                                          l_min=10, l_max=500, delta_l=20)
```

```python
plt.imshow(tmap)
```

```python
plt.plot(cls['EE'].bin_centers, cls['EE'].get_cl() / \
         (core.G3Units.arcmin*core.G3Units.microkelvin)**2)
```

```python
plt.plot(ells, dl['TT'])
```

```python
print(cls)
```

```python
np.random.standard_normal([75, 113])
```

```python

```
