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

# Testing Sign-flip Noise
I have implemented a way of generating many noise realizations of a single observation by flipping the signs of scans individually. In this notebook, we check whether the method works as expected by comparing to standard L-R map noise estimations.

```python
from spt3g import core, maps, calibration
from spt3g.mapspectra.map_analysis import calculate_powerspectra
from spt3g.std_processing.mapmakers.master_field_coadder import add_two_map_frames
import numpy as np
import matplotlib.pyplot as plt
```

## Noise with standard L-R noise

```python
d_normal = list(core.G3File('84203292_test_map.g3.gz'))
```

```python
plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 1, jfr-4)
    plt.imshow(d_normal[jfr]['T'] - d_normal[jfr+3]['T'])
    plt.colorbar()
plt.tight_layout()
```

```python
plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 2, 2*(jfr-5) + 1)
    plt.imshow(d_normal[jfr]['T'])
    plt.colorbar()
    
    plt.subplot(3, 2, 2*(jfr-5) + 2)
    plt.imshow(d_normal[jfr+3]['T'])
    plt.colorbar()
plt.tight_layout()
```

```python
# WEIGHT MAPS

plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 2, 2*(jfr-5) + 1)
    plt.imshow(d_normal[jfr]['Wpol'].TT)
    plt.title(d_normal[jfr]['Id'])
    plt.colorbar()
    
    plt.subplot(3, 2, 2*(jfr-5) + 2)
    plt.imshow(d_normal[jfr+3]['Wpol'].TT)
    plt.title(d_normal[jfr+3]['Id'])
    plt.colorbar()
plt.tight_layout()
```

## Sign-flip noise

```python
d_flipped = list(core.G3File('84203292_test_map_signflip2.g3.gz'))
```

```python
plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 1, jfr-4)
    plt.imshow(d_flipped[jfr]['T'] + d_flipped[jfr+3]['T'])
    plt.colorbar()
plt.tight_layout()
```

```python
plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 2, 2*(jfr-5) + 1)
    plt.imshow(d_flipped[jfr]['T'])
    plt.title(d_flipped[jfr]['Id'])
    plt.colorbar()
    
    plt.subplot(3, 2, 2*(jfr-5) + 2)
    plt.imshow(d_flipped[jfr+3]['T'])
    plt.title(d_flipped[jfr+3]['Id'])
    plt.colorbar()
plt.tight_layout()
```

```python
# WEIGHT MAPS

plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 2, 2*(jfr-5) + 1)
    plt.imshow(d_flipped[jfr]['Wpol'].TT)
    plt.title(d_flipped[jfr]['Id'])
    plt.colorbar()
    
    plt.subplot(3, 2, 2*(jfr-5) + 2)
    plt.imshow(d_flipped[jfr+3]['Wpol'].TT)
    plt.title(d_flipped[jfr+3]['Id'])
    plt.colorbar()
plt.tight_layout()
```

## Difference between sign-flip and L-R maps
Let's check that the maps are actually different and that we didn't screw something obvious up.

```python
plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 1, jfr-4)
    plt.imshow(d_normal[jfr]['T'] - d_normal[jfr+3]['T'] - (d_flipped[jfr]['T'] + d_flipped[jfr+3]['T']))
    plt.colorbar()
plt.tight_layout()
```

```python
plt.figure(figsize=(12, 15))
for jfr in np.arange(5,8):
    plt.subplot(3, 1, jfr-4)
    plt.imshow(d_normal[jfr+3]['T'] - d_flipped[jfr+3]['T'])
    plt.colorbar()
plt.tight_layout()
```

## Power spectra
Let's next check that the noise properties of the two maps are comparable.

```python
d_flipped = list(core.G3File('84203292_test_map_signflip2.g3.gz'))
map_left = d_flipped[5]
map_right = d_flipped[8]
# maps.RemoveWeights(map_left)
# maps.RemoveWeights(map_right)
add_two_map_frames(map_left, map_right)
map_added = map_left
flipped_spectra = calculate_powerspectra(map_added, delta_l = 25, lmin=100, lmax=6000)
```

```python
d_normal = list(core.G3File('84203292_test_map.g3.gz'))
map_left = d_normal[5]
map_right = d_normal[8]
add_two_map_frames(map_left, map_right, subtract=True)
map_added = map_left
normal_spectra = calculate_powerspectra(map_added, delta_l = 25, lmin=100, lmax=6000)
```

```python
plt.figure(figsize=(12,8))
for jspectra, spectra in enumerate(['TT', 'TE', 'EE', 'BB']):
    plt.subplot(2,2,jspectra+1)
    if spectra == 'TE':
        plt.semilogx(flipped_spectra[spectra].bin_centers,
                     flipped_spectra[spectra].get_cl() / (core.G3Units.microkelvin**2),
                     label='per-scan sign-flip')
        plt.semilogx(normal_spectra[spectra].bin_centers,
                     normal_spectra[spectra].get_cl() / (core.G3Units.microkelvin**2),
                     label='L-R')
    else:
        plt.loglog(flipped_spectra[spectra].bin_centers,
                   flipped_spectra[spectra].get_cl() / (core.G3Units.microkelvin**2),
                   label='per-scan sign-flip')
        plt.loglog(normal_spectra[spectra].bin_centers,
                   normal_spectra[spectra].get_cl() / (core.G3Units.microkelvin**2),
                   label='L-R')
        plt.ylim([1e-3, 2e-1])
    plt.title(spectra)
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$ [$\mu$K$^2$]')
plt.subplot(2,2,1)
plt.legend()
plt.tight_layout()
plt.savefig('signflip_noise_example.png')
```

```python
ms = flipped_spectra['TT']
```

```python
ms.
```
