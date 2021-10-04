---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import matplotlib.pyplot as plt
import numpy as np
from spt3g import core, mapmaker, calibration, maps, mapspectra
from glob import glob
import os
```

```python
data_dir = '/home/adama/SPT/spt_analysis/analyses/axions/20210917_subset_coadds/'
fnames = np.sort(glob(os.path.join(data_dir, 'coadd_ra0hdec-52.25_90GHz_*.g3.gz')))

for fname in fnames:
    print(fname)
    dmaps = list(core.G3File(fname))
    fr_maps = dmaps[0]
    maps.RemoveWeights(fr_maps)

    plt.figure(figsize=(12,20))
    plt.subplot(3,1,1)
    plt.imshow(fr_maps['T'],
               vmin=-500*core.G3Units.microkelvin,
               vmax=500*core.G3Units.microkelvin)
#     plt.axis([0,500,400,900])
    
    plt.subplot(3,1,2)
    plt.imshow(fr_maps['Q'],
               vmin=-400*core.G3Units.microkelvin,
               vmax=400*core.G3Units.microkelvin)
#     plt.axis([0,500,400,900])
    
    plt.subplot(3,1,3)
    plt.imshow(fr_maps['U'],
               vmin=-400*core.G3Units.microkelvin,
               vmax=400*core.G3Units.microkelvin)
#     plt.axis([0,500,400,900])
    
    plt.title(os.path.basename(fname))
#     plt.tight_layout()
```

```python
data_dir = '/home/adama/SPT/spt_analysis/analyses/axions/20210917_subset_coadds/'
fnames = np.sort(glob(os.path.join(data_dir, 'coadd_ra0hdec-52.25_90GHz_*.g3.gz')))

for fname in fnames:
    print(fname)
    dmaps = list(core.G3File(fname))
    fr_maps = dmaps[0]
    spectra = mapspectra.map_analysis.calculate_powerspectra(fr_maps,
                                                             qu_eb='qu',
                                                             delta_l=25)
    
    plt.figure(1)
    plt.plot(spectra['QQ'], label=os.path.basename(fname).split('_')[3])
    plt.title('QQ')
    plt.legend()
    plt.tight_layout()
    
    plt.figure(2)
    plt.plot(spectra['UU'], label=os.path.basename(fname).split('_')[3])
    plt.title('UU')
    plt.legend()
    plt.tight_layout()
```

```python
data_dir = '/home/adama/SPT/spt_analysis/analyses/axions/20210917_subset_coadds/'
fnames = np.sort(glob(os.path.join(data_dir, 'coadd_ra0hdec-59.75_90GHz_*.g3.gz')))

for fname in fnames:
    print(fname)
    dmaps = list(core.G3File(fname))
    fr_maps = dmaps[0]
    spectra = mapspectra.map_analysis.calculate_powerspectra(fr_maps,
                                                             qu_eb='qu',
                                                             delta_l=25)
    
    plt.figure(1)
    plt.plot(spectra['QQ'], label=os.path.basename(fname).split('_')[3])
    plt.title('QQ')
    plt.legend()
    plt.tight_layout()
    
    plt.figure(2)
    plt.plot(spectra['UU'], label=os.path.basename(fname).split('_')[3])
    plt.title('UU')
    plt.legend()
    plt.tight_layout()
```

```python
maps.RemoveWeights(fr_coadd)
```

```python
plt.figure(figsize=(12,8))
plt.imshow(fr_coadd['Q'],
           vmin=-50*core.G3Units.microkelvin,
           vmax=50*core.G3Units.microkelvin)
```

```python
mm = fr_maps['T']
```

```python
mm.extract_patch?
```

```python
plt.figure(figsize=(12,4))
for jpol, pol in enumerate(['T', 'Q', 'U']):
    mm = fr_maps[pol]
    xy = mm.angle_to_xy(alpha=32.25*core.G3Units.deg, delta=-51.02*core.G3Units.deg)
    patch = mm.extract_patch(int(xy[0]), int(xy[1]), width=20, height=20)
    plt.subplot(1, 3, jpol+1)
    plt.imshow(patch)
    plt.colorbar()
    plt.title(pol)
plt.savefig('focus_quasar_tqu.png', dpi=200)
```

```python
plt.imshow(patch)
```

```python

```
