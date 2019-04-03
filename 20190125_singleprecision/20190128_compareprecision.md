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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from spt3g import core, mapmaker, dfmux, calibration
```

```python
df_single = {fr['Id']: fr for fr in core.G3File('singleprecision/64502043_maps.g3')
             if 'Id' in fr.keys()}
df_double = {fr['Id']: fr for fr in core.G3File('doubleprecision/64502043_maps_v0.g3')
             if 'Id' in fr.keys()}
```

```python
for fr in df_double:
    print(fr)
```

```python
plt.figure(figsize=(12,6))
plt.imshow(np.abs(df_single['90GHz']['T'] - df_double['90GHz']['T']) / \
           df_double['90GHz']['T'],
           vmin=0, vmax=1.0)
plt.ylim([100,600])
plt.axis([800, 850, 400, 450])
plt.colorbar()
plt.tight_layout()
```

```python
plt.figure(figsize=(12,6))
plt.imshow(df_single['Left-90GHz']['T'] - df_double['Left-90GHz']['T'])
plt.ylim([100,600])
plt.axis([800, 850, 400, 450])
plt.colorbar()
plt.tight_layout()
```

```python
for jband, band in enumerate(list(df_single.keys())):
    if band != 'bsmap':
        plt.figure(jband, figsize=(15,4))
        plt.subplot(1,3,1)
        plt.imshow(df_single[band]['T'], vmin=-2000, vmax=2000)
        plt.title('{}: single-precision'.format(band))
        plt.axis([800, 850, 400, 450])
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(df_double[band]['T'], vmin=-2000, vmax=2000)
        plt.title('{}: double-precision'.format(band))
        plt.axis([800, 850, 400, 450])
        plt.colorbar()
        plt.tight_layout()
        plt.subplot(1,3,3)
        plt.imshow(df_single[band]['T'] - df_double[band]['T'], vmin=-20, vmax=20)
        plt.title('{}: single - double'.format(band))
        plt.axis([800, 850, 400, 450])
        plt.colorbar()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{}_diff_zoom.png'.format(band), dpi=200)
```

```python
for jband, band in enumerate(list(df_single.keys())):
    if band != 'bsmap':
        plt.figure(jband, figsize=(16,9))
        plt.subplot(3,1,1)
        plt.imshow(df_single[band]['T'], vmin=-2000, vmax=2000)
        plt.title('{}: single-precision'.format(band))
        plt.ylim([100,600])
        plt.colorbar()
        plt.subplot(3,1,2)
        plt.imshow(df_double[band]['T'], vmin=-2000, vmax=2000)
        plt.title('{}: double-precision'.format(band))
        plt.ylim([100,600])
        plt.colorbar()
        plt.tight_layout()
        plt.subplot(3,1,3)
        plt.imshow(df_single[band]['T'] - df_double[band]['T'], vmin=-20, vmax=20)
        plt.title('{}: single - double'.format(band))
        plt.ylim([100,600])
        plt.colorbar()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{}_diff_full.png'.format(band), dpi=200)
```

```python
plt.imshow(df_double['220GHz']['T'],
           vmin=0, vmax=1.0)
```

```python
df_single['GHz']['T'][20000]
```

```python

```
