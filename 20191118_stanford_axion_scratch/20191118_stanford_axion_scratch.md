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

# Scratch Work and Notes on Axions from Stanford Visit

```python
from spt3g import core, mapmaker
from spt3g.mapmaker import remove_weight
import matplotlib.pyplot as plt
import numpy as np
import pickle
```

Question: How many pixels are in one of Daniel's 2-arcmin pixel maps?

```python
map_fname = '/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/' + \
            'high_150GHz_left_maps/55845637/high_150GHz_left_maps_55845637.g3.gz'
d_single = list(core.G3File(map_fname))[5]
```

```python
print(d_single)
```

```python
arr_single_u = np.array(d_single['U'])
print(arr_single_u.shape)
print(len(arr_single_u[arr_single_u != 0]))
plt.imshow(arr_single_u)
```

Question: What does the covariance matrix for a single pixel look like in a single map.

```python
# load the coadd
fname_coadd = '/spt/user/ddutcher/coadds/20190917_full_90GHz.g3.gz'
d_coadd = list(core.G3File(fname_coadd))[0]
```

```python
# remove the weights
_, _, u_single_noweight = remove_weight(d_single['T'], d_single['Q'], d_single['U'], d_single['Wpol'])
_, _, u_coadd_noweight = remove_weight(d_coadd['T'], d_coadd['Q'], d_coadd['U'], d_coadd['Wpol'])
```

```python
plt.figure(figsize=(12,8))
plt.imshow(u_coadd_noweight, vmin=-0.05, vmax=0.05)
plt.colorbar()
plt.axis([1000, 1200, 800, 1000])
```

```python
plt.figure(figsize=(12,8))
plt.imshow(u_coadd_noweight, vmin=-1.0, vmax=1.0)
plt.colorbar()
```

```python
u_single_noweight[850, 1100]
```

```python
plt.figure(figsize=(12,8))
plt.imshow(u_single_noweight, vmin=-1.0, vmax=1.0)
plt.colorbar()
```

```python
u_residual = u_single_noweight - u_coadd_noweight

plt.figure(figsize=(12,8))
plt.imshow(u_residual, vmin=-1.0, vmax=1.0)
plt.colorbar()
```

```python
d_corr = pickle.load(open('correlation.pkl', 'rb'))
```

```python
plt.figure(figsize=(12,8))

plt.imshow(d_corr) #, vmin=-0.1, vmax=0.1)
plt.axis([1050,1150,800,900])
plt.colorbar()
```

```python
plt.figure(figsize=(12,8))

plt.imshow(d_corr) #, vmin=-0.1, vmax=0.1)
plt.axis([1090,1110,840,860])
plt.colorbar()
```

```python
plt.figure(figsize=(12,8))

plt.imshow(d_corr) #, vmin=-0.1, vmax=0.1)
# plt.axis([1090,1110,840,860])
plt.colorbar()
```

```python

```
