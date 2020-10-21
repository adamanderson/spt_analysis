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
from spt3g import core, mapmaker
import numpy as np
import matplotlib.pyplot as plt
import os.path
```

```python
sim_path = '/spt/user/kferguson/condor_output/noise_sims_for_axion_analysis'
```

```python
dsim = list(core.G3File(os.path.join(sim_path, 'mock_observed_sim_0085_150GHz_ra0hdec-52.g3.gz')))
ddata = list(core.G3File('/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/high_150GHz_left_maps'
                         '/51362637/high_150GHz_left_maps_51362637.g3.gz'))
```

```python
plt.imshow(dsim[7]['Q'])
plt.colorbar()
```

```python
plt.imshow(ddata[-1]['Q'])
plt.colorbar()
```

```python
arr_data   = np.hstack(np.array(ddata[-1]['Q']))
arr_data   = arr_data[(arr_data!=0) & np.isfinite(arr_data)]
arr_sim    = np.hstack(np.array(dsim[7]['Q']))
arr_sim    = arr_sim[(arr_sim!=0) & np.isfinite(arr_sim)]

plt.figure(1)
_ = plt.hist(arr_data, bins=np.linspace(-2,2,101), histtype='step', label='data')
_ = plt.hist(arr_sim, bins=np.linspace(-2,2,101), histtype='step', label='sim')
plt.legend()
plt.title('pixel values, obsid 51362637')
plt.tight_layout()
plt.savefig('pixel_values_51362637.png', dpi=200)
```

```python
len(arr_data)
```

```python
len(arr_sim)
```

```python
for fr in dsim:
    print(fr)
```

```python
ls /spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/high_150GHz_left_maps
```

```python

```
