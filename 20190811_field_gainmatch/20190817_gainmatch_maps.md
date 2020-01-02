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
from spt3g import core, mapmaker, calibration
import numpy as np
import matplotlib.pyplot as plt
import os.path
```

## Non-gain-matched maps

```python
mapdir = '/spt/user/adama/20190815_1500d_maps_20190801_to_20190815_test1/downsampled/'
d = list(core.G3File(os.path.join(mapdir, '1500d_maps_ra0hdec-44.75_81783460.g3.gz')))
```

```python
print(d[5])
```

```python
plt.figure(figsize=(12,8))
plt.imshow(d[0]['U'], vmin=-2000, vmax=2000)
plt.axis([1000,1100,600,700])
```

```python
np.std(d[6]['T'])
```

```python

```
