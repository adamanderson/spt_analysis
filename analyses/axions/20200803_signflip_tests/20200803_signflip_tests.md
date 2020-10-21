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

```python
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration, maps
```

```python
fname = 'axion_signflip_test_nolrsplit_ra0hdec-67.25_90GHz_91655356.g3.gz'
mapdata_signflip = list(core.G3File(fname))

fname = 'test_nolrsplit_ra0hdec-67.25_90GHz_91655356.g3.gz'
mapdata = list(core.G3File(fname))
```

```python
stokes = 'Q'

plt.figure(figsize=(12,6))
plt.imshow(mapdata_signflip[6][stokes])
plt.axis([500,1750,150,600])
plt.colorbar()
plt.title('90 GHz {} (signflipped)'.format(stokes))

plt.figure(figsize=(12,6))
plt.imshow(mapdata[6][stokes])
plt.axis([500,1750,150,600])
plt.colorbar()
plt.title('90 GHz {} (NO signflip)'.format(stokes))
```

```python
for stokes in ['T', 'Q', 'U']:
    print('\n{}:'.format(stokes))
    
    pixelvals = np.hstack(np.array(mapdata_signflip[6][stokes]))
    print('RMS: {:.2f}'.format(np.std(pixelvals[pixelvals!=0])))
    
    pixelvals = np.hstack(np.array(mapdata[6][stokes]))
    print('Signflip RMS: {:.2f}'.format(np.std(pixelvals[pixelvals!=0])))
```

```python

```
