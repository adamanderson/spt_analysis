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
from adama_utils import plot_noise
from spt3g import core, dfmux, calibration
import numpy as np
```

```python
obsids = [64576620, 64591411, 64606397]
```

```python
for obsid in obsids:
    d = [fr for fr in core.G3File('noise_corrected/{}_processed_noise.g3'
                                 .format(obsid))]
    bps = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/{}/offline_calibration.g3'
                                    .format(obsid))][0]["BolometerProperties"]
    plot_noise(d[0], bps, obsid, bywafer=True)
```

```python

```
