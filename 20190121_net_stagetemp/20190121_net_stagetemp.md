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
from adama_utils import plot_noise, plot_noise_comparison
from spt3g import core, dfmux, calibration
import numpy as np
```

```python
obsids = [64576620, 64591411, 64606397]
UChead_temp = {64576620: 0.269,
               64591411: 0.300,
               64606397: 0.300}
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
# overlay observations by wafer and band
obsid_pairs = [[64576620, 64591411],
               [64576620, 64606397]]
labels = [['64576620 (head = 269mK)', '64591411 (head = 300mK)'],
          ['64576620 (head = 269mK)', '64606397 (head = 300mK)']]

for jpair, obsids in enumerate(obsid_pairs):
    frame_list = []
    boloprops_list = []
    for oid in obsids:
        d = [fr for fr in core.G3File('noise_corrected/{}_processed_noise.g3'
                                      .format(oid))][0]
        frame_list.append(d)
        bps = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/{}/offline_calibration.g3'
                                       .format(oid))][0]["BolometerProperties"]
        boloprops_list.append(bps)
    plot_noise_comparison(frame_list, boloprops_list, obsids, bywafer=True, legend_labels=labels[jpair],
                          filestub='NETvTemp')
    
```

```python

```
