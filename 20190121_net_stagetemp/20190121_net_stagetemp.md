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
obsids = [63084862, 63661173, 64576620, 64591411, 64606397]
UChead_temp = {63084862: 0.280, # 1 Jan overnight
               63661173: 0.275,
               64576620: 0.269,
               64591411: 0.300,
               64606397: 0.300}
```

```python
data = {}
for obsid in obsids:
    d = [fr for fr in core.G3File('noise_corrected/{}_processed_noise.g3'
                                 .format(obsid))]
    bps = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/{}/offline_calibration.g3'
                                    .format(obsid))][0]["BolometerProperties"]
    data[obsid] = plot_noise(d[0], bps, obsid, bywafer=True)
```

```python
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']

for band in [90, 150, 220]:
    print('{} GHz\t\t\t'.format(band) + '\t'.join(wafer_list))
    for obsid in obsids:
        dstring = '{} ({:.3f}):\t'.format(obsid, UChead_temp[obsid])
        for wafer in wafer_list:
            dstring = dstring + '{:.1f}\t'.format(np.median(data[obsid][wafer][band]))
        print(dstring)
    print('')
```

```python
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']
net_low = {90:300, 150:200, 220:800}
for band in [90, 150, 220]:
    print('{} GHz\t\t\t'.format(band) + '\t'.join(wafer_list) + '\tall')
    for obsid in obsids:
        dstring = '{} ({:.3f}):\t'.format(obsid, UChead_temp[obsid])
        all_nets = np.array([])
        for wafer in wafer_list:
            nets = data[obsid][wafer][band]
            total_net = np.sqrt(1.0 / np.sum(1.0 / nets[nets>net_low[band]]**2.0))
            dstring = dstring + '{:.1f}\t'.format(total_net)
            all_nets = np.append(all_nets, nets[nets>net_low[band]])
        total_net = np.sqrt(1.0 / np.sum(1.0 / all_nets**2.0))
        dstring = dstring + '{:.1f}'.format(total_net)
        print(dstring)
    print('')
```

```python
# overlay observations by wafer and band
obsid_pairs = [[64576620, 63084862],
               [64576620, 63661173],
               [64576620, 64591411],
               [63084862, 64591411],
               [64576620, 64606397]]
labels = [['64576620 (head = 269mK)', '63084862 (head = 280mK)'],
          ['64576620 (head = 269mK)', '63661173 (head = 275mK)'],
          ['64576620 (head = 269mK)', '64591411 (head = 300mK)'],
          ['63084862 (head = 280mK)', '64591411 (head = 300mK)'],
          ['64576620 (head = 269mK)', '64606397 (head = 300mK)']]

for jpair, obsids in enumerate(obsid_pairs):
    print(obsids)
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
plt.close('all')
```

```python
len(all_nets)
```

```python
26 / 3.
```

```python

```
