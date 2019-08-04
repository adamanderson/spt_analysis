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

# Checking Some Anomalies with Boloprops
Kyle Ferguson and Joshua Sobrin raised some anomalies with the bolometer properties during July 2019. This note looks into some of these issues and tries to understand/correct them. Joshua highlighted four bolometers from the Saturn observation (71819554): '2019.3iy', '2019.5ow', '2019.clp', '2019.xfn'.

Kyle Ferguson also reports that the following bolometers make it into the maps with nan offsets: '2019.jc8', '2019.kbf', '2019.tlp', '2019.u4w', '2019.mda', '2019.4qm', '2019.no2', '2019.of7', '2019.p0a', '2019.xhp', '2019.6v2', '2019.hrx'.

Let's start by checking to see what these bolos look like in the HWM.

```python
import pydfmux
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
```

```python
hwm_file = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm = pydfmux.load_session(open(hwm_file))['hardware_map']
```

```python
joshua_bolo_names = ['2019.3iy', '2019.5ow', '2019.clp', '2019.xfn']
kyle_bolo_names   = ['2019.jc8', '2019.kbf', '2019.tlp', '2019.u4w',
                     '2019.mda', '2019.4qm', '2019.no2', '2019.of7',
                     '2019.p0a', '2019.xhp', '2019.6v2', '2019.hrx']

kyle_bolos = hwm.query(pydfmux.Bolometer).filter(pydfmux.Bolometer.name.in_(kyle_bolo_names))
joshua_bolos = hwm.query(pydfmux.Bolometer).filter(pydfmux.Bolometer.name.in_(joshua_bolo_names))
```

```python
bres = hwm.query(pydfmux.Bolometer).filter(pydfmux.Bolometer.name == '2019.0vs')
bres.pixel_type
```

```python
bps_fname = '/spt/data/bolodata/fullrate/calibrator/81036385/offline_calibration.g3'
bps_nominal_fname = '/spt/data/bolodata/fullrate/calibrator/81036385/nominal_online_cal.g3'
bps = list(core.G3File(bps_fname))[0]["BolometerProperties"]
bps_nominal = list(core.G3File(bps_nominal_fname))[0]["NominalBolometerProperties"]
```

```python
bp = bps['2019.jc8']
bp.x_offset
```

```python
for boloname in bps.keys():
    if np.isfinite(bps[boloname].x_offset) == False:
        print(boloname)
```

```python
nan_bolonames = [bname for bname in bps.keys() if np.isfinite(bps[bname].x_offset) == False]
```

```python
len(nan_bolonames)
```

```python
for name in kyle_bolo_names:
    if name not in nan_bolonames:
        print(name)
        
for name in joshua_bolo_names:
    if name not in nan_bolonames:
        print(name)
```

```python
bps['2019.0vs'].pixel_type
```

## Using updated HWM
I updated the HWM on 30 July, 2019 in order to remove the use of -9999 placeholders for channels that do not have x and y offsets, pixel numbers, and observing bands. Let's check that these got propagated correctly into the new bolometer properties.

```python
bps = list(core.G3File('/home/adama/SPT/spt_analysis/20190501_new_boloprops/60000000_2.g3'))\
            [0]['BolometerProperties']
bps_nominal = list(core.G3File('/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/nominal_online_cal.g3'))\
            [0]["NominalBolometerProperties"]
```

```python
bolonames = np.array([bolo for bolo in bps.keys()])
bands = np.array([bps[bolo].band for bolo in bps.keys()])
pixel_ids = np.array([bps[bolo].pixel_id for bolo in bps.keys()])
x_offsets = np.array([bps[bolo].x_offset for bolo in bps.keys()])
pol_angles = np.array([bps[bolo].pol_angle for bolo in bps.keys()])
pol_angles = np.array([bps[bolo].pol_angle for bolo in bps.keys()])
pixel_type = np.array([bps[bolo].pixel_type for bolo in bps.keys()])
```

```python
print(np.unique(bands[np.isfinite(bands)]))
print(np.unique(pixel_ids))
print(np.unique(pol_angles))
print(np.unique(pixel_type))
```

```python
x_offset_outliers = x_offsets[(x_offsets < -2*core.G3Units.deg) | \
                              (x_offsets > 2*core.G3Units.deg)]
print(x_offset_outliers)
```

```python
bp.pixel_type
```
