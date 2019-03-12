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

# Updates to rfracs
The takeaway from some work of Allen is that the 150 GHz detectors on some wafers W172, W174, W181, W203, W206 are insufficiently linear. This note generates new rfracs for the hardware map that hopefully improve this issue.

```python
import pydfmux
import numpy as np
import pandas as pd
```

```python
hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm = pydfmux.load_session(open(hwm_fname, 'r'))['hardware_map']
```

```python
bolos = hwm.query(pydfmux.Bolometer)
```

```python
bolonames = np.array([b.name for b in bolos])
rfracs = np.array([b.rfrac for b in bolos])
wafers = np.array([b.wafer.name for b in bolos])
bands = np.array([b.observing_band for b in bolos])

wafer_list = np.unique(wafers)
```

```python
# set default value of rfrac to 0.7
rfracs[:] = 0.8
```

```python
# set w176 and w177 to rfrac=0.8 in all bands
rfracs[(wafers=='w176') & (wafers=='w177')] = 0.8
```

```python
# set all other wafers to rfrac=0.7 at 90 and 150 GHz
for wafer in wafer_list:
    if wafer != 'w176' and wafer != 'w177':
        for band in [90, 150]:
            rfracs[(wafers==wafer) & (bands==band)] = 0.7
```

```python
# set all 150 GHz detectors on w172, w174, w181, w203, w206 to rfrac=0.6
for wafer in ['w172', 'w174', 'w181', 'w203', 'w206']:
    rfracs[(wafers==wafer) & (bands==150)] = 0.6
```

```python
# check everything
for band in [90, 150, 220]:
    print('{} GHZ:'.format(band))
    for wafer in wafer_list:
        print('{}: {:.4f}'.format(wafer,
                                  np.mean(rfracs[(wafers==wafer) & (bands==band)])))
```

```python
dict_tofile = {'name':bolonames, 'rfrac': rfracs}
df_tofile = pd.DataFrame(dict_tofile)
df_tofile.to_csv('rfrac_150ghz_06.csv', sep='\t', index=False)
```

## Cross-check
Let's check the results of the hardware map that we created using the rfracs above.

```python
hwm_fname_check = '/home/adama/SPT/hardware_maps_southpole/' + \
                  '2019/hwm_pole_rfractest/hwm.yaml'
hwm_check = pydfmux.load_session(open(hwm_fname_check, 'r'))['hardware_map']
```

```python
# check everything
bolos_check = hwm_check.query(pydfmux.Bolometer)

bolonames_check = np.array([b.name for b in bolos_check])
rfracs_check = np.array([b.rfrac for b in bolos_check])
wafers_check = np.array([b.wafer.name for b in bolos_check])
bands_check = np.array([b.observing_band for b in bolos_check])

for band in [90, 150, 220]:
    print('{} GHZ:'.format(band))
    for wafer in wafer_list:
        print('{}: {:.4f}'.format(wafer,
                                  np.mean(rfracs_check[(wafers_check==wafer) & \
                                                       (bands_check==band)])))
```

```python

```
