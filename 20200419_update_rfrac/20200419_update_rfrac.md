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

# Update rfrac
**Original date:** 19 April 2020  
**Author:** Adam Anderson

In mid-April, we experienced some latching of detectors on w206. This was an occasional phenomenon during 2019, and continues in 2020. All evidence points to it being driven by fluctuations in weather. While the effective livetime lost is small, we decided to try to fix the problem by adjusting the rfracs up by 0.05 on w206 (all bands). This note makes this tweak and performs a few checks.

```python
import pydfmux
import numpy as np
import pandas as pd
import os.path
```

```python
hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm = pydfmux.load_session(open(hwm_fname, 'r'))['hardware_map']

bolonames = np.array([b.name for b in bolos])
rfracs = np.array([b.rfrac for b in bolos])
wafers = np.array([b.wafer.name for b in bolos])
bands = np.array([b.observing_band for b in bolos])

wafer_list = np.unique(wafers)
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
# adjust rfracs from default
new_rfracs = {('w172', 90): 0.7,
              ('w174', 90): 0.7,
              ('w180', 90): 0.7,
              ('w181', 90): 0.7,
              ('w188', 90): 0.7,
              ('w204', 90): 0.7,
              ('w206', 90): 0.75,
              ('w172', 150): 0.7,
              ('w174', 150): 0.7,
              ('w180', 150): 0.7,
              ('w181', 150): 0.7,
              ('w188', 150): 0.7,
              ('w204', 150): 0.7,
              ('w206', 150): 0.75,
              ('w172', 220): 0.7,
              ('w206', 220): 0.85}

for waferband_pair, new_rfrac in new_rfracs.items():
    rfracs[(wafers==waferband_pair[0]) & \
           (bands==waferband_pair[1])] = new_rfrac
```

```python
# save the new custom bolo properties to a csv file
dict_tofile = {'name':bolonames, 'rfrac': rfracs}
df_tofile = pd.DataFrame(dict_tofile)
df_tofile.to_csv('rfracs_high_el_310mK.csv', sep='\t', index=False)
```

## Cross-check
Let's check the results of the hardware map that we created using the rfracs above.

```python
hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm_check = pydfmux.load_session(open(hwm_fname, 'r'))['hardware_map']

bolos_check = hwm_check.query(pydfmux.Bolometer)

bolonames_check = np.array([b.name for b in bolos_check])
rfracs_check = np.array([b.rfrac for b in bolos_check])
wafers_check = np.array([b.wafer.name for b in bolos_check])
bands_check = np.array([b.observing_band for b in bolos_check])
```

```python
# check everything
for band in [90, 150, 220]:
    print('{} GHZ:'.format(band))
    for wafer in wafer_list:
        print('{}: {:.4f}'.format(wafer,
                                  np.mean(rfracs_check[(wafers_check==wafer) & \
                                                       (bands_check==band)])))
```

```python

```
