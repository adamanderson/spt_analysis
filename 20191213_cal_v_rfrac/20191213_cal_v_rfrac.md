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

# Analysis of cal vs. rfrac data

```python
import numpy as np
from spt3g import core, calibration
import matplotlib.pyplot as plt
from functools import reduce
```

```python
datapath = '/spt/user/production/calibration/calibrator/'
obsids = {'w206': [93507113, 93508217, 93509295, 93510380, 93511444, 93512566]} #,
#           'w174': [93708436, 93709515, 93710641, 93713130, 93714249, 93715328]}
#'w174': [93513647, 93514727, 93515794, 93516849]}
```

```python
bps = list(core.G3File('/spt/data/bolodata/fullrate/calibrator/'
                       '93508217/offline_calibration.g3'))[0]["BolometerProperties"]
```

```python
cal_response = {}
cal_rfrac = {}
bololist = {}

for wafer in obsids.keys():
    cal_response[wafer] = {}
    cal_rfrac[wafer] = {}
    bololist[wafer] = {}
    
    for obsid in obsids[wafer]:
        cal_data = list(core.G3File('{}/{}.g3'.format(datapath, obsid)))[0]
        cal_response[wafer][obsid] = []
        cal_rfrac[wafer][obsid] = []
        bololist[wafer][obsid] = []

        for bolo in cal_data["CalibratorResponse"].keys():
            if bps[bolo].band/core.G3Units.GHz == 90 and \
               bps[bolo].wafer_id == wafer and \
               np.isfinite(cal_data["CalibratorResponse"][bolo]) and\
               np.isfinite(cal_data["CalibratorResponseRfrac"][bolo]):
                cal_response[wafer][obsid].append(cal_data["CalibratorResponse"][bolo])
                cal_rfrac[wafer][obsid].append(cal_data["CalibratorResponseRfrac"][bolo])
                bololist[wafer][obsid].append(bolo)
        
        cal_response[wafer][obsid] = np.array(cal_response[wafer][obsid])
        cal_rfrac[wafer][obsid] = np.array(cal_rfrac[wafer][obsid])
        bololist[wafer][obsid] = np.array(bololist[wafer][obsid])
```

```python
for jwafer, wafer in enumerate(cal_response):
    plt.figure(jwafer)
    
    for obsid in cal_response[wafer]:
        plt.hist(cal_response[wafer][obsid]/(core.G3Units.watt*1e-15),
                 bins=np.linspace(0,3,21),
                 histtype='stepfilled', alpha=0.5,
                 label='rfrac = {:.2f}'.format(np.mean(cal_rfrac[wafer][obsid])))
    plt.title(wafer)
    plt.legend()
    plt.xlabel('(nominal) calibrator response [fW]')
    plt.ylabel('bolometers')
    plt.tight_layout()
    plt.savefig('{}_cal_hist.png'.format(wafer), dpi=150)
```

```python
for jwafer,wafer in enumerate(cal_rfrac.keys()):
    common_bolos = reduce(np.intersect1d, bololist[wafer].values())
    random_bolos = np.random.choice(common_bolos, 10, replace=False)

    plt.figure(jwafer)
    for bolo in random_bolos:
        rfracs    = np.array([cal_rfrac[wafer][obsid][bololist[wafer][obsid]==bolo] \
                              for obsid in cal_rfrac[wafer].keys()])
        responses = np.array([cal_response[wafer][obsid][bololist[wafer][obsid]==bolo] \
                              for obsid in cal_response[wafer].keys()])
        plt.plot(rfracs,
                 responses/(core.G3Units.watt*1e-15), 
                 'o-', label='{}'.format(bolo))
    plt.title(wafer)
    plt.legend()
    plt.ylabel('rfrac')
    plt.ylabel('(nominal) calibrator response [fW]')
    plt.tight_layout()
    plt.savefig('{}_cal_by_bolo.png'.format(wafer), dpi=150)
```

```python
wafer = 'w206'
obsid_0 = 93507113
common_bolos = reduce(np.intersect1d, bololist[wafer].values())

cal_response_0 = np.array([cal_response[wafer][obsid_0][bololist[wafer][obsid_0]==bolo] \
                           for bolo in common_bolos])
for obsid in cal_response[wafer].keys():
    if obsid != obsid_0:
        responses  = np.array([cal_response[wafer][obsid][bololist[wafer][obsid]==bolo] \
                               for bolo in common_bolos])
        plt.hist((responses - cal_response_0)/(core.G3Units.watt*1e-15),
                 bins=np.linspace(-0.5, 1.5, 51), histtype='step',
                 label='{:.2f}'.format(np.mean(responses - cal_response_0)))
plt.legend()
```

```python

```

```python

```
