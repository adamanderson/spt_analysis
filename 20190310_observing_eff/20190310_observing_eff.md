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
from spt3g import core, dfmux, calibration
from spt3g.std_processing import obsid_to_g3time
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from glob import glob
from datetime import datetime
```

```python
testfile = '/spt/data/bolodata/downsampled/ra0hdec-44.75/68757610/0001.g3'
dtest = [fr for fr in core.G3File(testfile)]
```

```python
print(dtest[0])
```

```python
(dtest[0]["ObservationStop"].time - \
 dtest[0]["ObservationStart"].time) / core.G3Units.sec
```

```python
def calc_observing_eff(tstart, tstop):
    # find all observations with obsids that fall between start and stop
    obsids_all = []
    for source in ['ra0hdec-44.75', 'ra0hdec-52.25',
                   'ra0hdec-59.75', 'ra0hdec-67.25']:
        dirnames = glob('/spt/data/bolodata/downsampled/{}/*'.format(source))
        obsids = np.array([int(dirname.split('/')[-1]) for dirname in dirnames])
        times = np.array([obsid_to_g3time(obsid).time/core.G3Units.second
                          for obsid in obsids])
        obsids_all = np.append(obsids_all, obsids[(times < tstop) & (times > tstart)])
        return obsids_all
```

```python
tstart = datetime(year=2018, month=9, day=1).timestamp()
tstop = datetime(year=2018, month=9, day=30).timestamp()
```

```python
calc_observing_eff(mar1.timestamp(), now.timestamp())
```

```python
# find all observations with obsids that fall between start and stop
obsids_all = []
dirnames_all = []
for source in ['ra0hdec-44.75', 'ra0hdec-52.25',
               'ra0hdec-59.75', 'ra0hdec-67.25']:
    dirnames = np.array(glob('/spt/data/bolodata/downsampled/{}/*'.format(source)))
    obsids = np.array([int(dirname.split('/')[-1]) for dirname in dirnames])
    times = np.array([obsid_to_g3time(obsid).time/core.G3Units.second
                      for obsid in obsids])
    obsids_all = np.append(obsids_all, obsids[(times < tstop) & (times > tstart)])
    dirnames_all = np.append(dirnames_all, dirnames[(times < tstop) & (times > tstart)])
    
#     obsids = np.array([int(dirname.split('/')[-1]) for dirname in dirnames])
#     times = np.array([obsid_to_g3time(obsid).time/core.G3Units.second
#                       for obsid in obsids])

obs_tstart = []
obs_tstop = []
for dirname in dirnames_all:
    print(dirname)
    f = core.G3File('{}/0000.g3'.format(dirname))
    fr = f.next()
    obs_tstart.append(fr["ObservationStart"].time/core.G3Units.second)
    obs_tstop.append(fr["ObservationStop"].time/core.G3Units.second)
obs_tstart = np.sort(np.array(obs_tstart))
obs_tstop = np.sort(np.array(obs_tstop))
obs_tall = np.sort(np.append(obs_tstart, obs_tstop))
tlive = [np.sum(obs_tstop[obs_tstop<t] - obs_tstart[obs_tstop<t]) for t in obs_tall]
```

```python
matplotlib.rcParams.update({'font.size': 13})
times = np.linspace(tstart, tstop, 50)
dts = np.array([datetime.utcfromtimestamp(ts) for ts in times])
livetime = np.interp(times, obs_tall, tlive)

fig = plt.figure()
plt.plot(dts, livetime / (3600*24), color='C0', linewidth=1.5)
plt.xlabel('time')
plt.ylabel('livetime [d]', color='C0')
plt.gca().tick_params('y', colors='C0')
plt.title('livetime in Sept. 2018')
plt.grid()

xfmt = mdates.DateFormatter('%m-%d-%y')
plt.gca().xaxis.set_major_formatter(xfmt)
plt.xticks(rotation=25)

ax2 = plt.gca().twinx()
plt.ylabel('efficiency', color='C1')
plt.ylim([0,1])
plt.plot(dts, livetime / (times - times[0]), color='C1', linewidth=1.5)
plt.gca().tick_params('y', colors='C1')

plt.setp(list(plt.gca().spines.values()), linewidth=1.5)
plt.gca().xaxis.set_tick_params(width=1.5)
plt.gca().yaxis.set_tick_params(width=1.5)
plt.tight_layout()
plt.savefig('livetime_201809.png', dpi=200)
```

```python
np.interp?
```

```python

```
