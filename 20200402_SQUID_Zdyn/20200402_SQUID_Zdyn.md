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
from spt3g import core, calibration
import numpy as np
import matplotlib.pyplot as plt
import pydfmux
from glob import glob
```

```python
fname = '/spt/user/production/calibration/noise/87407220.g3'
d = list(core.G3File(fname))[0]

fname = '/spt/data/bolodata/fullrate/noise/87407220/offline_calibration.g3'
boloprops = list(core.G3File(fname))[0]["BolometerProperties"]

hwm_file = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm = pydfmux.load_session(open(hwm_file))['hardware_map']
```

```python
freq_dict = {b.name: b.lc_channel.frequency * 1e-6 for b in hwm.query(pydfmux.Bolometer)}
```

```python
fname = '/spt/user/production/calibration/noise/87407220.g3'
d = list(core.G3File(fname))[0]

low_zdyn_pstrings = {'w177': ['005/13/1/2/*'],
                     'w174': ['005/15/2/4/*'],
                     'w204': ['015/12/1/3/*'],
                     'w181': ['015/14/2/4/*'],
                     'w172': ['015/8/1/3/*', '015/8/2/3/*']}
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']
jwafer = 1
xranges = {90:[0, 1200], 150:[0,1000], 220:[0,3000]}
plt.figure(jwafer, figsize=(12,20))
# for wafer in wafer_list:
for wafer, pstrings in low_zdyn_pstrings.items():
    pstrings = low_zdyn_pstrings[wafer]
    for jband, band in enumerate([90, 150, 220]):
        plt.subplot(5,3,jband+1 + 3*(jwafer-1))
        net = np.array([d["NET_3.0Hz_to_5.0Hz"][bolo] \
                        for bolo in d["NET_3.0Hz_to_5.0Hz"].keys()
                        if boloprops[bolo].band / core.G3Units.GHz == band and \
                           boloprops[bolo].wafer_id == wafer]) / \
                (1e-6*core.G3Units.K * np.sqrt(core.G3Units.sec))
        net_freqs = np.array([freq_dict[bolo] \
                        for bolo in d["NET_3.0Hz_to_5.0Hz"].keys()
                        if boloprops[bolo].band / core.G3Units.GHz == band and \
                           boloprops[bolo].wafer_id == wafer])
        plt.plot(net_freqs, net, 'o', markersize=3)
#         _ = plt.hist(net, bins=np.linspace(xranges[band][0], xranges[band][1], 41), histtype='step')
        
        net_low_zdyn = []
        net_low_zdyn_freqs = []
        for pstring in pstrings:
            bolos = hwm.bolos_from_pstring(pstring).name
            net_low_zdyn.extend([d["NET_3.0Hz_to_5.0Hz"][bolo] \
                                 for bolo in bolos \
                                 if bolo in d["NET_3.0Hz_to_5.0Hz"].keys() and \
                                    boloprops[bolo].band / core.G3Units.GHz == band])
            net_low_zdyn_freqs.extend([freq_dict[bolo] \
                                 for bolo in bolos \
                                 if bolo in d["NET_3.0Hz_to_5.0Hz"].keys() and \
                                    boloprops[bolo].band / core.G3Units.GHz == band])
        net_low_zdyn = np.array(net_low_zdyn) / (1e-6*core.G3Units.K * np.sqrt(core.G3Units.sec))
#         _ = plt.hist(net_low_zdyn, bins=np.linspace(xranges[band][0], xranges[band][1], 41),
#                      histtype='step', color='k')
        plt.plot(net_low_zdyn_freqs, net_low_zdyn, 'o', markersize=3)
        plt.title('{}: {} GHz\n'\
                  'NET all: {:.0f} uK rtsec\n'\
                  'NET low Zdyn: {:.0f} uK rtsec\n'.format(wafer, band,
                                                           np.median(net),
                                                           np.median(net_low_zdyn)))
        plt.ylim(xranges[band])
            
    jwafer += 1
plt.tight_layout()
plt.savefig('net_by_freq.png', dpi=200)
```

```python
low_zdyn_pstrings = {'w177': ['005/13/1/2/*'],
                     'w174': ['005/15/2/4/*'],
                     'w204': ['015/12/1/3/*'],
                     'w181': ['015/14/2/4/*'],
                     'w172': ['015/8/1/3/*', '015/8/2/3/*']}
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']

fname_list = glob('/spt/user/production/calibration/noise/87*.g3')
print(fname_list)

jwafer = 1
xranges = {90:[0, 1200], 150:[0,1000], 220:[0,3000]}
plt.figure(jwafer, figsize=(12,40))

net = {}
net_low_zdyn = {}
for fname in fname_list:
    print(fname)
    d = list(core.G3File(fname))[0]
    
    for wafer in wafer_list:
        if wafer not in net:
            net[wafer] = {90:[], 150:[], 220:[]}
        if wafer not in net_low_zdyn:
            net_low_zdyn[wafer] = {90:[], 150:[], 220:[]}
            
        for jband, band in enumerate([90, 150, 220]):
            net[wafer][band].extend([d["NET_3.0Hz_to_5.0Hz"][bolo] \
                            for bolo in d["NET_3.0Hz_to_5.0Hz"].keys()
                            if boloprops[bolo].band / core.G3Units.GHz == band and \
                               boloprops[bolo].wafer_id == wafer])
            if wafer in low_zdyn_pstrings:
                for pstring in low_zdyn_pstrings[wafer]:
                    bolos = hwm.bolos_from_pstring(pstring).name
                    net_low_zdyn[wafer][band].extend([d["NET_3.0Hz_to_5.0Hz"][bolo] \
                                         for bolo in bolos \
                                         if bolo in d["NET_3.0Hz_to_5.0Hz"].keys() and \
                                            boloprops[bolo].band / core.G3Units.GHz == band])

        jwafer += 1
```

```python
plt.figure(jwafer, figsize=(12,20))
jwafer = 1

for wafer in wafer_list:
# for wafer, pstrings in low_zdyn_pstrings.items():
    for jband, band in enumerate([90, 150, 220]):
        plt.subplot(10,3,jband+1 + 3*(jwafer-1))
        net_plot = np.array(net[wafer][band]) / (1e-6*core.G3Units.K * np.sqrt(core.G3Units.sec))
        _ = plt.hist(net_plot[np.isfinite(net_plot)],
                     bins=np.linspace(xranges[band][0], xranges[band][1], 41),
                     histtype='step', normed=True)

        if len(net_low_zdyn[wafer][band])>0:
            net_low_zdyn_plot = np.array(net_low_zdyn[wafer][band]) / (1e-6*core.G3Units.K * np.sqrt(core.G3Units.sec))
            _ = plt.hist(net_low_zdyn_plot[np.isfinite(net_low_zdyn_plot)], \
                         bins=np.linspace(xranges[band][0], xranges[band][1], 41),
                         histtype='step', color='k', normed=True)
            plt.title('{}: {} GHz\n'\
                      'NET all: {:.0f} uK rtsec\n'\
                      'NET low Zdyn: {:.0f} uK rtsec\n'.format(wafer, band,
                                                               np.median(net_plot[np.isfinite(net_plot)]),
                                                               np.median(net_low_zdyn_plot[np.isfinite(net_low_zdyn_plot)])))
        else:
            plt.title('{}: {} GHz\n'\
                      'NET all: {:.0f} uK rtsec\n'.format(wafer, band,
                                                           np.median(net_plot[np.isfinite(net_plot)])))
            
    jwafer += 1
plt.tight_layout()
plt.savefig('net_all_normed.png', dpi=200)
```

```python
freq_dict
```

```python
b = bolos
```

```python
b.
```

```python

```
