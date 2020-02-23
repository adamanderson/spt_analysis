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

# Checks of Pol. Rotation Fitting
This note does some checks of the output of the pol. rotation angle fitting script.

```python
import numpy as np
import matplotlib.pyplot as plt
import pickle
from spt3g import core
from glob import glob
from matplotlib import gridspec
from scipy.stats import norm
```

```python
with open('testout.pkl', 'rb') as f:
    d = pickle.load(f)
```

## $\Delta \chi^2$ fit results

```python
d.keys()
```

```python
plt.plot(d['scan_angles'], d['scan_chi2'], 'o')
plot_angles = np.linspace(np.min(d['scan_angles']), np.max(d['scan_angles']), 101)
plt.plot(plot_angles, np.polyval(d['quadratic_approx_coeffs'], plot_angles))
np.array(d['angle_interval'])*180/np.pi
```

## Grid tests
Let's test the output from the grid.

```python
fnames = glob('/spt/user/adama/pol_angle_test1/*pkl')
```

```python
data = {}
for fn in fnames:
    with open(fn, 'rb') as f:
        key = os.path.splitext(os.path.basename(fn))[0]
        data[key] = pickle.load(f)
        data[key]['obsid'] = int(key.split('_')[4])
```

```python
plt.plot(d['scan_angles']*180/np.pi, d['scan_chi2'], 'o')
plot_angles = np.linspace(np.min(d['scan_angles']), np.max(d['scan_angles']), 101)
plt.plot(plot_angles*180/np.pi, np.polyval(d['quadratic_approx_coeffs'], plot_angles))
np.array(d['angle_interval'])*180/np.pi
```

```python
obsids = np.array([data[key]['obsid'] for key in data])
angle = np.array([data[key]['fit_angle'] for key in data])
err_lo = np.array([data[key]['angle_interval'][0] for key in data])
err_hi = np.array([data[key]['angle_interval'][1] for key in data])

plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(3,1)
gs.update(hspace=0.0)
plt.subplot2grid((3,1), (0,0), colspan=1, rowspan=2)
plt.errorbar(obsids, angle*180/np.pi, yerr=[(angle-err_lo)*180/np.pi,
                                            (err_hi-angle)*180/np.pi],
             marker='o', fmt=None)
plt.plot(obsids, angle*180/np.pi, marker='o', linewidth=0, color='C0')
plt.ylim([-9.9,10])
plt.grid()
plt.ylabel('polarization angle offset [deg]')
# plt.tight_layout()


plt.subplot2grid((3,1), (2,0), colspan=1, rowspan=2)
pulls = angle/((err_hi - err_lo)/2)
plt.plot(obsids, pulls, 'o')
plt.ylim([-4,4])
plt.ylabel('offset / $\sigma_{offset}$')
plt.xlabel('observation ID [sec]')
plt.grid()

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig('2018_ee_pol_angle.pdf')
```

```python
plt.figure()
_ = plt.hist(pulls[(pulls>-4) & (pulls<4)], bins=np.linspace(-4,4,21), normed=True, label='data')
mu, sigma = norm.fit(pulls[(pulls>-4) & (pulls<4)])
x = np.linspace(-4, 4, 100)
p = norm.pdf(x, 0, 1)
plt.plot(x, p, 'k', linewidth=2, label='unit gaussian')
plt.xlabel('best fit pol. angle per observation')
plt.ylabel('observations (normalized)')
plt.tight_layout()
plt.legend()
plt.savefig('pol_angle_normed_residuals.png', dpi=200)
print('mu = {}'.format(mu))
print('sigma = {}'.format(sigma))
```

## Debugging

```python
from calc_pol_angle import calc_chi2
```

```python
obs_fname = '/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/high_150GHz_left_maps/' + \
            '53615774/high_150GHz_left_maps_53615774.g3.gz'
coadd_fname = '/spt/user/ddutcher/coadds/20190917_full_150GHz.g3.gz'

# Read the data with a pipeline.
class MapExtractor(object):
    def __init__(self, frequency=None):
        self.observation_maps = core.G3Frame(core.G3FrameType.Map)
        self.delta_f_weights = None
        self.map_frequency = frequency
    def __call__(self, frame):
        if frame.type == core.G3FrameType.PipelineInfo and \
           "weight_high_freq" in frame.keys() and \
           "weight_low_freq" in frame.keys():
            self.delta_f_weights = (frame["weight_high_freq"] - frame["weight_low_freq"]) / core.G3Units.sec
        elif frame.type == core.G3FrameType.Map and \
             (self.map_frequency is None or self.map_frequency in frame['Id']):
            self.observation_maps['T'] = frame['T']
            self.observation_maps['Q'] = frame['Q']
            self.observation_maps['U'] = frame['U']
            self.observation_maps['Wpol'] = frame['Wpol']
            
# individual observation
pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=obs_fname)
map_extractor = MapExtractor(frequency='150')
pipe.Add(map_extractor)
pipe.Run()
obs_maps = map_extractor.observation_maps
delta_f_weights = map_extractor.delta_f_weights

# coadded observation
pipe_coadd = core.G3Pipeline()
pipe_coadd.Add(core.G3Reader, filename=coadd_fname)
map_extractor_coadd = MapExtractor()
pipe_coadd.Add(map_extractor_coadd)
pipe_coadd.Run()
coadd_maps = map_extractor_coadd.observation_maps

```

```python
angle_scan = np.linspace(-2, 2, 10)
chi2_scan = calc_chi2(angle_scan, obs_maps, coadd_maps,
                      np.array([-40, 40, -70, -40])*core.G3Units.deg)
```

```python
plt.plot(angle_scan, chi2_scan, 'o')
```

```python
plt.imshow(coadd_maps['Q'])
```

```python
m = obs_maps['Q']
```

```python
m.angle_to_xy(-5*core.G3Units.deg, -50*core.G3Units.deg)
```

```python
m.shape
```

```python
import os.path
os.path.basename('a/b.c')
```

```python

```
