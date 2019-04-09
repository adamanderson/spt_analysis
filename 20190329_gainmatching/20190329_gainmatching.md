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

# Fourier-space gain-matching
We would ideally gain-match using a continuously-running optical signal which would monitor the instantaneous gain of the detectors in each polarization pair. In the limit that we are not doing this, we need to gain-match to another optical signal present in each scan of the telescope. On a scan-by-scan basis, the power spectrum of a bolometer timestream consists of a.) a relatively flat noise floor dominated by photon noise with a non-negligible readout noise component, and b.) a 1/f component that is dominated by optical atmospheric fluctuations with possible contributions from exotic readout noise terms. This situation leaves us with only one option: we must gain-match on the 1/f noise, implicitly assuming that it is due to optical signals.

Let $d_x(t)$ and $d_y(t)$ be the two detector timestreams in a polarization pair. The gain-matched pair-differenced timestream is given by:
$$
d_-(t) = d_x(t) - Ad_y(t),
$$
where $A$ is a free gain-matching parameter. We want to choose an $A$ that provides "optimal" gain-matching in some sense. Ultimately, the entire purpose of gain-matching is to reduce 1/f noise, so it is natural to choose $A$ such that the noise in some low-frequency interval is minimized. In other words, we want to find
$$
\begin{align}
\hat{A} &= \underset{A}{\arg\min} \sum_{i \in F} \left| \tilde{d}_-(f_i) \right|^2 \\
&= \underset{A}{\arg\min} \sum_{i \in F} \left| \tilde{d}_x(f_i) - A \tilde{d}_y(f_i) \right|^2
\end{align}
$$

To find $\hat{A}$, we simply solve for the first-order conditions:
$$
\begin{align}
0 &= \frac{d}{dA}  \sum_{i \in F} \left(\tilde{d}^*_x(f_i) - A \tilde{d}^*_y(f_i) \right) \left(\tilde{d}_x(f_i) - A \tilde{d}_y(f_i) \right) |_{A = \hat{A}}\\
0 &=\sum_{i \in F} 2 \hat{A} \tilde{d}^*_y(f_i)\tilde{d}_y(f_i) - \tilde{d}^*_x(f_i)\tilde{d}_y(f_i) - \tilde{d}^*_y(f_i)\tilde{d}_x(f_i)\\
\hat{A} &= \frac{\sum_{i \in F} \left(\tilde{d}^*_x(f_i)\tilde{d}_y(f_i) + \tilde{d}^*_y(f_i)\tilde{d}_x(f_i) \right)}{\sum_{i \in F} \tilde{d}^*_y(f_i)\tilde{d}_y(f_i)}
\end{align}
$$

Let's implement this in a noise stare and then perform some Monte Carlo simulations to check for biases and optimality.

```python
from spt3g import core, dfmux, calibration
import numpy as np
import matplotlib.pyplot as plt
import os.path
import adama_utils
from importlib import reload
from scipy.signal import welch

%matplotlib inline
```

```python
from spt3g.calibration.template_groups import get_template_groups

d = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/68609192/offline_calibration.g3')]
bps = d[0]['BolometerProperties']
groups = get_template_groups(bps, per_pixel=True, per_wafer=True, include_keys=True)
good_pairs = []
for group, pair in groups.items():
    if len(pair) == 2:
        good_pairs.append(pair)
```

## Identify good polarization pairs
Let's write a pipeline segment to do some flagging and identify good polarization pairs for gain-matching analysis.

```python

pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=['/spt/data/bolodata/fullrate/noise/68609192/offline_calibration.g3',
                                  '/spt/data/bolodata/fullrate/noise/68609192/0000.g3'])
pipe.Add(core.FlagIncompletePixelPairs)
pipe.Add(core.)
```

## Quick test of method with a noise stare

```python
good_pairs
```

```python
reload(adama_utils)
```

```python
# get the timestreams
ts_data = adama_utils.get_raw_timestreams(['2019.038', '2019.1fv'], 70005920, file_name='0000.g3', scan_num=[0],
                              plot=False, data=True, cut_turnarounds=False,
                              psd=False, units=core.G3TimestreamUnits.Tcmb)
```

```python
ts_data
```

```python
ff, psd = welch(ts_data['2019.038'], fs=152.5, nperseg=1024)
plt.loglog(ff, psd)
ff, psd = welch(ts_data['2019.1fv'], fs=152.5, nperseg=1024)
plt.loglog(ff, psd)
```

```python
tsx = ts_data['2019.038'] - np.mean(ts_data['2019.038'])
tsy = ts_data['2019.1fv'] - np.mean(ts_data['2019.1fv'])
```

```python
fftx = np.fft.fft(tsx)
ffty = np.fft.fft(tsy)
Ahat = np.sum(np.conj(fftx)*ffty + np.conj(ffty)*fftx) / (2.*np.sum(np.abs(ffty)**2.))
```

```python
ff, psd = welch(tsx, fs=152.5, nperseg=1024*4)
plt.loglog(ff, psd)
ff, psd = welch(tsy, fs=152.5, nperseg=1024*4)
plt.loglog(ff, psd)

ff, psd = welch(tsx - tsy, fs=152.5, nperseg=1024*4)
plt.loglog(ff, psd)
```

```python
plt.plot(tsx)
plt.plot(tsy)
plt.plot(tsx - Ahat*tsy)
plt.xlim([0,300])
```

```python

```
