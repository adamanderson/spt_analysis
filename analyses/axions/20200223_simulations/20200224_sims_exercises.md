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

# Some Notes on Simulations
These are some general notes on the SPT-3G simulations pipeline. It turns out that I am not very familiar with it, and this notebook contains some studies to familiarize myself.

```python
from healpy import fitsfunc, visufunc, pixelfunc
import numpy as np
import matplotlib.pyplot as plt
```

## Generating simulated skies
The script `make_3g_sims.py` generates simulated skies as fits files with different categories of sources: underlying cmb, atmospheric and instrumental noise, foregrounds, and total.

```python
ls sim_test_noise
```

```python
sim_cmb_1 = fitsfunc.read_map('sim_test_noise/cmb/cmb_alms_0001.fits')
```

```python
sim_total_1 = fitsfunc.read_map('sim_test_noise/total/total_150ghz_map_3g_0001.fits')
sim_cmb_1 = fitsfunc.read_map('sim_test_noise/cmb/cmb_alms_0001.fits')
sim_atm_noise_1 = fitsfunc.read_map('sim_test_noise/noise/atm_150ghz_alms_0001.fits')
sim_inst_noise_1 = fitsfunc.read_map('sim_test_noise/noise/inst_150ghz_alms_0001.fits')
```

```python
visufunc.mollview(sim_total_1, notext=True, cbar=False, title='')
visufunc.graticule(20)
# plt.axis([-0.5, 0.5, -0.75, 0.25])
```

```python
from astropy.io import fits
```

```python
hdu = fits.open('sim_test_noise/noise/atm_150ghz_alms_0001.fits')
```

```python
hdu.info()
```

```python
hdu_total = fits.open('sim_test_noise/total/total_150ghz_map_3g_0001.fits')
```

```python
hdu_total.info()
```

```python
hh = hdu_total[1]
hh.header
```

```python

```

```python

```
