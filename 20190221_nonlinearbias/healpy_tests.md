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
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
m = np.arange(hp.nside2npix(32))
hp.mollview(m, title="Mollview image RING")
hp.mollview(m, nest=True, title="Mollview image NESTED")
```

```python
test_map = hp.read_map('lensed_cmb_lmax7000_nside8192_interp0.3_method1_pol_1_sim_65_lensed_map.fits')
```

```python
hp.mollview(test_map, title="Mollview image RING", fig=1)
hp.graticule()
plt.savefig('test_map.png', dpi=800)
```

```python
len(test_map)
```

```python
hp.mollview?
```

```python
hp.graticule?
```

```python

```
