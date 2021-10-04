---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import ksztools
import numpy as np
import matplotlib.pyplot as plt
```

```python
nbins = 1
Cshot_matrix = np.zeros((nbins, nbins))
ell_bin_edges = np.linspace(2000, 7000, nbins+1)
```

```python
model = ksztools.kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)
```

```python
ksztools.Fisher4Point
```

```python

```
