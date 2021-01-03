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
from ksztools import kSZModel
import numpy as np
import matplotlib.pyplot as plt
```

```python
nbins = 1
Cshot_matrix = np.zeros((nbins, nbins))
ell_bin_edges = np.linspace(2000, 7000, nbins+1)
```

```python
model = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)

L_plot = np.logspace(0., np.log10(250), 21)
```

```python
CLKK_plot = np.array([model.CLKK(L, 0, 0, components=['reion', 'late']) for L in L_plot])
CLKK_plot_reion = np.array([model.CLKK(L, 0, 0, components=['reion']) for L in L_plot])
CLKK_plot_late = np.array([model.CLKK(L, 0, 0, components=['late']) for L in L_plot])
```

```python
NLKK_plot = np.array([model.NLKK(L, 0, 0) for L in L_plot])
```

```python
plt.loglog(L_plot, CLKK_plot * (L_plot**2) / (2*np.pi) / (model.Ktotal(0)**2),
           label='total')
plt.loglog(L_plot, CLKK_plot_reion * (L_plot**2) / (2*np.pi) / (model.Ktotal(0)**2),
           label='reionization')
plt.loglog(L_plot, CLKK_plot_late * (L_plot**2) / (2*np.pi) / (model.Ktotal(0)**2),
           label='late-time')
plt.loglog(L_plot, NLKK_plot * (L_plot**2) / (2*np.pi) / (model.Ktotal(0)**2),
           label='noise')
plt.axis([50, 300, 1e-4, 1e-2])
plt.legend()
plt.grid()
```

```python
z_plot = np.linspace(0.1, 12, 200) 
```

```python
dCLKKdz_plot       = np.array([model.dCLKKdz(z, 10, 0, 0, components=['reion', 'late'], interp=True) \
                               for z in z_plot])
dCLKKdz_plot_reion = np.array([model.dCLKKdz(z, 10, 0, 0, components=['reion'], interp=True) \
                               for z in z_plot])
dCLKKdz_plot_late  = np.array([model.dCLKKdz(z, 10, 0, 0, components=['late'], interp=True) \
                               for z in z_plot])
```

```python
plt.plot(z_plot, dCLKKdz_plot)
plt.plot(z_plot, dCLKKdz_plot_reion)
plt.plot(z_plot, dCLKKdz_plot_late)
# plt.plot(z_plot, dCLKKdz_plot_late + dCLKKdz_plot_reion)
```

```python
dCLKKdz_plot_L10       = np.array([model.dCLKKdz(z, 10, 0, 0, components=['reion', 'late'], interp=True) \
                               for z in z_plot])
dCLKKdz_plot_L30       = np.array([model.dCLKKdz(z, 30, 0, 0, components=['reion', 'late'], interp=True) \
                               for z in z_plot])
dCLKKdz_plot_L100       = np.array([model.dCLKKdz(z, 100, 0, 0, components=['reion', 'late'], interp=True) \
                               for z in z_plot])
```

```python
plt.plot(z_plot, dCLKKdz_plot_L10 / model.CLKK(10, 0, 0, components=['reion', 'late']))
plt.plot(z_plot, dCLKKdz_plot_L30 / model.CLKK(30, 0, 0, components=['reion', 'late']))
plt.plot(z_plot, dCLKKdz_plot_L100 / model.CLKK(100, 0, 0, components=['reion', 'late']))
plt.axis([0, 12, 0, 0.5])
```

```python
plt.plot(z_plot, dCLKKdz_plot_L10)
plt.plot(z_plot, dCLKKdz_plot_L30)
plt.plot(z_plot, dCLKKdz_plot_L100)
# plt.axis([0, 12, 0, 0.25])
```

```python

```
