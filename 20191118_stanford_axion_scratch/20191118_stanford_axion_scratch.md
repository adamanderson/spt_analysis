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

# Scratch Work and Notes on Axions from Stanford Visit

```python
from spt3g import core, mapmaker
from spt3g.mapmaker import remove_weight
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pol_angle import fit_pol_angle, chi2_pol_angle
```

Question: How many pixels are in one of Daniel's 2-arcmin pixel maps?

```python
map_fname = '/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/' + \
            'high_150GHz_left_maps/55845637/high_150GHz_left_maps_55845637.g3.gz'
d_single = list(core.G3File(map_fname))[5]
```

```python
arr_single_u = np.array(d_single['U'])
print(arr_single_u.shape)
print(len(arr_single_u[arr_single_u != 0]))
plt.imshow(arr_single_u)
```

Question: What does the covariance matrix for a single pixel look like in a single map.

```python
# load the coadd
fname_coadd = '/spt/user/ddutcher/coadds/20190917_full_90GHz.g3.gz'
d_coadd = list(core.G3File(fname_coadd))[0]
```

```python
# remove the weights
_, _, u_single_noweight = remove_weight(d_single['T'], d_single['Q'], d_single['U'], d_single['Wpol'])
_, _, u_coadd_noweight = remove_weight(d_coadd['T'], d_coadd['Q'], d_coadd['U'], d_coadd['Wpol'])
```

```python
plt.figure(figsize=(12,8))
plt.imshow(u_coadd_noweight, vmin=-0.05, vmax=0.05)
plt.colorbar()
plt.axis([1000, 1200, 800, 1000])
```

```python
plt.figure(figsize=(12,8))
plt.imshow(u_coadd_noweight, vmin=-1.0, vmax=1.0)
plt.colorbar()
```

```python
u_single_noweight[850, 1100]
```

```python
plt.figure(figsize=(12,8))
plt.imshow(u_single_noweight, vmin=-1.0, vmax=1.0)
plt.colorbar()
```

```python
u_residual = u_single_noweight - u_coadd_noweight

plt.figure(figsize=(12,8))
plt.imshow(u_residual, vmin=-1.0, vmax=1.0)
plt.colorbar()
```

```python
d_corr = pickle.load(open('correlation.pkl', 'rb'))
```

```python
plt.figure(figsize=(12,8))

plt.imshow(d_corr) #, vmin=-0.1, vmax=0.1)
plt.axis([1050,1150,800,900])
plt.colorbar()
```

```python
plt.figure(figsize=(12,8))

plt.imshow(d_corr) #, vmin=-0.1, vmax=0.1)
plt.axis([1090,1110,840,860])
plt.colorbar()
```

```python
plt.figure(figsize=(12,8))

plt.imshow(d_corr) #, vmin=-0.1, vmax=0.1)
# plt.axis([1090,1110,840,860])
plt.colorbar()
```

## Tests of Polarization Angle Fitting


### Covariance matrix set to identity


Check the map-space polarization angle fitting. Let's set the covariance matrix equal to the identity. This will result in nonsensical values of $\chi^2$, but it will be a useful sanity check.

```python
result = fit_pol_angle(d_single, d_coadd)
```

```python
result
```

```python
plt.imshow(d_coadd['Q'])
```

```python
alpha_plot = np.linspace(-0.5, 0.5, 100)

chi2_plot = np.zeros(len(alpha_plot))
for jalpha, alpha in enumerate(alpha_plot):
    chi2_plot[jalpha] = chi2_pol_angle(alpha, d_single, d_coadd, map_covariance='1')
    print(chi2_plot[jalpha])
    
# chi2_plot = [chi2_pol_angle(alpha, d_single, d_coadd, map_covariance='1') for alpha in alpha_plot]
```

```python
plt.plot(alpha_plot, chi2_plot)
```

### Covariance matrix set to rescaled identity


The result above is horribly discrepant from zero. This is surely due to the fact that the covariance matrix is just the identity. As a more realistic approximation, let's still assume the covariance matrix is diagonal, but weight by the average noise in a pixel. We can later extend this by using the weights map, but this refinement should only change the chi-square slightly because the pixels on-average have fairly similar weights.

To be precise, our covariance matrix in this test is given by
\begin{equation}
\Sigma =
\begin{pmatrix}
\textrm{Var}(Q) \times I_n & 0 \\
0 & \textrm{Var}(U) \times I_n
\end{pmatrix}.
\end{equation}

```python
qvals = np.hstack(np.array(d_single['Q']))
std_q = np.std(qvals[qvals!=0])
uvals = np.hstack(np.array(d_single['U']))
std_u = np.std(uvals[uvals!=0])

result = fit_pol_angle(d_single, d_coadd, 'std', std_q, std_u)
```

```python
print(result)
```

```python
qvals = np.hstack(np.array(d_single['Q']))
_ = plt.hist(qvals[qvals!=0],
         bins=np.linspace(-100,100,101))
# plt.gca().set_yscale('log')
```

```python
alpha_plot = np.linspace(-1.0, 1.0)
chi2_plot = [chi2_pol_angle(alpha, d_single, d_coadd, map_covariance='1') for alpha in alpha_plot]
```

```python
plt.plot(alpha_plot, chi2_plot)
```

```python

```
