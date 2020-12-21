---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from ksztools import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd
import pickle
```

```python
from matplotlib.patches import Ellipse
from scipy.stats import chi2, multivariate_normal

def cov_ellipse(center, fisher_matrix, color, label):
    # compute contour parameters
    CLfactor = np.sqrt(chi2.ppf(0.684, 2, loc=0, scale=1))
    covariance = inv(fisher_matrix)
    ellipse_a = np.sqrt( (covariance[0,0] + covariance[1,1])/2 + \
                         np.sqrt((covariance[0,0] - covariance[1,1])**2 / 4 + covariance[0,1]**2) ) * CLfactor
    ellipse_b = np.sqrt( (covariance[0,0] + covariance[1,1])/2 - \
                         np.sqrt((covariance[0,0] - covariance[1,1])**2 / 4 + covariance[0,1]**2) ) * CLfactor
    ellipse_rot = (180 / np.pi) * 0.5 * np.arctan2(2*covariance[0,1], (covariance[0,0] - covariance[1,1]))

    contour = Ellipse(xy=center, width=2*ellipse_a, height=2*ellipse_b, angle=ellipse_rot,
                      facecolor='none', linewidth=4, edgecolor=color, label=label)
    
    return contour

def parameter_errs(fisher_matrix):
    covariance = inv(fisher_matrix)
    errs = [np.sqrt(covariance[ivar, ivar]) for ivar in np.arange(covariance.shape[0])]
    return errs
```

## 1-Bin Forecast

```python
# fiducial forecast values
planck_params = {'tau': 0.06}
planck_errs = {'tau': 0.0073}
ell_bin_edges = np.linspace(2000,7000,2) #[1000, 5000, 7000]
nbins = len(ell_bin_edges)-1
Cshot_matrix = np.zeros((nbins, nbins))

x0 = {'tau': 0.053, 'delta_z_re': 1.2, 'A_late': 1, 'alpha_late': 0,
      'shot_noise': Cshot_matrix}
dx = {'tau': 0.0002, 'delta_z_re': 0.05, 'A_late': 0.05, 'alpha_late': 0.05,
      'shot_noise': np.ones((nbins, nbins))}
prior = {'tau': 0.0073, 'delta_z_re': None, 'A_late': None, 'alpha_late': None,
         'shot_noise': None}
```

```python
# General forecast parameters
fsky = 0.7

model_4pt_Nbin = kSZModel(tau=x0['tau'],
                          delta_z_re=x0['delta_z_re'],
                          A_late=x0['A_late'],
                          alpha_late=x0['alpha_late'],
                          beam_fwhm_arcmin=1.0,
                          noise_uKarcmin=1.0,
                          ell_bin_edges=ell_bin_edges,
                          Cshot=Cshot_matrix)
```

```python
fisher_4pt = Fisher4Point(model_4pt_Nbin, fsky=0.7)
nlkk, clkk, deriv_clkk, fisher_matrix = fisher_4pt.calc_fisher(x0, dx, prior)
```

```python
print(pd.DataFrame(fisher_matrix))
```

```python
np.diag(np.sqrt(inv(fisher_matrix)))
```

```python
1./np.sqrt(fisher_matrix[0,0])
```

```python
1./np.sqrt(fisher_matrix[1,1])
```

```python
contour = cov_ellipse([0.06, x0['delta_z_re']], fisher_matrix, color='k', label='1 uK^@2')
# plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.add_artist(contour)
plt.axis([0.02, 0.1,
          x0['delta_z_re']-20, x0['delta_z_re']+20])
# plt.axis([0.05, 0.07, -1.5, 4])
plt.grid()
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.tight_layout()
```

## Long Runs

```python
tau_by_bins = []
deltaz_by_bins = []
nbins = [1, 4, 10, 20]
for nbin in [1, 4, 10, 20]:
    with open('long_runs/test4_nbins={}.pkl'.format(nbin), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{} BINS:'.format(nbin))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}\n'.format(errs_marg[1]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}\n'.format(errs_unmarg[1]))
    
    tau_by_bins.append(errs_marg[0])
    deltaz_by_bins.append(errs_marg[1])
```

```python
plt.subplot(2,1,1)
plt.semilogy(nbins, tau_by_bins, 'o')
plt.ylabel('$\sigma(\\tau)$')

plt.subplot(2,1,2)
plt.semilogy(nbins, deltaz_by_bins, 'o')
plt.ylabel('$\sigma(\Delta z)$')
plt.xlabel('$N_{bins}$')
plt.tight_layout()

plt.savefig('sigma_param_vs_Nbins.png', dpi=200)
```

## Experimental Configurations

```python
for expt in ['S4-wide-simple', 'SPT4-simple']:
    with open('long_runs/test5_{}_nbins=20.pkl'.format(expt), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{}:'.format(expt))
    print('sigma(tau) [marg.] = {}'.format(errs_marg[0]))
    print('sigma(delta z) [marg.] = {}\n'.format(errs_marg[1]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}\n'.format(errs_unmarg[1]))
```

```python
prior = {'tau':0.0073, 'delta_z_re':None}

with open('long_runs/test5_{}_nbins=20.pkl'.format('S4-wide-simple'), 'rb') as f:
    d_s4 = pickle.load(f)
with open('long_runs/test5_{}_nbins=20.pkl'.format('SPT4-simple'), 'rb') as f:
    d_spt4 = pickle.load(f)

contour_s4 = cov_ellipse([0.06, x0['delta_z_re']], d_s4['fisher_matrix'], 'r', 'S4')
contour_spt4 = cov_ellipse([0.06, x0['delta_z_re']], d_spt4['fisher_matrix'], 'g', 'SPT4')
# plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.add_patch(contour_s4)
ax.add_patch(contour_spt4)
ax.legend([contour_s4, contour_spt4], ['S4', 'SPT4'])
plt.axis([x0['tau']-0.03, x0['tau']+0.03,
          x0['delta_z_re']-1.5, x0['delta_z_re']+1.5])
# plt.axis([-1, 1, -100, 100])
# plt.axis([-0.2, 0.2, -20, 20])
plt.axis([0.035, 0.085, -2, 4])
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.tight_layout()
plt.savefig('s4_spt4_4pt_compare.png', dpi=200)
# plt.axis([-1, 1, -100, 100])
# plt.savefig('s4_spt4_4pt_compare_zoomout.png', dpi=200)
```

## Vary noise level

```python
tau_by_noise = []
deltaz_by_noise = []
noiselevels = [1, 3, 5, 7, 9]
for noiselevel in noiselevels:
    with open('long_runs/test6_{}uKarcmin_default_nbins=20.pkl'.format(noiselevel), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{} uK-arcmin:'.format(noiselevel))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}\n'.format(errs_marg[1]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}\n'.format(errs_unmarg[1]))
    
    tau_by_noise.append(errs_marg[0])
    deltaz_by_noise.append(errs_marg[1])
```

```python
plt.subplot(2,1,1)
plt.plot(noiselevels, tau_by_noise, 'o-')
plt.axis([1,6,0,0.012])

plt.subplot(2,1,2)
plt.plot(noiselevels, deltaz_by_noise, 'o-')
plt.axis([1,6,0,3.5])
```

```python

```
