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
from ksztools import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from numpy.linalg import inv
import pandas as pd
import pickle
from matplotlib.patches import Rectangle
from glob import glob
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
                      facecolor='none', linewidth=2, edgecolor=color, label=label)
    
    return contour

def parameter_errs(fisher_matrix):
    covariance = inv(fisher_matrix)
    errs = [np.sqrt(covariance[ivar, ivar]) for ivar in np.arange(covariance.shape[0])]
    return errs
```

## Vary number of bins
These outputs should be compared with Table I of 1803.07036.

```python
plt.figure(1)
ax = plt.subplot(1,1,1)

tau_by_nbins = []
deltaz_by_nbins = []
nbins_list = [1, 2, 4, 8, 12, 16, 20]
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
for nbins, color in zip(nbins_list, color_list):
    with open('test4v3_1uKarcmin_fsky0p7_1arcmin_default_nbins={}.pkl'.format(nbins), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{} Bin:'.format(nbins))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}\n'.format(errs_marg[1]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}\n'.format(errs_unmarg[1]))
    
    tau_by_nbins.append(errs_marg[0])
    deltaz_by_nbins.append(errs_marg[1])
    
    contour = cov_ellipse([0.06, 1.2], d['fisher_matrix'], color, '{} bins'.format(nbins))
    ax.add_patch(contour)
    
plt.axis([0.035, 0.085, -1.5, 4])
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.tight_layout()
plt.legend()
plt.title('$\Delta_T$ = 1$\mu$K-arcmin, $f_{sky} = 0.7$, $\\theta_{FHWM} = 1$arcmin')
plt.savefig('reion_params_vs_nbins_contours.png', dpi=200)


plt.figure(2, figsize=(5,6))
plt.subplot(2,1,1)
plt.semilogy(nbins_list, tau_by_nbins, 'o-')
plt.axis([0, 21, 0.001, 0.04])
plt.grid()
plt.ylabel('$\sigma(\\tau)$')
plt.title('$\Delta_T$ = 1$\mu$K-arcmin, $f_{sky} = 0.7$, $\\theta_{FHWM} = 1$arcmin')

plt.subplot(2,1,2)
plt.semilogy(nbins_list, deltaz_by_nbins, 'o-')
plt.axis([0, 21, 0.1, 3.5])
plt.grid()
plt.xlabel('# bins')
plt.ylabel('$\sigma(\Delta z)$')
plt.tight_layout()

plt.savefig('reion_params_vs_nbins.png', dpi=200)
```

## Vary noise level

```python
tau_by_noise = []
deltaz_by_noise = []
noise_list = [1, 2, 3, 4, 5, 6]
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink']
for noise, color in zip(noise_list, color_list):
    with open('test4v3_{}uKarcmin_fsky0p7_1arcmin_default_nbins=20.pkl'.format(noise), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
#     errs_marg = np.diag(np.sqrt(inv(np.delete(np.delete(d['fisher_matrix'], [2,3], axis=1), [2,3], axis=0))))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{} uK-arcmin:'.format(noise))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}'.format(errs_marg[1]))
    print('sigma(A_late) = {}'.format(errs_marg[2]))
    print('sigma(alpha_late) = {}\n'.format(errs_marg[3]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}'.format(errs_unmarg[1]))
    print('sigma(A_late) [unmarg.] = {}'.format(errs_unmarg[2]))
    print('sigma(alpha_late) [unmarg.] = {}\n'.format(errs_unmarg[3]))
    
    tau_by_noise.append(errs_marg[0])
    deltaz_by_noise.append(errs_marg[1])

plt.figure(1, figsize=(5,6))
plt.subplot(2,1,1)
plt.plot(noise_list, tau_by_noise, 'o-')
plt.axis([1, 6, 0, 0.012])
plt.grid()
plt.ylabel('$\sigma(\\tau)$')
plt.title('$\Delta_T$ = 1$\mu$K-arcmin, $f_{sky} = 0.7$, $\\theta_{FHWM} = 1$arcmin')

plt.subplot(2,1,2)
plt.plot(noise_list, deltaz_by_noise, 'o-')
plt.axis([1, 6, 0, 3.5])
plt.grid()
plt.xlabel('map noise [$\mu$K-arcmin]')
plt.ylabel('$\sigma(\Delta z)$')
plt.tight_layout()

plt.savefig('reion_params_vs_noise.png', dpi=200)
```

## Comparison of SPT-4 and S4 ILC noise curves

```python
s4_ilc_noise = np.loadtxt('ilc_residuals_s4.csv', delimiter=',')
spt4_ilc_noise = np.load('cmb_ilc_residuals_90-150-220-225-286-345_reducedradiopower4.0.npy').item()
```

```python
s4_ilc_interp = interp1d(s4_ilc_noise[:,0],
                         s4_ilc_noise[:,1])
ell_plot = np.linspace(1,10000,10000)
spt4_ilc_interp = interp1d(ell_plot,
                           ell_plot*(ell_plot+1)*spt4_ilc_noise['cl_ilc_residual'] / (2*np.pi))

plt.subplot(2,1,1)
plt.semilogy(s4_ilc_noise[:,0],
             s4_ilc_noise[:,1])
plt.semilogy(ell_plot,
             ell_plot*(ell_plot+1)*spt4_ilc_noise['cl_ilc_residual'] / (2*np.pi))
plt.semilogy()
plt.xlim([500, 7000])
plt.title('minimum-variance ILC noise curves')
plt.grid()
plt.ylabel('$D_\ell$ [$\mu$K$^2$]')

plt.subplot(2,1,2)
ell_plot = np.linspace(500, 7000)
plt.plot(ell_plot, spt4_ilc_interp(ell_plot) / s4_ilc_interp(ell_plot))
plt.ylabel('ratio of noise (SPT-4 / S4)')
plt.xlabel('multipole $\ell$')
plt.grid()

plt.tight_layout()
```

Next let's compute the effective map noise level at ell=5000 for each case.

```python
np.sqrt(spt4_ilc_interp(5000) * 2*np.pi / (5000*5001)) * (10800 / np.pi)
```

```python
np.sqrt(s4_ilc_interp(5000) * 2*np.pi / (5000*5001)) * (10800 / np.pi)
```

```python
ell_plot = np.linspace(500,7000)
coeffs_s4 = np.polyfit(ell_plot, s4_ilc_interp(ell_plot), deg=5)
plt.semilogy(ell_plot, s4_ilc_interp(ell_plot))
plt.semilogy(ell_plot, np.polyval(coeffs_s4, ell_plot), '--')
plt.semilogy(ell_plot,
             ell_plot*(ell_plot+1)/(2*np.pi)*((5*np.pi/10800)**2) * \
             np.exp(ell_plot**2 * (1.4/np.sqrt(8*np.log(2)) * np.pi/10800)**2))
plt.grid()
```

```python
coeffs_spt4 = np.polyfit(ell_plot, spt4_ilc_interp(ell_plot), deg=9)
plt.semilogy(ell_plot, spt4_ilc_interp(ell_plot))
plt.semilogy(ell_plot, np.polyval(coeffs_spt4, ell_plot), '--')
plt.semilogy(ell_plot,
             ell_plot*(ell_plot+1)/(2*np.pi)*((3*np.pi/10800)**2) * \
             np.exp(ell_plot**2 * (1.2/np.sqrt(8*np.log(2)) * np.pi/10800)**2))
plt.grid()
```

## Plot all ILC noise curves

```python
for fname in glob('ilc_residuals/*/*npz'):
    dic_for_nl = np.load( fname, allow_pickle = 1, encoding = 'latin1')['arr_0'].item()
    res_ilc_tt = dic_for_nl['all_cl_ilc_residual']['TT']
    el = np.arange(len(res_ilc_tt))
    label = fname.split('/')[1]
    plt.semilogy(el, el*(el+1)/(2*np.pi) * res_ilc_tt,
                 label=label)
plt.xlim([100,5000])
plt.legend()
```

## SPT-4 and S4 ILC with real noise curves

```python
# No prior on A_late, alpha_late
ax = plt.subplot(1,1,1)
planck_tau = Rectangle((0.060-0.007, -100), 2*0.007, 200,
                       color='tab:red', alpha=0.3, linewidth=0,
                       label='Planck')
ax.add_patch(planck_tau)
plt.plot([-100,100], [1.2, 1.2], 'k--', linewidth=1)
plt.plot([0.06, 0.06], [-100, 100], 'k--', linewidth=1)

tau_by_noise = []
deltaz_by_noise = []
expt_list = ['SPT4-simple', 'S4-wide-simple']
expt_name_list = ['SPT-4', 'S4-wide']
color_list = ['tab:blue', 'tab:orange']
for expt, expt_name, color in zip(expt_list, expt_name_list, color_list):
    with open('test3v3-2_{}_nbins=20.pkl'.format(expt), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{}'.format(expt_name))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}'.format(errs_marg[1]))
    print('sigma(A_late) = {}'.format(errs_marg[2]))
    print('sigma(alpha_late) = {}\n'.format(errs_marg[3]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}'.format(errs_unmarg[1]))
    print('sigma(A_late) [unmarg.] = {}'.format(errs_unmarg[2]))
    print('sigma(alpha_late) [unmarg.] = {}\n'.format(errs_unmarg[3]))
    
    tau_by_noise.append(errs_marg[0])
    deltaz_by_noise.append(errs_marg[1])
    
    contour = cov_ellipse([0.06, 1.2], d['fisher_matrix'], color, '{}'.format(expt_name))
    ax.add_patch(contour)
    
    cov[expt] = inv(d['fisher_matrix'])[:4,:4]
    
plt.axis([0.035, 0.085, -1.5, 4])
# plt.axis([-0.1, 0.22, -12, 14])
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.title('No prior on $A_{late}$ and $\\alpha_{late}$')
plt.tight_layout()
plt.legend()

plt.savefig('reion_params_s4_spt4.png', dpi=200)

with open('s4_spt4_covariance.pkl', 'wb') as f:
    pickle.dump(cov, f)
```

```python
ax = plt.subplot(1,1,1)
planck_tau = Rectangle((0.060-0.007, -100), 2*0.007, 200,
                       color='tab:red', alpha=0.3, linewidth=0,
                       label='Planck')
ax.add_patch(planck_tau)
plt.plot([-100,100], [1.2, 1.2], 'k--', linewidth=1)
plt.plot([0.06, 0.06], [-100, 100], 'k--', linewidth=1)

tau_by_noise = []
deltaz_by_noise = []
expt_list = ['SPT4-simple', 'S4-wide-simple']
expt_name_list = ['SPT-4', 'S4-wide']
color_list = ['tab:blue', 'tab:orange']
cov = {}
for expt, expt_name, color in zip(expt_list, expt_name_list, color_list):
    with open('test3v3-2_{}_nbins=20.pkl'.format(expt), 'rb') as f:
        d = pickle.load(f)
    
    # 50% priors on A_late and alpha_late
    d['fisher_matrix'][2,2] += 1/(0.5**2)
    d['fisher_matrix'][3,3] += 1/(0.5**2)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{}'.format(expt_name))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}'.format(errs_marg[1]))
    print('sigma(A_late) = {}'.format(errs_marg[2]))
    print('sigma(alpha_late) = {}\n'.format(errs_marg[3]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}'.format(errs_unmarg[1]))
    print('sigma(A_late) [unmarg.] = {}'.format(errs_unmarg[2]))
    print('sigma(alpha_late) [unmarg.] = {}\n'.format(errs_unmarg[3]))
    
    tau_by_noise.append(errs_marg[0])
    deltaz_by_noise.append(errs_marg[1])
    
    contour = cov_ellipse([0.06, 1.2], d['fisher_matrix'], color, '{}'.format(expt_name))
    ax.add_patch(contour)
    
    cov[expt] = inv(d['fisher_matrix'])[:2,:2]

with open('s4_spt4_covariance.pkl', 'wb') as f:
    pickle.dump(cov, f)

plt.axis([0.035, 0.085, -1.5, 4])
# plt.axis([-0.1, 0.22, -12, 14])
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.title('50% prior on $A_{late}$ and $\\alpha_{late}$')
plt.tight_layout()
plt.legend()

plt.savefig('reion_params_s4_spt4_prior_latetimekSZ.png', dpi=200)
```

## All experiments

```python
# No prior on A_late, alpha_late
ax = plt.subplot(1,1,1)
planck_tau = Rectangle((0.060-0.007, -100), 2*0.007, 200,
                       color='tab:red', alpha=0.3, linewidth=0,
                       label='Planck')
ax.add_patch(planck_tau)
plt.plot([-100,100], [1.2, 1.2], 'k--', linewidth=1)
plt.plot([0.06, 0.06], [-100, 100], 'k--', linewidth=1)

tau_by_noise = []
deltaz_by_noise = []
expt_list = ['s4deepv3r025', 's4wide', 'sobaseline', 'spt4_C3', 'spt3g', 's4deepv3r025plusspt4HF']
expt_name_list = ['s4deepv3r025', 's4wide', 'sobaseline', 'spt4_C3', 'spt3g', 's4deepv3r025plusspt4HF']
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:purple']
cov = {}
for expt, expt_name, color in zip(expt_list, expt_name_list, color_list):
    with open('outputs/final_{}_nbins=20.pkl'.format(expt), 'rb') as f:
        d = pickle.load(f)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{}'.format(expt_name))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}'.format(errs_marg[1]))
    print('sigma(A_late) = {}'.format(errs_marg[2]))
    print('sigma(alpha_late) = {}\n'.format(errs_marg[3]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}'.format(errs_unmarg[1]))
    print('sigma(A_late) [unmarg.] = {}'.format(errs_unmarg[2]))
    print('sigma(alpha_late) [unmarg.] = {}\n'.format(errs_unmarg[3]))
    
    tau_by_noise.append(errs_marg[0])
    deltaz_by_noise.append(errs_marg[1])
    
    contour = cov_ellipse([0.06, 1.2], d['fisher_matrix'], color, '{}'.format(expt_name))
    ax.add_patch(contour)
    
    cov[expt] = inv(d['fisher_matrix'])[:4,:4]
    
plt.axis([0.035, 0.085, -1.5, 4])
# plt.axis([-0.1, 0.22, -12, 14])
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.title('No prior on $A_{late}$ and $\\alpha_{late}$')
plt.tight_layout()
plt.legend()

plt.savefig('reion_params_s4_spt4.png', dpi=200)

with open('ksz_4pt_covariances.pkl', 'wb') as f:
    pickle.dump(cov, f)
```

```python
cov
```

```python
# No prior on A_late, alpha_late
ax = plt.subplot(1,1,1)
planck_tau = Rectangle((0.060-0.007, -100), 2*0.007, 200,
                       color='tab:red', alpha=0.3, linewidth=0,
                       label='Planck')
ax.add_patch(planck_tau)
plt.plot([-100,100], [1.2, 1.2], 'k--', linewidth=1)
plt.plot([0.06, 0.06], [-100, 100], 'k--', linewidth=1)

tau_by_noise = []
deltaz_by_noise = []
expt_list = ['s4deepv3r025', 's4wide', 'sobaseline', 'spt4_C3', 'spt3g', 's4deepv3r025plusspt4HF']
expt_name_list = ['s4deepv3r025', 's4wide', 'sobaseline', 'spt4_C3', 'spt3g', 's4deepv3r025plusspt4HF']
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:purple']
for expt, expt_name, color in zip(expt_list, expt_name_list, color_list):
    with open('outputs/final_{}_nbins=20.pkl'.format(expt), 'rb') as f:
        d = pickle.load(f)
    
    # 50% priors on A_late and alpha_late
    d['fisher_matrix'][2,2] += 1/(0.5**2)
    d['fisher_matrix'][3,3] += 1/(0.5**2)
    
    errs_marg = np.diag(np.sqrt(inv(d['fisher_matrix'])))
    errs_unmarg = 1./np.diag(np.sqrt(d['fisher_matrix']))
    
    print('{}'.format(expt_name))
    print('sigma(tau) = {}'.format(errs_marg[0]))
    print('sigma(delta z) = {}'.format(errs_marg[1]))
    print('sigma(A_late) = {}'.format(errs_marg[2]))
    print('sigma(alpha_late) = {}\n'.format(errs_marg[3]))
    print('sigma(tau) [unmarg.] = {}'.format(errs_unmarg[0]))
    print('sigma(delta z) [unmarg.] = {}'.format(errs_unmarg[1]))
    print('sigma(A_late) [unmarg.] = {}'.format(errs_unmarg[2]))
    print('sigma(alpha_late) [unmarg.] = {}\n'.format(errs_unmarg[3]))
    
    tau_by_noise.append(errs_marg[0])
    deltaz_by_noise.append(errs_marg[1])
    
    contour = cov_ellipse([0.06, 1.2], d['fisher_matrix'], color, '{}'.format(expt_name))
    ax.add_patch(contour)
    
    cov[expt] = inv(d['fisher_matrix'])[:4,:4]
    
plt.axis([0.035, 0.085, -1.5, 4])
# plt.axis([-0.1, 0.22, -12, 14])
plt.xlabel('$\\tau$')
plt.ylabel('$\Delta z_{re}$')
plt.title('50% prior on $A_{late}$ and $\\alpha_{late}$')
plt.tight_layout()
plt.legend()

plt.savefig('reion_params_s4_spt4.png', dpi=200)
```

## Scratch

```python
with open('test_SPT4-simple_nbins=20.pkl', 'rb') as f:
    d1 = pickle.load(f)
```

```python
d1['fisher_matrix'][:4, :4]
```

```python
plt.plot(ell_plot, np.exp(-ell_plot**2 * (1.2/(8*np.log(2)) * np.pi/10800)**2))
plt.plot(ell_plot, np.exp(-ell_plot**2 * (1.4/(8*np.log(2)) * np.pi/10800)**2))
```

```python
10800/np.pi * np.sqrt(25/(5000*5001) * (2*np.pi))
```

```python
ell_plot2 = np.linspace(100,8000)
plt.semilogy(ell_plot2,
             ell_plot2*(ell_plot2+1)/(2*np.pi)*((5*np.pi/10800)**2) * \
             np.exp(ell_plot2**2 * (1.4/np.sqrt(8*np.log(2)) * np.pi/10800)**2))
# plt.semilogy(ell_plot2,
#              ell_plot2*(ell_plot2+1)/(2*np.pi)*((10*np.pi/10800)**2) * \
#              np.exp(ell_plot2**2 * (1.4/np.sqrt(8*np.log(2)) * np.pi/10800)**2))
plt.grid()
```

```python
np.exp(ell_plot2**2 * (1.4/np.sqrt(8*np.log(2)) * np.pi/10800)**2)
```

```python
with open('s4_spt4_covariance.pkl', 'rb') as f:
    cov = pickle.load(f)
```

```python
cov
```

```python
np.sqrt(4.77085060e-04)
```

```python

```
