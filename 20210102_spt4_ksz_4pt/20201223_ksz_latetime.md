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

## Checking the late-time kSZ power spectrum

```python
nbins = 1
Cshot_matrix = np.zeros((nbins, nbins))
ell_bin_edges = np.linspace(2000, 7000, nbins+1)
```

```python
model1 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)

model2 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1.2, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)

model3 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1.0, alpha_late=1,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)
```

```python
ell_plot = np.linspace(1000,7000)

plt.plot(ell_plot, model1.DlkSZ_Shaw(ell_plot))
plt.plot(ell_plot, model2.DlkSZ_Shaw(ell_plot))
```

```python
z_plot = np.linspace(0.1,10)

plt.plot(z_plot, model1.dDlkSZ3000dz_Shaw(z_plot))
plt.plot(z_plot, model2.dDlkSZ3000dz_Shaw(z_plot))
```

```python
z_plot = np.linspace(0.1,10)

plt.subplot(2,1,1)
plt.plot(z_plot, model1.dCkSZdz_late(3000, z_plot) * (3000*3001 / (2*np.pi)),
         label='$A_{late} = 1$, $\\alpha_{late} = 0$')
plt.plot(z_plot, model2.dCkSZdz_late(3000, z_plot) * (3000*3001 / (2*np.pi)),
         '--', label='$A_{late} = 1.2$, $\\alpha_{late} = 0$')
plt.plot(z_plot, model3.dCkSZdz_late(3000, z_plot) * (3000*3001 / (2*np.pi)),
         '--', label='$A_{late} = 1$, $\\alpha_{late} = 1$ ($\ell = 3000$)')
plt.plot(z_plot, model3.dCkSZdz_late(4000, z_plot) * (4000*4001 / (2*np.pi)),
         '--', label='$A_{late} = 1$, $\\alpha_{late} = 1$ ($\ell = 4000$)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(ell_plot, model1.dCkSZdz_late(ell_plot, 1) * (ell_plot*(ell_plot+1) / (2*np.pi)),
         label='$A_{late} = 1$')
plt.plot(ell_plot, model2.dCkSZdz_late(ell_plot, 1) * (ell_plot*(ell_plot+1) / (2*np.pi)),
         label='$A_{late} = 1.2$')
plt.plot(ell_plot, model3.dCkSZdz_late(ell_plot, 1) * (ell_plot*(ell_plot+1) / (2*np.pi)),
         label='$A_{late} = 1.2$')
```

```python
np.sum(model1.dDlkSZ3000dz_Shaw(z_plot)) * (z_plot[1] - z_plot[0])
```

```python
model1.DlkSZ_Shaw(3000)
```

## Checking the patchy kSZ power spectrum

```python
model1 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)

model2 = kSZModel(tau=0.07, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)

model3 = kSZModel(tau=0.06, delta_z_re=0.5*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1,
                 noise_uKarcmin=2,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)
```

```python
ell_plot = np.linspace(1000,7000)

plt.plot(ell_plot, model1.DlkSZ_reion_flat(ell_plot),
         label='$\\tau = 0.06$, $\sigma_z = 1.0$')
plt.plot(ell_plot, model2.DlkSZ_reion_flat(ell_plot),
         label='$\\tau = 0.07$, $\sigma_z = 1.0$')
plt.plot(ell_plot, model3.DlkSZ_reion_flat(ell_plot),
         label='$\\tau = 0.06$, $\sigma_z = 0.5$')
```

```python
plt.semilogy(ell_plot, model1.Cl_total(ell_plot) * (ell_plot*(ell_plot+1)) / (2*np.pi))
```

## Scratch

```python
# construct interpolated noise curves
s4_ilc_noise = np.loadtxt('ilc_residuals_s4.csv', delimiter=',')
spt4_ilc_noise = np.load('cmb_ilc_residuals_90-150-220-225-286-345_reducedradiopower4.0.npy').item()

s4_ilc_interp = interp1d(s4_ilc_noise[:,0],
                         s4_ilc_noise[:,1])
ell_plot = np.linspace(1,10000,10000)
spt4_ilc_interp = interp1d(ell_plot,
                           ell_plot*(ell_plot+1)*spt4_ilc_noise['cl_ilc_residual'] / (2*np.pi))

ell_interp = np.linspace(500, 7000)
noise_coeffs_s4 = np.polyfit(ell_interp, s4_ilc_interp(ell_interp), deg=5)
noise_coeffs_spt4 = np.polyfit(ell_interp, spt4_ilc_interp(ell_interp), deg=5)

noise_curve_s4 = np.poly1d(noise_coeffs_s4)
noise_curve_spt4 = np.poly1d(noise_coeffs_spt4)
```

```python
model1 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1.4,
                 noise_uKarcmin=0.01,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=noise_curve_s4)
```

```python
model2 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1.0,
                 noise_uKarcmin=0.01,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=noise_curve_spt4)
```

```python
model3 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1.0,
                 noise_uKarcmin=4,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)
```

```python
model4 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1.0,
                 noise_uKarcmin=5,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)
```

```python
model5 = kSZModel(tau=0.06, delta_z_re=1.0*np.sqrt(8.*np.log(2.)), A_late=1, alpha_late=0,
                 beam_fwhm_arcmin=1.0,
                 noise_uKarcmin=7,
                 ell_bin_edges=ell_bin_edges,
                 Cshot=Cshot_matrix,
                 noise_curve_interp=None)
```

```python
plt.semilogy(ell_plot, model1.Cl_total(ell_plot) * (ell_plot*(ell_plot+1)) / (2*np.pi),
            label='S4')
plt.semilogy(ell_plot, model2.Cl_total(ell_plot) * (ell_plot*(ell_plot+1)) / (2*np.pi),
            label='SPT4')
plt.semilogy(ell_plot, model3.Cl_total(ell_plot) * (ell_plot*(ell_plot+1)) / (2*np.pi), '--',
            label='4 uK-arcmin')
plt.semilogy(ell_plot, model4.Cl_total(ell_plot) * (ell_plot*(ell_plot+1)) / (2*np.pi), '--',
            label='5 uK-arcmin')
plt.semilogy(ell_plot, model5.Cl_total(ell_plot) * (ell_plot*(ell_plot+1)) / (2*np.pi), '--',
            label='7 uK-arcmin')
plt.grid()
plt.axis([0,7000,10,2000])
plt.legend()
```

```python

```
