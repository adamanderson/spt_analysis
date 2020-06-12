---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Nonlinearity Parameters
This is a calculation of the nonlinearity parameters for FDM proposed by Ed Young to calculate for the December 2019 readout assessment. See his note `HWP_constraints_readout.pdf` for more information (also located in this directory).

```python
import numpy as np
```

```python
def g_1(eta, P_elec, loopgain, w_mod, tau_0, C):
    return -1 * eta / (2*P_elec) * loopgain / (loopgain + 1) * \
           (loopgain + 1 + w_mod**2 * tau_0**2) / \
           ((loopgain + 1)**2 + w_mod**2 * tau_0**2) * C

def tau_1(eta, P_elec, loopgain, w_mod, tau_0, C):
    return tau_0 * eta / P_elec * (loopgain**2) / (loopgain + 1) / \
           ((loopgain + 1)**2 + w_mod**2 * tau_0**2) * C
```

```python
# setup of typical parameters for SPT-3G
params_3g = {'eta': 0.04e-12, # [W/K]
             'loopgain': 10,
             'w_mod': 4*2*np.pi*2, # [Hz]
             'C': 1} # [set by decree]

# CMB-S4 P electrical
P_elec_s4 = {30: 0.8 * 1e-12,
             40: 1.7 * 1e-12,
             85: 3.3 * 1e-12,
             145: 4.7 * 1e-12,
             95: 3.0 * 1e-12,
             155: 5.0 * 1e-12,
             220: 9.5 * 1e-12,
             270: 13.1 * 1e-12} # [W]

# CMB-S4 tau_0
tau_0_s4 =  {30: 0.135,
             40: 0.135,
             85: 0.047,
             145: 0.047,
             95: 0.042,
             155: 0.042,
             220: 0.024,
             270: 0.024} # [W]
```

```python
params = params_3g
for band in P_elec_s4:
    print('{:d} GHz'.format(band))
    params['P_elec'] = P_elec_s4[band]
    params['tau_0'] = tau_0_s4[band]
    print('g_1 [1/K] = {:.5f}'.format(g_1(**params)))
    print('tau_1 [us/K] = {:.1f}'.format(tau_1(**params) * 1e6))
    print()
```

```python
band
```

```python

```
