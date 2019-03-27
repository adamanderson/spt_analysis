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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
```

```python
with open('/big_scratch/pydfmux_output/20181225/20181225_073517_measure_noise/'
          'data/9999_BOLOS_INFO_AND_MAPPING.pkl', 'rb') as f:
    d = pickle.load(f, encoding='latin1')
    
with open('/home/adama/20171111/20171111_162308_measure_noise_baseline_noise_testing3/'
          'data/8138_BOLOS_INFO_AND_MAPPING.pkl', 'rb') as f:
    dold = pickle.load(f, encoding='latin1')
```

```python
new_noise = np.array([d[bolo]['noise']['i_phase']['median_noise'] for bolo in d])
old_noise = np.array([dold[bolo]['noise']['noise_i']['median_noise'] for bolo in dold])
```

```python
matplotlib.rcParams.update({'font.size': 13})

_ = plt.hist(old_noise, np.linspace(0,80,41),
             histtype='stepfilled', alpha=0.5, color='C1')
_ = plt.hist(old_noise, np.linspace(0,80,41),
             histtype='step', linewidth=2, color='C1',
             label='before')

_ = plt.hist(new_noise, np.linspace(0,80,41),
             histtype='stepfilled', alpha=0.5, color='C0')
_ = plt.hist(new_noise, np.linspace(0,80,41),
             histtype='step', linewidth=2, color='C0',
             label='after')

plt.plot([13, 13], [0, 3000], 'k--')
plt.plot([30, 30], [0, 3000], '--', color='C2')
plt.axis([0, 80, 0, 3000])
plt.legend(frameon=False)
plt.xlabel('current noise [pA/rtHz]')
plt.ylabel('bolometers')
plt.setp(list(plt.gca().spines.values()), linewidth=1.5)
plt.gca().xaxis.set_tick_params(width=1.5)
plt.gca().yaxis.set_tick_params(width=1.5)
plt.tight_layout()
plt.savefig('NEI_before_vs_after.pdf')
```

```python
dold['005/11/1/4/2']['noise']['noise_i']['median_noise']
```

```python

```
