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

# Check New rfracs
So... I thought that we should actually check that the new rfracs were ever implemented, after the fiasco with the stage temperature fluctuations, which we attributed to the lower values of rfrac.

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path

%matplotlib inline
```

```python
basepath = '/big_scratch/pydfmux_output'
tuninglist = ['20190320/20190320_092816_drop_bolos_cycle_tune_0121',
              '20190320/20190320_211813_drop_bolos_cycle_tune_0122',
              '20190321/20190321_181749_drop_bolos_cycle_tune_0123',
              '20190322/20190322_160221_drop_bolos_cycle_tune_0125',
              '20190323/20190323_045720_drop_bolos_cycle_tune_0126',
              '20190323/20190323_131316_drop_bolos_cycle_tune_0127',
              '20190323/20190324_001416_drop_bolos_cycle_tune_0128',
              '20190324/20190324_193449_drop_bolos_cycle_tune_0129',
              '20190325/20190325_145251_drop_bolos_cycle_tune_0130',
              '20190326/20190326_094224_drop_bolos_cycle_tune_0131']
for dirname in tuninglist:
    fname = os.path.join(basepath, dirname, 'data/TOTAL_DATA.pkl')
    with open(fname, 'rb') as f:
        d = pickle.load(f)
        
    rfracs = []
    for mod in d.keys():
        if 'subtargets' in d[mod]:
            for chan in d[mod]['subtargets']:
                rfracs.append(d[mod]['subtargets'][chan]['achieved_rfrac'])
              
    plt.figure()
    plt.hist(rfracs, bins=np.linspace(0,1,101))
    plt.title(dirname.split('/')[1])
```

```python

```
