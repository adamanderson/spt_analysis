---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3 (v3)
    language: python
    name: python3-v3
---

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np
```

This is a super quick check that the axion limit does not depend on the choice of upper limit in the uniform prior. I ran the limit calculation with prior range of [0,0.5], [0,1.0] and [0,2.0] degrees, and compare the limits below. They differ at the 0.01% level, which is good enough for our purposes. The command for regenerating the limits was e.g.:

# ```
python fit_oscillation.py data --amp-prior-max 0.01745 --outfile data_results_amppriormax_1p0deg.pkl --verbose
# ```

```python
limit_data = {}
with open('data_results.pkl', 'rb') as f:
    limit_data[0.5] = pickle.load(f)
with open('data_results_amppriormax_1p0deg.pkl', 'rb') as f:
    limit_data[1.0] = pickle.load(f)
with open('data_results_amppriormax_2p0deg.pkl', 'rb') as f:
    limit_data[2.0] = pickle.load(f)
```

```python
limits = {}

plt.figure(1, figsize=(10,6))
for ampmax in limit_data:
    freqs = np.array(list(limit_data[ampmax]['A_upperlimit'].keys()))
    limits[ampmax] = np.array(list(limit_data[ampmax]['A_upperlimit'].values()))
    
    plt.plot(freqs, limits[ampmax], label='prior max = {:.2f}'.format(ampmax))
plt.legend()
plt.tight_layout()
plt.xlabel('frequency [d^-1]')
plt.ylabel('amplitude [rad]')
plt.savefig('limit_prior_check.png', dpi=200)

plt.figure(2, figsize=(10,6))
plt.plot(freqs, limits[2.0] - limits[1.0], label='limit(prior max = 2.0) - limit(prior max = 1.0)'.format(ampmax))
plt.plot(freqs, limits[1.0] - limits[0.5], label='limit(prior max = 1.0) - limit(prior max = 0.5)'.format(ampmax))
plt.legend()
plt.xlabel('frequency [d^-1]')
plt.ylabel('amplitude [rad]')
plt.tight_layout()
plt.savefig('limit_prior_check_diffs.png', dpi=200)
```

```python
plt.figure(figsize=(10,6))
for ampmax in limit_data:
    freqs = np.array(list(limit_data[ampmax]['A_upperlimit'].keys()))
    limits = np.array(list(limit_data[ampmax]['A_upperlimit'].values()))
    
    plt.plot(freqs, limits, label='prior max = {:.2f}'.format(ampmax))
plt.legend()
plt.xlim([0.5, 0.6])
plt.tight_layout()
```

```python

```
