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
from spt3g import core, calibration
import numpy as np
import scipy
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
dold2019 = list(core.G3File('/spt/user/production/calibration/boloproperties/60000000.g3'))[0]
dnew2019_rcw38 = list(core.G3File('/home/adama/SPT/spt3g_software/calibration/scripts/testout_rcw38_mean.g3'))[0]
dnew2019_mat5a = list(core.G3File('/home/adama/SPT/spt3g_software/calibration/scripts/testout_mat5a_mean.g3'))[0]

bolos_old_2019 = np.array(list(dold2019['BolometerProperties'].keys()))
bolos_new_2019 = np.array(list(dnew2019_rcw38['BolometerProperties'].keys()))
```

## Comparison of RCW38 Offsets

```python
plt.figure(figsize=(12,8))

for bolo in bolos_new_2019:
    if bolo in bolos_old_2019:
        plt.plot([dold2019['BolometerProperties'][bolo].x_offset,
                  dnew2019_rcw38['BolometerProperties'][bolo].x_offset],
                 [dold2019['BolometerProperties'][bolo].y_offset,
                  dnew2019_rcw38['BolometerProperties'][bolo].y_offset], 'k-', linewidth=0.5)
plt.title('difference between old and new 2019 (part 1) offsets')
plt.savefig('offset_shift_2019_rcw38.png', dpi=400)
```

```python
plt.figure(figsize=(12,8))
xoffset = [dold2019['BolometerProperties'][bolo].x_offset \
           for bolo in bolos_old_2019]
yoffset = [dold2019['BolometerProperties'][bolo].y_offset \
           for bolo in bolos_old_2019]
plt.plot(xoffset, yoffset, 'k.', linewidth=0.5, markersize=2)
plt.title('old 2019 offsets offsets')
plt.savefig('offset_old_2019_rcw38.png', dpi=400)
```

```python
plt.figure(figsize=(12,8))
xoffset = [dnew2019_rcw38['BolometerProperties'][bolo].x_offset \
           for bolo in bolos_new_2019]
yoffset = [dnew2019_rcw38['BolometerProperties'][bolo].y_offset \
           for bolo in bolos_new_2019]
plt.plot(xoffset, yoffset, 'k.', linewidth=0.5, markersize=2)
plt.title('new 2019 offsets offsets based on RCW38')
plt.savefig('offset_new_2019_rcw38.png', dpi=400)
```

## Comparison of MAT5A Offsets

```python
plt.figure(figsize=(12,8))
bolos_old_2019 = np.array(list(dold2019['BolometerProperties'].keys()))
bolos_new_2019 = np.array(list(dnew2019_mat5a['BolometerProperties'].keys()))

X = []
Y = []
U = []
V = []
for bolo in bolos_old_2019:
    if bolo in bolos_new_2019:
        plt.plot([dold2019['BolometerProperties'][bolo].x_offset,
                  dnew2019_mat5a['BolometerProperties'][bolo].x_offset],
                 [dold2019['BolometerProperties'][bolo].y_offset,
                  dnew2019_mat5a['BolometerProperties'][bolo].y_offset], 'k-', linewidth=0.5)
# plt.quiver(X, Y, U, V, angles='xy', headaxislength=0, headlength=0, linewidth=0.2)
plt.title('difference between old and new 2019 (part 1) offsets')
plt.savefig('offset_shift_2019_mat5a.png', dpi=400)
```

```python
plt.figure(figsize=(12,8))
xoffset = [dold2019['BolometerProperties'][bolo].x_offset \
           for bolo in bolos_old_2019
           if bolo in bolos_new_2019]
yoffset = [dold2019['BolometerProperties'][bolo].y_offset \
           for bolo in bolos_old_2019
           if bolo in bolos_new_2019]
plt.plot(xoffset, yoffset, 'k.', linewidth=0.5, markersize=2)
plt.title('old 2019 offsets offsets')
plt.savefig('offset_old_2019_mat5a.png', dpi=400)
```

```python
plt.figure(figsize=(12,8))
xoffset = [dnew2019_mat5a['BolometerProperties'][bolo].x_offset \
           for bolo in bolos_old_2019
           if bolo in bolos_new_2019]
yoffset = [dnew2019_mat5a['BolometerProperties'][bolo].y_offset \
           for bolo in bolos_old_2019
           if bolo in bolos_new_2019]
plt.plot(xoffset, yoffset, 'k.', linewidth=0.5, markersize=2)
plt.title('new 2019 offsets offsets based on MAT5A')
plt.savefig('offset_new_2019_mat5a.png', dpi=400)
```

## Final plots


Start with histogram of error between new and old offsets (using RCW38, which appear to be marginally more reliable).

```python
error = []
new_bolos = []
shifted_bolos = []
nan_bolos = []

for bolo in bolos_old_2019:
    if bolo in bolos_new_2019:
        bolo_error = np.sqrt((dold2019['BolometerProperties'][bolo].x_offset - \
                              dnew2019_rcw38['BolometerProperties'][bolo].x_offset)**2 + \
                             (dold2019['BolometerProperties'][bolo].y_offset - \
                              dnew2019_rcw38['BolometerProperties'][bolo].y_offset)**2)
        error.append(bolo_error)
        if bolo_error > 0.00025:
            shifted_bolos.append(bolo)
        if not np.isfinite(bolo_error):
            nan_bolos.append(bolo)

for bolo in bolos_new_2019:
    if bolo not in bolos_old_2019:
        print(bolo)
        new_bolos.append(bolo)
        
        
error = np.array(error)
plt.gca().set_yscale('log')
_ = plt.hist(error[np.isfinite(error)], bins=np.linspace(0,0.001,101))
```

```python
plt.figure(figsize=(12,8))
xoffset = np.array([dnew2019_rcw38['BolometerProperties'][bolo].x_offset \
                    for bolo in bolos_new_2019])
yoffset = np.array([dnew2019_rcw38['BolometerProperties'][bolo].y_offset \
                    for bolo in bolos_new_2019])
plt.plot(xoffset, yoffset, 'k.', linewidth=0.5, markersize=2,
         label='all offsets')

xoffset = [dnew2019_rcw38['BolometerProperties'][bolo].x_offset \
           for bolo in shifted_bolos]
yoffset = [dnew2019_rcw38['BolometerProperties'][bolo].y_offset \
           for bolo in shifted_bolos]
plt.plot(xoffset, yoffset, 'r.', linewidth=0.5, markersize=6,
         label='bolometers with shift > 0.00025')

xoffset = [dnew2019_rcw38['BolometerProperties'][bolo].x_offset \
           for bolo in nan_bolos]
yoffset = [dnew2019_rcw38['BolometerProperties'][bolo].y_offset \
           for bolo in nan_bolos]
plt.plot(xoffset, yoffset, '.', color='C1', linewidth=0.5, markersize=6,
         label='offsets that were `nan` in old offsets')
plt.legend()

# plt.axis([-0.003, -0.002, 0.004, 0.005])
plt.title('new 2019 offsets offsets based on RCW38')
plt.savefig('offset_new_2019_rcw38.png', dpi=400)
```

Calculate number of bolometers with `nan` offsets in old and new offset lists.

```python
xoffset_new = np.array([dnew2019_rcw38['BolometerProperties'][bolo].x_offset \
                        for bolo in bolos_new_2019])
yoffset_new = np.array([dnew2019_rcw38['BolometerProperties'][bolo].y_offset \
                        for bolo in bolos_new_2019])
xoffset_old = np.array([dold2019['BolometerProperties'][bolo].x_offset \
                        for bolo in bolos_old_2019])
yoffset_old = np.array([dold2019['BolometerProperties'][bolo].y_offset \
                        for bolo in bolos_old_2019])

bolos_lost = []
for bolo in bolos_old_2019:
    if np.isfinite(dold2019['BolometerProperties'][bolo].x_offset) and \
       bolo in bolos_new_2019 and \
       not np.isfinite(dnew2019_rcw38['BolometerProperties'][bolo].x_offset):
        bolos_lost.append(bolo)

n_good_offsets_new = len(xoffset_new[np.isfinite(xoffset_new)])
n_good_offsets_old = len(xoffset_old[np.isfinite(xoffset_old)])

print('old # bolos with good offsets = {}'.format(n_good_offsets_old))
print('new # bolos with good offsets = {}'.format(n_good_offsets_new))
print('# bolos with good offsets in old but not new = {}'.format(len(bolos_lost)))
```

```python
bolos_new_2019[(xoffset_new > -0.0026) & (xoffset_new < -0.0024) & \
               (yoffset_new > 0.0044) & (yoffset_new < 0.0046)]
```

```python

```
