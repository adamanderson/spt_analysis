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

# New BolometerProperties
Around April 20, various people discovered that our polarization angles were incorrectly defined. I propagated the change to the 2018 (both 1st and 2nd half) and 2019 hardware maps, and realized that we also needed to regenerate the bolometer properties. It turns out that the existing bolometer properties were quite old and contained some bolometer matching errors because they were not based on the most up-to-date hardware maps. My "propagation" of the polarization angles from the latest hardware map to the bolometer properties therefore included a variety of other changes due to improvements in the hardware map.

For 2018 (2nd half) and 2019 bolometer properties, I did the following:
1. Update hardware map with matching changes, as needed. For 2018 (2nd half) data, I used `hardware_maps_southpole/2018/hwm_pole_run2_post_event_5`. For 2019 data, I used `hardware_maps_southpole/2019/hwm_pole`.
1. Generate new nominal bolometer properties by loading the hardware map and running `make_new_bpm()` from the MTS.
1. Get the value of `"BolometerPropertiesBasedOn"` from the existing bolometer properties. Use this as an argument to `build_bolo_props_40000000.py` (2nd half 2018 data) and `build_bolo_props_60000000.py` (2019 data) to generate new bolometer properties frames.

The 1st half 2018 data is slightly complicated by the fact that we had not yet added the `"BolometerPropertiesBasedOn"` field to the `BolometerProperties` frame at the time that we generated this file. I therefore did not regenerate the bolometer properties with a version of the `build_bolo_props.py` script. Instead of regenerating the `BolometerProperties` with a new hardware map, I simply modified the polarization angles of the existing frame in-place. Specifically, I did the following:
1. Run `make_new_bpm()` from the MTS using `hardware_maps_southpole/2018/hwm_pole_run2_include_darks_v2/`.
1. Replace polarization angles in existing `BolometerProperties` frame with polarization angles from the nominal properties generated in the previous step, using the script `update_boloprops_inplace.py` in this directory. If a bolometer in the old `BolometerProperties` is not present in the new HWM, then remove that bolometer from the `BolometerProperties` frame altogether. There were 34 such bolometers.

Note that I also modified `build_bolo_props_*.py` and `build_cal_frames.py` in order to use an optional new method for outlier removal. The idea is to cut values of pointing offsets that differ by more than 2 pixel spacings from the median offset for all bolometers at the same pixel. This works slightly better than the current method, although there may still be a few outliers that could be hand-removed.

```python
# dold2018_1 = list(core.G3File('/spt/user/production/calibration/boloproperties/31000000.g3'))[0]
# dnew2018_1 = list(core.G3File('/home/adama/SPT/spt_analysis/20190501_new_boloprops/31000000_new.g3'))[0]
dold2018_2 = list(core.G3File('/spt/user/production/calibration/boloproperties/40000000.g3'))[0]
dnew2018_2 = list(core.G3File('/home/adama/SPT/spt3g_software/calibration/scripts/40000000_new.g3'))[0]
dold2019 = list(core.G3File('/spt/user/production/calibration/boloproperties/60000000.g3'))[0]
dnew2019 = list(core.G3File('/home/adama/SPT/spt3g_software/calibration/scripts/60000000_new.g3'))[0]
dnominal2019 = list(core.G3File('/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/nominal_online_cal.g3'))[0]
```

## Detector Check


Let's check the number of bolometers in each set of properties and the number with valid offsets between 2018 and 2019. The large discrepancy in number of detectors is due to the fact that the old bolometer properties uses data from July 3 2018 (quite old!).

```python
def check_detectors(dold, dnew):
    ndet_old = len(dold['BolometerProperties'].keys())
    ndet_new = len(dnew['BolometerProperties'].keys())
    print('# detectors (new / old) = {} / {}'.format(ndet_new, ndet_old))
    
    xoffsets_old = np.array([dold['BolometerProperties'][bolo].x_offset \
                             for bolo in dold['BolometerProperties'].keys()])
    xoffsets_new = np.array([dnew['BolometerProperties'][bolo].x_offset \
                             for bolo in dnew['BolometerProperties'].keys()])
    ndet_old = len(xoffsets_old[np.isfinite(xoffsets_old)])
    ndet_new = len(xoffsets_new[np.isfinite(xoffsets_new)])
    print('# valid offsets (new / old) = {} / {}'.format(ndet_new, ndet_old))
```

```python
print('2018 (part 1):')
check_detectors(dold2018_1, dnew2018_1)
```

```python
print('2018 (part 2):')
check_detectors(dold2018_2, dnew2018_2)
```

```python
print('2019:')
check_detectors(dold2019, dnew2019)
```

## Polarization Angle Check


I had wanted to check that all the bolometers have +/-90 degree shifts in polarization angle due to the recent polarization angle convention change. Unfortunately, the new bolometer properties also reflect new matching, which results in differences of polarization angle at other spurious values.

```python
def check_pol_angles(dold, dnew):
    bolos_old = list(dold['BolometerProperties'].keys())
    bolos_new = list(dnew['BolometerProperties'].keys())
    bolos = np.intersect1d(bolos_old, bolos_new)
    angles_old = np.array([dold['BolometerProperties'][b].pol_angle \
                           for b in bolos])
    angles_new = np.array([dnew['BolometerProperties'][b].pol_angle \
                           for b in bolos])
    return angles_old, angles_new, bolos
```

The 2018 1st half polarization angles change by peculiar quantities. This is not cause for concern because during the 2nd half of 2018 we also redefined our polarization angles to reflect across the x (or was it y?) axis, in order to correctly account for how rays propagate through the optics and onto the sky. Reflection about x corresponds to a transformation $\phi \rightarrow 180^\circ - \phi$, which induces values of $\Delta \phi$ that are not multiples of $45^\circ$.

```python
pol_old, pol_new, bolo = check_pol_angles(dold2018_1, dnew2018_1)
dangle = (pol_old - pol_new) / core.G3Units.deg
print(np.unique(dangle[np.isfinite(dangle)]))
_ = plt.hist(np.round(dangle[np.isfinite(dangle)]),
             bins=np.linspace(-270, 90, 91))
plt.gca().set_yscale('log')
```

```python
pol_old, pol_new, bolo = check_pol_angles(dold2018_2, dnew2018_2)
dangle = (pol_old - pol_new) / core.G3Units.deg
print(np.round(np.unique(dangle[np.isfinite(dangle)])))
_ = plt.hist(np.round(dangle[np.isfinite(dangle)]),
             bins=np.linspace(-270, 90, 91))
plt.gca().set_yscale('log')
```

```python
pol_old, pol_new, bolos = check_pol_angles(dold2019, dnew2019)
dangle = (pol_old - pol_new) / core.G3Units.deg
print(np.round(np.unique(dangle[np.isfinite(dangle)])))
_ = plt.hist(np.round(dangle[np.isfinite(dangle)]),
             bins=np.linspace(-180, 180, 91))
```

## Offsets Check


Let's check that the offsets are actually similar between new and old bolometer properties. In particular, for 2018 data, the offsets really should be totally unchanged.

```python
plt.figure(figsize=(12,8))
bolos_old_2018 = np.array(list(dold2018_1['BolometerProperties'].keys()))
bolos_new_2018 = np.array(list(dnew2018_1['BolometerProperties'].keys()))
for bolo in bolos_old_2018:
    if bolo in bolos_new_2018:
        plt.plot([dold2018_1['BolometerProperties'][bolo].x_offset,
                  dnew2018_1['BolometerProperties'][bolo].x_offset],
                 [dold2018_1['BolometerProperties'][bolo].y_offset,
                  dnew2018_1['BolometerProperties'][bolo].y_offset], 'k-', linewidth=0.5)
plt.title('difference between old and new 2018 (part 1) offsets')
```

```python
plt.figure(figsize=(12,8))
bolos_old_2018 = np.array(list(dold2018_2['BolometerProperties'].keys()))
bolos_new_2018 = np.array(list(dnew2018_2['BolometerProperties'].keys()))
for bolo in bolos_old_2018:
    if bolo in bolos_new_2018:
        plt.plot([dold2018_2['BolometerProperties'][bolo].x_offset,
                  dnew2018_2['BolometerProperties'][bolo].x_offset],
                 [dold2018_2['BolometerProperties'][bolo].y_offset,
                  dnew2018_2['BolometerProperties'][bolo].y_offset], 'k-', linewidth=0.5)
plt.title('difference between old and new 2018 (part 2) offsets')
```

```python
plt.figure(figsize=(12,8))
bolos_old_2019 = np.array(list(dold2019['BolometerProperties'].keys()))
bolos_new_2019 = np.array(list(dnew2019['BolometerProperties'].keys()))
for bolo in bolos_old_2019:
    if bolo in bolos_new_2019:
        if dnew2019['BolometerProperties'][bolo].coupling == \
            calibration.BolometerCouplingType.Optical:
            plt.plot([dold2019['BolometerProperties'][bolo].x_offset,
                      dnew2019['BolometerProperties'][bolo].x_offset],
                     [dold2019['BolometerProperties'][bolo].y_offset,
                      dnew2019['BolometerProperties'][bolo].y_offset], 'k-', linewidth=0.5)
plt.title('difference between old and new 2019 offsets')
```

## Detailed inspection of 2019 offsets

```python
plt.figure(figsize=(12,8))
plt.plot([dold2019['BolometerProperties'][bolo].x_offset \
          for bolo in dold2019['BolometerProperties'].keys()],
         [dold2019['BolometerProperties'][bolo].y_offset \
          for bolo in dold2019['BolometerProperties'].keys()], '.')
plt.title('old 2019 offsets')
```

```python
plt.figure(figsize=(12,8))
plt.plot([dnew2019['BolometerProperties'][bolo].x_offset \
          for bolo in dnew2019['BolometerProperties'].keys()],
         [dnew2019['BolometerProperties'][bolo].y_offset \
          for bolo in dnew2019['BolometerProperties'].keys()], '.')
plt.title('new 2019 offsets')
```

To find remaining outliers, let's look for bolometers whose offsets differ from nominal. Unfortunately all the bolometers offsets differ from nominal, but they do so in a very systematic way across the focal plane. To identify outliers, we therefore look for bolometers whose offsets are different from nominal in a way that is very unlike the average differences of their neighbors.

```python
bolonames = np.array([bolo for bolo in dnominal2019["NominalBolometerProperties"].keys()])
xnominal = np.array([dnominal2019["NominalBolometerProperties"][bolo].x_offset \
                     for bolo in dnominal2019["NominalBolometerProperties"].keys()])
ynominal = np.array([dnominal2019["NominalBolometerProperties"][bolo].y_offset \
                     for bolo in dnominal2019["NominalBolometerProperties"].keys()])
xnominal = xnominal[np.isfinite(xnominal) & np.isfinite(ynominal)]
ynominal = ynominal[np.isfinite(xnominal) & np.isfinite(ynominal)]
bolonames = bolonames[np.isfinite(xnominal) & np.isfinite(ynominal)]

ddx = []
ddy = []
dd_bolos = []
bololist = np.intersect1d(list(dnew2019['BolometerProperties'].keys()),
                          list(dnominal2019["NominalBolometerProperties"].keys()))
for bolo in bololist:
    dx = dnew2019['BolometerProperties'][bolo].x_offset - \
         dnominal2019["NominalBolometerProperties"][bolo].x_offset
    dy = dnew2019['BolometerProperties'][bolo].y_offset - \
         dnominal2019["NominalBolometerProperties"][bolo].y_offset

    x0 = dnominal2019["NominalBolometerProperties"][bolo].x_offset
    y0 = dnominal2019["NominalBolometerProperties"][bolo].y_offset
    nearest_bolos = bolonames[np.argsort(np.abs(x0 - xnominal))][:10]
    dx_nearest = np.array([dnew2019['BolometerProperties'][bolo2].x_offset - \
                           dnominal2019["NominalBolometerProperties"][bolo2].x_offset \
                           for bolo2 in nearest_bolos])
    dx_nearest = np.median(dx_nearest[np.isfinite(dx_nearest)])
    dy_nearest = np.array([dnew2019['BolometerProperties'][bolo2].y_offset - \
                           dnominal2019["NominalBolometerProperties"][bolo2].y_offset \
                           for bolo2 in nearest_bolos])
    dy_nearest = np.median(dy_nearest[np.isfinite(dy_nearest)])
    ddx.append(dx_nearest - dx)
    ddy.append(dy_nearest - dy)
    dd_bolos.append(bolo)
```

```python
ddx = np.asarray(ddx)
ddy = np.asarray(ddy)
ddr = np.sqrt(ddx**2 + ddy**2)
dd_bolos = np.asarray(dd_bolos)
plt.figure(figsize=(8,6))
_ = plt.hist(ddx[np.isfinite(ddx)],
             bins=np.linspace(-0.001, 0.001, 101),
             histtype='step', label='x')
_ = plt.hist(ddy[np.isfinite(ddy)],
             bins=np.linspace(-0.001, 0.001, 101),
             histtype='step', label='y')
plt.legend()
plt.gca().set_yscale('log')

plt.figure(figsize=(8,6))
_ = plt.hist(ddr[np.isfinite(ddr)],
             bins=np.linspace(0, 0.001, 101),
             histtype='step', label='r')
plt.legend()
plt.gca().set_yscale('log')
```

We can define an arbitrary cut on the difference of bolometers from nominal below to try to identify outliers.

```python
# bolos_anomalous = dd_bolos[(np.abs(ddx)>0.0003) | (np.abs(ddy)>0.0012)]
bolos_anomalous = dd_bolos[(np.abs(ddr)>0.0008)]
```

```python
bolos_anomalous
```

```python
xoffset = [dnew2019['BolometerProperties'][bolo].x_offset \
           for bolo in dnew2019['BolometerProperties'].keys() \
           if dnew2019['BolometerProperties'][bolo].coupling == calibration.BolometerCouplingType.Optical]
yoffset = [dnew2019['BolometerProperties'][bolo].y_offset \
           for bolo in dnew2019['BolometerProperties'].keys() \
           if dnew2019['BolometerProperties'][bolo].coupling == calibration.BolometerCouplingType.Optical]
plt.figure(figsize=(12,8))
plt.plot(xoffset, yoffset, '.')
for bolo in bolos_anomalous:
    if dnew2019['BolometerProperties'][bolo].coupling == calibration.BolometerCouplingType.Optical:
        plt.plot(dnew2019['BolometerProperties'][bolo].x_offset,
                 dnew2019['BolometerProperties'][bolo].y_offset, 'r.')
plt.axis([-0.02, 0.02, -0.02, 0.02])
plt.title('new 2019 offsets')


plt.figure(figsize=(12,8))
for bolo in bolos_new_2019:
    if bolo in dnominal2019["NominalBolometerProperties"]:
        if dnominal2019["NominalBolometerProperties"][bolo].coupling == \
           calibration.BolometerCouplingType.Optical:
            plt.plot([dnew2019['BolometerProperties'][bolo].x_offset,
                      dnominal2019["NominalBolometerProperties"][bolo].x_offset],
                     [dnew2019['BolometerProperties'][bolo].y_offset,
                      dnominal2019["NominalBolometerProperties"][bolo].y_offset], 'k-', linewidth=0.5)
for bolo in bolos_anomalous:
    if bolo in dnominal2019["NominalBolometerProperties"]:
        if dnominal2019["NominalBolometerProperties"][bolo].coupling == \
           calibration.BolometerCouplingType.Optical:
            plt.plot([dnew2019['BolometerProperties'][bolo].x_offset,
                      dnominal2019["NominalBolometerProperties"][bolo].x_offset],
                     [dnew2019['BolometerProperties'][bolo].y_offset,
                      dnominal2019["NominalBolometerProperties"][bolo].y_offset], 'r-', linewidth=0.5)
plt.axis([-0.02, 0.02, -0.02, 0.02])
plt.title('difference between nominal offsets and new 2019 offsets')
```

```python
dd_bolos[(np.abs(ddr)>0.008)]
```

## Follow-up on question from NDH
Nicholas Huang raised the question of why there are about 1000 detectors in the elnod observation `/spt/user/production/calibration/elnod/71084755.g3`, which are not present in the bolometer properties `/spt/user/production/calibration/boloproperties/60000000.g3`. Let's try to understand why this is occurring.

```python
boloprops = list(core.G3File('/spt/user/production/calibration/boloproperties/60000000.g3'))\
                [0]["BolometerProperties"]
# recent elnod: 81046089
# problematic elnod: 71084755
elnod = list(core.G3File('/spt/user/production/calibration/elnod/81046089.g3'))\
                [0]["ElnodSlopes"]
```

```python
bolos_in_elnod_not_bps = []
bolos_bps = list(boloprops.keys())
bolos_elnod = list(elnod.keys())
for bolo in bolos_elnod:
    if bolo not in bolos_bps:
        bolos_in_elnod_not_bps.append(bolo)
```

```python
bolos_in_elnod_not_bps
```

```python
/spt/user/production/calibration/elnod/81046089.g3
```

```python

```
