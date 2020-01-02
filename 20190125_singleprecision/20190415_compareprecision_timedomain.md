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

# Comparing single- and double-precision in the time domain
There have been some unexpected disagreement between single- and double-precision data in the map domain. It's hard for me to tell whether these issues are introduced at the map binning level, or at the time-domain processing level. In order to understand this issue, I decided to save data at each level of timestream processing and compare between single- and double-precision processing piplines.

```python
from spt3g import core, calibration, dfmux
import numpy as np
import matplotlib.pyplot as plt
print(core.__file__)
```

## After calibration
I saved timestreams after the `CalibrateRawTimestreams` pipeline segment. This is the `v4` mapmaking script. Let's write a quick pipeline segment to compute and save the difference between them in each scan frame.

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_maps_v4.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_maps_v4.g3')
write_diff = core.G3Writer('64502043_v4_diff.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['CalTimestreamsDiff'] = core.G3TimestreamMap()
        new_frame['CalTimestreamsSingle'] = fr_single['CalTimestreams']
        new_frame['CalTimestreamsDouble'] = fr_double['CalTimestreams']
        for bolo in fr_single['CalTimestreams'].keys():
            new_frame['CalTimestreamsDiff'][bolo] = fr_single['CalTimestreams'][bolo] - \
                                                    fr_double['CalTimestreams'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
fdiff = list(core.G3File('64502043_v4_diff.g3'))
```

```python
np.array(fdiff[3]['CalTimestreamsDiff']['2019.03w']) / np.array(fdiff[3]['CalTimestreamsSingle']['2019.03w'])
```

```python
plt.plot(np.array(fdiff[3]['CalTimestreamsDiff']['2019.03w']) / np.array(fdiff[3]['CalTimestreamsSingle']['2019.03w']))
```

```python
fractional_diff = []
for scan in fdiff:
    for bolo in scan['CalTimestreamsDiff'].keys():
        fractional_diff_arr = np.array(scan['CalTimestreamsDiff'][bolo]) / \
                                np.array(scan['CalTimestreamsSingle'][bolo])
        fractional_diff.append(np.max(fractional_diff_arr) - np.min(fractional_diff_arr))
```

```python
_ = plt.hist(fractional_diff)
```

## After common-mode filter

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_maps_v6.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_maps_v6.g3')
write_diff = core.G3Writer('64502043_v6_diff.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['CMFilteredTimestreamsDiff'] = core.G3TimestreamMap()
        new_frame['CMFilteredTimestreamsSingle'] = fr_single['CMFilteredTimestreams']
        new_frame['CMFilteredTimestreamsDouble'] = fr_double['CMFilteredTimestreams']
        for bolo in fr_single['CMFilteredTimestreams'].keys():
            if bolo not in fr_double['CMFilteredTimestreams'].keys():
                print('Cannot find bolometer {}'.format(bolo))
            else:
                new_frame['CMFilteredTimestreamsDiff'][bolo] = fr_single['CMFilteredTimestreams'][bolo] - \
                                                               fr_double['CMFilteredTimestreams'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
fdiff = list(core.G3File('diff/64502043_v6_diff.g3'))
fractional_diff = []
for scan in fdiff:
    for bolo in scan['CMFilteredTimestreamsDiff'].keys():
        fractional_diff_arr = np.array(scan['CMFilteredTimestreamsDiff'][bolo])
        fractional_diff.append(np.max(fractional_diff_arr) - np.min(fractional_diff_arr))
fractional_diff = np.array(fractional_diff)
```

```python
plt.plot(scan['CMFilteredTimestreamsDiff']['2019.0gv'] / scan['CMFilteredTimestreamsSingle']['2019.0gv'])
```

```python
dsingle = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'singleprecision/64502043_maps_v6.g3'))
ddouble = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'doubleprecision/64502043_maps_v6.g3'))
```

```python
for jframe in np.arange(0,10):
    diff_std = []
    for bolo in fdiff[jframe]['CMFilteredTimestreamsDiff'].keys():
        diff_std.append(np.std(fdiff[jframe]['CMFilteredTimestreamsDiff'][bolo]))
    _ = plt.hist(diff_std, bins=np.linspace(0,10,101), histtype='step')
plt.gca().set_yscale('log')
```

```python

```

## Raw data

```python
fsingle = core.G3File('/spt/data/bolodata/downsampled/ra0hdec-67.25/64085020/0000.g3')
fdouble = core.G3File('/spt/data/bolodata/downsampled/ra0hdec-67.25/64085020/0000.g3')
write_diff = core.G3Writer('64502043_v7_diff.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['RawTimestreams_IDiff'] = core.G3TimestreamMap()
        new_frame['RawTimestreams_ISingle'] = fr_single['RawTimestreams_I']
        new_frame['RawTimestreams_IDouble'] = fr_double['RawTimestreams_I']
        for bolo in fr_single['RawTimestreams_I'].keys():
            new_frame['RawTimestreams_IDiff'][bolo] = fr_single['RawTimestreams_I'][bolo] - \
                                                    fr_double['RawTimestreams_I'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
fdiff = list(core.G3File('64502043_v7_diff.g3'))
fractional_diff = []
for scan in fdiff:
    for bolo in scan['RawTimestreams_IDiff'].keys():
        fractional_diff_arr = np.array(scan['RawTimestreams_IDiff'][bolo]) / \
                                np.array(scan['RawTimestreams_ISingle'][bolo])
        fractional_diff.append(np.max(fractional_diff_arr) - np.min(fractional_diff_arr))
fractional_diff = np.array(fractional_diff)
```

```python
_ = plt.hist(fractional_diff[np.isfinite(fractional_diff)], bins=np.linspace(-1e-7, 1e-7, 51))
plt.gca().set_yscale('log')
plt.xlabel('fractional difference [max - min of (single - double) / single per scan]')
plt.ylabel('number of bolometer-scans')
plt.title('read & write timestreams only')
plt.tight_layout()
plt.savefig('diff_readwriteonly.png', dpi=200)
```

```python
np.std(fractional_diff)
```

## After poly filter
At this point, I have also completed the modification to the common-mode filter to reduce the round-off errors there. A useful thing is therefore to compare the errors after the poly filter with the error after the common-mode filter.

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_maps_v10.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_maps_v10.g3')
write_diff = core.G3Writer('diff/64502043_v10_diff.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['PolyFilteredTimestreamsDiff'] = core.G3TimestreamMap()
        new_frame['PolyFilteredTimestreamsSingle'] = fr_single['PolyFilteredTimestreams']
        new_frame['PolyFilteredTimestreamsDouble'] = fr_double['PolyFilteredTimestreams']
        for bolo in fr_single['PolyFilteredTimestreams'].keys():
            if bolo not in fr_double['PolyFilteredTimestreams'].keys():
                print('Cannot find bolometer {}'.format(bolo))
            else:
                new_frame['PolyFilteredTimestreamsDiff'][bolo] = fr_single['PolyFilteredTimestreams'][bolo] - \
                                                               fr_double['PolyFilteredTimestreams'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
fdiff = list(core.G3File('diff/64502043_v10_diff.g3'))
fractional_diff = []
for scan in fdiff:
    for bolo in scan['PolyFilteredTimestreamsDiff'].keys():
        fractional_diff_arr = np.array(scan['PolyFilteredTimestreamsDiff'][bolo]) / \
                                np.array(scan['PolyFilteredTimestreamsSingle'][bolo])
        fractional_diff.append(np.median(fractional_diff_arr))
fractional_diff = np.array(fractional_diff)
```

```python
plt.plot(fdiff[1]['PolyFilteredTimestreamsDiff']['2019.hqx'])
```

```python
for jframe in np.arange(0,10):
    diff_std = []
    for bolo in fdiff[jframe]['PolyFilteredTimestreamsDiff'].keys():
        diff_std.append(np.std(fdiff[jframe]['PolyFilteredTimestreamsDiff'][bolo]))
    _ = plt.hist(diff_std, bins=np.linspace(0,0.1,101), histtype='step')
plt.gca().set_yscale('log')
```

## Checking weights

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_maps_v12.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_maps_v12.g3')
write_diff = core.G3Writer('diff/64502043_v12_diff.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['TodWeightsDiff'] = core.G3MapDouble()
        new_frame['TodWeightsSingle'] = fr_single['TodWeights']
        new_frame['TodWeightsDouble'] = fr_double['TodWeights']
        for bolo in fr_single['TodWeights'].keys():
            if bolo not in fr_double['TodWeights'].keys():
                print('Cannot find bolometer {}'.format(bolo))
            else:
                new_frame['TodWeightsDiff'][bolo] = fr_single['TodWeights'][bolo] - \
                                                    fr_double['TodWeights'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
fdiff = list(core.G3File('diff/64502043_v12_diff.g3'))
weights_diff = np.array([fdiff[0]['TodWeightsDiff'][bolo]
                         for bolo in fdiff[0]['TodWeightsDiff'].keys()])
weights_single = np.array([fdiff[0]['TodWeightsSingle'][bolo]
                           for bolo in fdiff[0]['TodWeightsDiff'].keys()])
bolos = np.array([bolo for bolo in fdiff[0]['TodWeightsDiff'].keys()])
_ = plt.hist(weights_diff / weights_single, bins=np.linspace(-2e-3, 2e-3, 101))
plt.gca().set_yscale('log')
```

```python
ratio = weights_diff / weights_single
plt.plot(weights_single, weights_diff, '.')
plt.xlim([0, 0.1])
```

```python
plt.subplot(2,1,1)
_ = plt.hist(weights_single, bins=np.linspace(0,0.1,101))
plt.subplot(2,1,2)
plt.plot(weights_single, weights_diff, '.')
plt.xlim([0, 0.1])
```

```python
bolos[np.abs(weights_diff)>1e-5]
```

```python
plt.plot(fr_single['PolyFilteredTimestreams']['2019.tfk'] - \
         fr_double['PolyFilteredTimestreams']['2019.tfk'])
print(fdiff[0]['TodWeightsSingle']['2019.tfk'])
print(fdiff[0]['TodWeightsDouble']['2019.tfk'])
```

```python
plt.plot(fr_single['PolyFilteredTimestreams']['2019.008'] - \
         fr_double['PolyFilteredTimestreams']['2019.008'])
print(fdiff[0]['TodWeightsSingle']['2019.008'])
print(fdiff[0]['TodWeightsDouble']['2019.008'])
```

## Checking PSDs

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_maps_v12.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_maps_v12.g3')
write_diff = core.G3Writer('diff/64502043_v12_diff.g3')

while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['TodWeightsDiff'] = core.G3MapDouble()
        new_frame['TodWeightsSingle'] = fr_single['TodWeights']
        new_frame['TodWeightsDouble'] = fr_double['TodWeights']
        for bolo in fr_single['TodWeights'].keys():
            if bolo not in fr_double['TodWeights'].keys():
                print('Cannot find bolometer {}'.format(bolo))
            else:
                new_frame['TodWeightsDiff'][bolo] = fr_single['TodWeights'][bolo] - \
                                                    fr_double['TodWeights'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
dsingle = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'singleprecision/64502043_maps_v12.g3'))
ddouble = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'doubleprecision/64502043_maps_v12.g3'))
```

```python
for jframe in np.arange(5,15):
    bolos_single = np.array([bolo for bolo in dsingle[jframe]['TodWeights'].keys()])
    bolos_double = np.array([bolo for bolo in ddouble[jframe]['TodWeights'].keys()])
    bolos = np.intersect1d(bolos_single, bolos_double)
    wafers = 
    weights_diff = np.array([dsingle[jframe]['TodWeights'][bolo] - \
                             ddouble[jframe]['TodWeights'][bolo]
                             for bolo in bolos])

    _ = plt.hist(weights_diff, bins=np.linspace(-2e-4, 2e-4, 101),
                 histtype='step', label='{}'.format(jframe))
    plt.gca().set_yscale('log')
plt.legend()
```

```python
# plt.figure(1)
plt.loglog(ddouble[5]["PolyFilteredFreq"] / core.G3Units.Hz,
           ddouble[5]["PolyFilteredPSD"]['2019.hqx'])
# plt.loglog(fr_double["PolyFilteredFreq"] / core.G3Units.Hz,
#            fr_double["PolyFilteredPSD"]['2019.itw'])

# plt.figure(2)
plt.loglog(dsingle[5]["PolyFilteredFreq"] / core.G3Units.Hz,
           dsingle[5]["PolyFilteredPSD"]['2019.hqx'])
# plt.loglog(fr_double["PolyFilteredFreq"] / core.G3Units.Hz,
#            fr_double["PolyFilteredPSD"]['2019.008'])
```

```python
plt.semilogx(ddouble[5]["PolyFilteredFreq"] / core.G3Units.Hz,
             ddouble[5]["PolyFilteredPSD"]['2019.hqx'] - \
             dsingle[5]["PolyFilteredPSD"]['2019.hqx'])
```

```python
plt.plot(ddouble[5]["DeflaggedTimestreams"]['2019.l6i'])
plt.plot(dsingle[5]["DeflaggedTimestreams"]['2019.l6i'])

plt.figure(2)
plt.plot(ddouble[5]["DeflaggedTimestreams"]['2019.l6i'] - \
         dsingle[5]["DeflaggedTimestreams"]['2019.l6i'])
```

```python
plt.semilogx(ddouble[5]["PolyFilteredFreq"] / core.G3Units.Hz,
             ddouble[5]["PolyFilteredPSD"]['2019.008'] - \
             dsingle[5]["PolyFilteredPSD"]['2019.008'])
```

```python
plt.figure(1)
plt.plot(dsingle[4]["DeflaggedTimestreams"]['2019.hqx'])
plt.plot(dsingle[4]["DeflaggedTimestreams"]['2019.95x'])

plt.figure(2)
plt.plot(ddouble[4]["DeflaggedTimestreams"]['2019.hqx'])
plt.plot(ddouble[4]["DeflaggedTimestreams"]['2019.95x'])

plt.figure(3)
plt.plot(ddouble[4]["DeflaggedTimestreams"]['2019.95x'] - \
         dsingle[4]["DeflaggedTimestreams"]['2019.95x'])

plt.figure(4)
plt.plot(ddouble[4]["DeflaggedTimestreams"]['2019.hqx'] - \
         dsingle[4]["DeflaggedTimestreams"]['2019.hqx'])

plt.figure(5)
plt.plot(ddouble[4]["DeflaggedTimestreams"]['2019.hqx'] - \
         dsingle[4]["DeflaggedTimestreams"]['2019.hqx'] - \
         (ddouble[4]["DeflaggedTimestreams"]['2019.95x'] - \
         dsingle[4]["DeflaggedTimestreams"]['2019.95x']))
```

## Deflagged timestreams

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_maps_v10.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_maps_v10.g3')
write_diff = core.G3Writer('diff/64502043_v10_diff.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['DeflaggedTimestreamsDiff'] = core.G3TimestreamMap()
        new_frame['DeflaggedTimestreamsSingle'] = fr_single['DeflaggedTimestreams']
        new_frame['DeflaggedTimestreamsDouble'] = fr_double['DeflaggedTimestreams']
        for bolo in fr_single['DeflaggedTimestreams'].keys():
            if bolo not in fr_double['DeflaggedTimestreams'].keys():
                print('Cannot find bolometer {}'.format(bolo))
            else:
                new_frame['DeflaggedTimestreamsDiff'][bolo] = fr_single['DeflaggedTimestreams'][bolo] - \
                                                               fr_double['DeflaggedTimestreams'][bolo]
        print(new_frame)
        write_diff.Process(new_frame)
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
write_diff.Process(core.G3Frame(core.G3FrameType.EndProcessing))
```

```python
dsingle = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'singleprecision/64502043_maps_v10.g3'))
ddouble = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'doubleprecision/64502043_maps_v10.g3'))
```

```python
print(dsingle[10])
```

```python

```
