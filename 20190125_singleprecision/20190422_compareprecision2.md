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
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
ddiff_polyfilter = list(core.G3File('diff/64502043_v10_diff.g3'))
ddiff_weights = list(core.G3File('diff/64502043_v12_diff.g3'))
```

```python
print(ddiff_polyfilter[0])
```

```python
for jframe in [0]: #range(len(ddiff_polyfilter)):
    diff_std = np.array([np.std(ddiff_polyfilter[jframe]["DeflaggedTimestreamsDiff"][bolo])
                for bolo in ddiff_polyfilter[jframe]["DeflaggedTimestreamsDiff"].keys()])
    double_std = np.array([np.std(ddiff_polyfilter[jframe]["DeflaggedTimestreamsDouble"][bolo])
                  for bolo in ddiff_polyfilter[jframe]["DeflaggedTimestreamsDiff"].keys()])
    _ = plt.hist(diff_std / double_std,
                 bins=np.linspace(0, 1e-5, 101),
                 histtype='step')
plt.gca().set_yscale('log')
plt.xlabel('std(single - double) / std(double)')
plt.ylabel('bolometers')
plt.title('std. of difference TOD between single- and double-precision\n' + \
          'divided by std. of double-precision TOD (no common-mode filter)')
plt.tight_layout()
plt.savefig('std_diff_over_std_double_noCM.png', dpi=200)
```

```python
diff_std = np.array([np.std(ddiff_polyfilter[0]["PolyFilteredTimestreamsDiff"][bolo])
                for bolo in ddiff_polyfilter[0]["PolyFilteredTimestreamsDiff"].keys()])
double_std = np.array([np.std(ddiff_polyfilter[0]["PolyFilteredTimestreamsDouble"][bolo])
              for bolo in ddiff_polyfilter[0]["PolyFilteredTimestreamsDiff"].keys()])
bolos = np.array([bolo for bolo in ddiff_polyfilter[0]["PolyFilteredTimestreamsDiff"].keys()])
plt.figure()
_ = plt.hist(diff_std, bins=np.linspace(0,0.03,101))
plt.gca().set_yscale('log')

plt.figure()
_ = plt.hist(double_std, bins=np.linspace(0,20.0,101))
```

```python
dbp = list(core.G3File('/spt/data/bolodata/downsampled/ra0hdec-67.25/64085020/offline_calibration.g3'))
boloprops = dbp[0]['BolometerProperties']
len(bolos[diff_std>0.02])
```

```python
for bolo in bolos[diff_std>0.02]:
    print('{} / {}: {}'.format(boloprops[bolo].wafer_id,
                               boloprops[bolo].band / core.G3Units.GHz,
                               bolo))
```

```python
for bolo in bolos[diff_std>0.02]:
    plt.plot(ddiff_polyfilter[0]["PolyFilteredTimestreamsDouble"][bolo])
```

```python
draw = list(core.G3File('/spt/data/bolodata/downsampled/ra0hdec-67.25/64085020/0000.g3'))
```

```python
dsingle = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'singleprecision/64502043_maps_v10.g3'))
ddouble = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'doubleprecision/64502043_maps_v10.g3'))
```

## Dealing with flagging

```python
for j in range(len(dsingle)):
    if dsingle[j].type == core.G3FrameType.Scan:
        print(len(dsingle[j]['Flags']) - len(ddouble[j]['Flags']))
```

```python
for jframe in range(len(dsingle)):
    print('frame {}'.format(jframe))
    if 'Flags' in dsingle[jframe].keys():
        for bolo in dsingle[jframe]['Flags'].keys():
            if bolo in dsingle[jframe]['Flags'].keys() and \
               bolo in ddouble[jframe]['Flags'].keys() and \
               list(dsingle[jframe]['Flags'][bolo]) != list(ddouble[jframe]['Flags'][bolo]):
                print('{} vs. {}'.format(dsingle[jframe]['Flags'][bolo],
                                         ddouble[jframe]['Flags'][bolo]))
```

```python
for jframe in range(len(dsingle)):
    np.setdiff1d(list(dsingle[jframe]['Flags'].keys()),
                 list(dsingle[jframe]['Flags'].keys()))
```

## Checking weights again

```python
dsingle = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'singleprecision/64502043_tod_v13.g3'))
ddouble = list(core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'doubleprecision/64502043_tod_v13.g3'))
# ddiff = list(core.G3File('diff/64502043_v10_diff.g3'))
```

```python
print(dsingle[4])
```

```python
dweights_diff = []
dweights_single = []
for bolo in dsingle[4]["TodWeights"].keys():
    if bolo in dsingle[4]["TodWeights"].keys() and \
       bolo in ddouble[4]["TodWeights"].keys():
        dweights_diff.append(dsingle[4]["TodWeights"][bolo] - ddouble[4]["TodWeights"][bolo])
        dweights_single.append(dsingle[4]["TodWeights"][bolo])
dweights_diff = np.array(dweights_diff)
dweights_single = np.array(dweights_single)
```

```python
_ = plt.hist(dweights_diff / dweights_single,
             bins=np.linspace(-5e-7, 5e-7, 101),
             histtype='step')
plt.xlabel('TOD weights ((single - double) / single)')
plt.ylabel('bolometers')
plt.tight_layout()
plt.savefig('tod_weights_diff.png', dpi=200)
```

## Checking pointing

```python
plt.plot((dsingle[4]["OnlineBoresightAz"] - ddouble[4]["OnlineBoresightAz"]) / ddouble[4]["OnlineBoresightAz"])
plt.ylabel('OnlineBoresightAz error ((single - double) / double)')
plt.xlabel('samples')
plt.tight_layout()
plt.savefig('online_boresight_az_error.png', dpi=200)
```

```python
plt.plot((dsingle[4]["OnlineBoresightEl"] - ddouble[4]["OnlineBoresightEl"]) / ddouble[4]["OnlineBoresightEl"])
plt.ylabel('OnlineBoresightEl error ((single - double) / double)')
plt.xlabel('samples')
plt.tight_layout()
plt.savefig('online_boresight_el_error.png', dpi=200)
```

```python
plt.plot((dsingle[4]["OnlineBoresightRa"] - ddouble[4]["OnlineBoresightRa"]) / core.G3Units.arcmin)
```

```python
plt.plot((dsingle[14]["OnlineBoresightDec"] - ddouble[14]["OnlineBoresightDec"]) / core.G3Units.arcmin)
```

```python
plt.plot((dsingle[4]["RawBoresightAz"] - ddouble[4]["RawBoresightAz"]) / core.G3Units.arcmin)
```

```python
plt.plot((dsingle[4]["RawBoresightEl"] - ddouble[4]["RawBoresightEl"]) / core.G3Units.arcmin)
```

```python
plt.plot(np.array(dsingle[4]["PixelPointing"]['2019.2tq']) - np.array(ddouble[4]["PixelPointing"]['2019.2tq']))
plt.ylim([-2,2])
plt.ylabel('PixelPointing error (single - double)')
plt.xlabel('samples')
plt.title('single scan for detector 2019.2tq')
plt.tight_layout()
plt.savefig('pixel_pointing_error.png', dpi=200)
```

```python
plt.plot(np.array(ddouble[4]["PixelPointing"]['2019.2tq']))
plt.plot(np.array(dsingle[4]["PixelPointing"]['2019.2tq']))
```

## Checking Pointing
Let's get some statistics on the pointing.

```python
fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'singleprecision/64502043_tod_v13.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                           'doubleprecision/64502043_tod_v13.g3')
n_different_pixels = []
fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    print(len(n_different_pixels))
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        bolos_single = np.unique(list(fr_single['PixelPointing'].keys()))
        bolos_double = np.unique(list(fr_double['PixelPointing'].keys()))
        bolos = np.intersect1d(bolos_single, bolos_double)
        
        for bolo in bolos:
            pointing_diff = np.array(fr_single['PixelPointing'][bolo]) - \
                            np.array(fr_double['PixelPointing'][bolo])
            n_different_pixels.append(len(pointing_diff[pointing_diff!=0]))
    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break
```

```python
_ = plt.hist(n_different_pixels,
             bins=np.linspace(0,50,51),
             histtype='step')
plt.xlabel('number of samples per scan hitting different pixels\n(double vs. single)')
plt.ylabel('samples')
plt.legend()
plt.tight_layout()
plt.savefig('pointing_sample_errors.png', dpi=200)
```

```python
_ = plt.hist(n_different_pixels,
             bins=np.linspace(0,10000,101),
             histtype='step')
plt.gca().set_yscale('log')
plt.xlabel('number of samples per scan hitting different pixels\n(double vs. single)')
plt.ylabel('samples')
plt.legend()
plt.tight_layout()
plt.savefig('pointing_sample_errors_zoom.png', dpi=200)
```

```python
np.sum(n_different_pixels)
```

```python
np.sum(n_different_pixels) / (8000*20*8000)
```

```python

```
