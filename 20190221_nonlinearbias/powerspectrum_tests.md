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

## Quick estimation of power spectrum


Stupid test to make sure that I can easily generate power spectra in 3G software from Daniel's E-mode maps. I am just blindly applying functions here based on their docstrings. We'll do a more detailed write-up of what is going on down below.

```python
from spt3g import core, mapmaker, mapspectra
from spt3g.mapmaker import remove_weight
from spt3g.mapspectra.map_analysis import calculateCls, constructEB
import numpy as np
import matplotlib.pyplot as plt
```

```python
# load Daniel's maps
d = [fr for fr in core.G3File('/spt/user/ddutcher/coadds/coadd2018_noW201wafCMpoly19mhpf300.g3')]
```

```python
map_frame = d[4]
```

```python
apod = mapspectra.apodmask.makeBorderApodization(
           map_frame['Wpol'], apod_type='cos',
           radius_arcmin=15.,zero_border_arcmin=10,
           smooth_weights_arcmin=5)
plt.imshow(apod)
plt.colorbar()
```

```python
# ell_centers = np.linspace(100,4000,201)
# ell_bins = mapspectra.basicmaputils.get_reg_spaced_ell_bins(ell_centers)
# T_noweight, Q_noweight, U_noweight = remove_weight(map_frame['T'],
#                                                    map_frame['Q'],
#                                                    map_frame['U'],
#                                                    map_frame['Wpol'])
# Q_flat_noweight, U_flat_noweight = mapspectra.basicmaputils.flatten_pol(Q_noweight,
#                                                                         U_noweight)
# Cls = mapspectra.basicmaputils.get_map_cls(t=T_noweight,
#                                            q=Q_flat_noweight,
#                                            u=U_flat_noweight,
#                                            apod_mask=apod,
#                                            ell_bins=ell_bins)

Cls = calculateCls(fr=map_frame, apod_mask=apod)
```

```python
Cls.keys()
```

```python
# plt.loglog(ell_centers, Cls[0], '.')
plt.figure()
plt.loglog(Cls['ell'], Cls['EE'] * Cls['ell'] * (Cls['ell']+1)/(2*np.pi) / (core.G3Units.arcmin**2), '.')
plt.axis([300, 4000, 10, 1000])

plt.figure()
plt.plot(Cls['ell'], Cls['TT'] * Cls['ell'] * (Cls['ell']+1)/(2*np.pi) / ((core.G3Units.arcmin)**2), '.')
plt.xlim([0,2000])
```

```python
core.G3Units.microkelvin
```

```python
print(d[1])
```

## Detailed look at power spectrum estimation
In order to avoid confusion, here is a detailed explanation of each of the steps going from raw CMB map to power spectrum, as implemented in the 3G software pipeline. Let's assume that we are starting with map coadds, such as the ones that Daniel has produced here:

/spt/user/ddutcher/coadds/coadd2018_noW201wafCMpoly19mhpf300.g3

Given this map and an apodization mask, we can simply calculate the Cl's with: `spt3g.mapspectra.map_analysis.calculateCls`

This function does several things:
1. Removes weights from the T, Q, and U maps.
1. Runs `flatten_pol`. This converts the polarization angle definition from curved-space convention to flat-space convention.
1. Converts Q and U to E and B with `constructEB`.
1. Computes power spectrum with `av_ps`

In order to get a feel for what each of these steps does, let's copy the code for the power spectrum estimation here and go through it piece-by-piece.


Step 1 is a bunch of boilerplate and removing the weights from the maps. Recall from the maximum-likelihood solution of the mapmaking problem...

```python
map_frame['Wpol'][1000,1000]
```

```python
fr = map_frame
cross_map = None
apod_mask = apod
qu = False
flatten_pol=True
e_mode_method='basic'
b_mode_method='chi'
ell_bins = None
ell_min = 50
ell_max = 5000
delta_ell = 50
ell_weights_2d = None
kspace_filt = None
return_2d = False
t_only = False
realimag = 'real'
average_equivalent_cross_spectra = True

equivalent_ps = {'TE':'ET','EB':'BE','TB':'BT',
                     'TQ':'QT','QU':'UQ','TU':'UT'}

ps = dict()

if realimag != '':
    realimag = '.'+realimag

valid = mapmaker.mapmakerutils.ValidateMapFrames(fr)
res = fr['T'].res
if ell_bins is None:
    ells = np.arange(ell_min, ell_max+1, delta_ell)
    ell_bins = mapspectra.basicmaputils.get_reg_spaced_ell_bins(ells)
else:
    ells = np.mean(ell_bins, axis=1)

if apod_mask is None and 'apod_mask' in fr:
    apod_mask = fr['apod_mask']

if kspace_filt is None:
    kspace_filt = np.ones(np.shape(apod_mask))
if qu:
    qe_key, ub_key = 'Q', 'U'
else:
    qe_key, ub_key = 'E', 'B'

if not isinstance(ell_weights_2d, dict):
    tmp = dict()
    for k in ['T', qe_key, ub_key]:
        tmp[k] = ell_weights_2d
    ell_weights_2d = tmp


if fr['T'].is_weighted and fr['Q'].is_weighted and fr['U'].is_weighted:
    t, q, u = mapmaker.remove_weight(fr['T'], fr['Q'], fr['U'], fr['Wpol'])
elif not (fr['T'].is_weighted or fr['Q'].is_weighted or
          fr['U'].is_weighted):
    t, q, u = fr['T'], fr['Q'], fr['U']
else:
    raise ValueError('Map has combination of weighted/unweighted maps')

```

I went down the rabbit hole of trying to figure out how/why `flatten_pol` works, and... it turns out that it doesn't work. The motivation for `flatten_pol` is that Q and U are defined on the sphere in terms of the orthogonal coordinates RA and dec. In SPT, we make maps in flat space. In practice, this means that a.) we define a function $f: S^2 \rightarrow \mathbf{R}^2$ called a "map projection" which we apply to our pointing information before we estimate a map, and b.) we compute Fourier transforms in the cartesian basis $(x,y)$ instead of on the sphere. This is fine except that Q and U are defined with respect to polar angles on the sphere instead of with respect to the Cartesian axes. The `flatten_pol` operation attempts to undo the rotation introduced by the map projection, so that Q and U are once again defined with respect to the cartesian basis. There are 

The problem with this procedure is obvious: map projection does not correspond to a pure rotation of vectors in the tangent plane. Thus, *any* rotation that is applied to the polarization basis will not accomplish the desired result.

Another issue is that it isn't entirely clear how necessary `flatten_pol` is anyway. The function is intended to solve a problem that is very much like the bias in coordinates caused by map projection, but we don't attempt to correct for that.

Clearly we need a simulation to quantify the extent of the bias introduced by using/not using `flatten_pol`.

```python
if flatten_pol:
    q, u = mapspectra.basicmaputils.flatten_pol(q, u)
```

There are several different options for doing the E/B conversion within the `constructEB` function:
* `basic` : naive combination of Q/U
* `chiB` : the Smith/Zaldarriaga chi-B estimator (ref?)
* `smith` : the "original Smith B mode estimator" (ref?)

One of the methods above is described in astro-ph/0610059 (need to read this). For E-mode estimation, we use the `basic` estimator. The main issue with this estimator is that it results in substantial E/B leakage, but the B-mode power is small, so the bias to the E-mode power spectrum estimate is tolerable.

The `basic` estimator is defined as (flat space):
$$
E_\ell = Q_\ell \cos 2 \phi_\ell + U_\ell \sin 2 \phi_\ell
$$
where $\ell = (\ell_x, \ell_y)$ and $\phi_\ell = \arctan \ell_y / \ell_x$

```python
if qu:
    qe_ft = mapspectra.basicmaputils.map_to_ft(q, apod_mask, kspace_filt)
    ub_ft = mapspectra.basicmaputils.map_to_ft(u, apod_mask, kspace_filt)
else:
    qe_ft, ub_ft = constructEB(q,u, apod_mask = apod_mask,
                               e_mode_method = e_mode_method,
                               b_mode_method = b_mode_method,
                               kspace_filt = kspace_filt)
t_ft =   mapspectra.basicmaputils.map_to_ft(t, apod_mask, kspace_filt)  

map_fft={'T': t_ft, qe_key: qe_ft, ub_key: ub_ft}
```



```python
for pol, ft in map_fft.items():
        for xpol, xft in cross_map_fft.items():
            if return_2d:
                ps['ell'] = mapspectra.basicmaputils.make_ellgrid(
                    res, np.shape(ft))
                ps[pol+xpol] = ne.evaluate("(ft*xft)%s"%realimag)
            else:
                ps['ell'] = ells
                ps[pol+xpol] =  mapspectra.basicmaputils.av_ps(
                    ne.evaluate("(ft*xft)%s"%realimag), res, ell_bins,
                    ell_weights_2d[pol])
```
