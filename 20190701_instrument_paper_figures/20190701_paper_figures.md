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

# Figures for Instrument Paper

```python
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
from glob import glob
import pydfmux
import pickle
import pandas as pd
import os.path
from datetime import datetime
from spt3g.std_processing import obsid_to_g3time
import matplotlib
import matplotlib.dates as mdates
from matplotlib import gridspec
from scipy.optimize import bisect
```

## $1/f$ Noise Figure

```python
fr = list(core.G3File('gainmatching_noise_73124800.g3'))[1]
print(fr)
```

```python
def readout_noise(x, readout):
    return np.sqrt(readout)*np.ones(len(x))
def photon_noise(x, photon, tau):
    return np.sqrt(photon / (1 + 2*np.pi*((x*tau)**2)))
def atm_noise(x, A, alpha):
    return np.sqrt(A * (x)**(-1*alpha))
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout + (A * (x)**(-1*alpha)) + photon / (1 + 2*np.pi*((x*tau)**2)))
def horizon_model(x, readout, A, alpha):
    return np.sqrt(readout + (A * (x)**(-1*alpha)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout
def horizon_knee_func(x, readout, A, alpha):
    return (A * (x)**(-1*alpha)) - readout
```

```python
plt.figure(1, figsize=(10,4))

# subplot 1
ax1 = plt.subplot(1,2,1)

group = '150.0_w204'

ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
asd_diff = np.array(fr['AverageASDDiff'][group])
ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
asd_sum = np.array(fr['AverageASDSum'][group])

par_diff = fr["AverageASDDiffFitParams"][group]
par_sum = fr["AverageASDSumFitParams"][group]
plt.loglog(ff_diff, asd_diff, 'k', label='mean 150 GHz pair-difference')
plt.loglog(ff_sum, asd_sum, '0.6', label='mean 150 GHz pair-sum')
plt.loglog(ff_diff, noise_model(ff_diff, *list(par_diff)), 'C0--', label='noise model fit (see text)')

plt.xlabel('frequency [Hz]')
plt.ylabel('NET [$\mu$K$\sqrt{s}$]')
plt.legend(frameon=False)



# subplot 2
ax2 = plt.subplot(1,2,2)

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}
colors = {90.: 'C2', 150.: 'C3', 220.: 'C0'}
# colors = {90.: 'r', 150.: 'g', 220.: 'b'}

for jband, band in enumerate([90., 150., 220.]):
    A_sqrt = []
    alpha = []
    whitenoise = []
    noise_at_100mHz = []
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)

        par = fr["AverageASDDiffFitParams"][group]
        
        whitenoise.append(np.sqrt(par[0]))
        A_sqrt.append(np.sqrt(par[1]))
        alpha.append(par[2])
        
        noise_at_100mHz.append(atm_noise(0.1, par[1], par[2]))
        
    if band == 90.:
        bandname = 95.
    else:
        bandname = band
    _ = plt.hist(noise_at_100mHz, histtype='step', bins=np.linspace(0, 2000, 21),
                 linewidth=1.4, color=colors[band])
    _ = plt.hist(noise_at_100mHz, histtype='stepfilled', bins=np.linspace(0, 2000, 21),
                 label='{:.0f} GHz (median = {:.0f} $\mu$K$\sqrt{{s}}$)'\
                 .format(bandname, np.median(noise_at_100mHz)),
                 alpha=0.25, color=colors[band])
    plt.xlabel('residual atmospheric noise at 0.1 Hz\n($\sqrt{A (0.1 Hz)^{-\\alpha}}$) [$\mu$K$\sqrt{s}$]')
    plt.ylabel('wafer-averaged observations')
plt.legend(frameon=False)


plt.tight_layout()
plt.savefig('pair_differenced_noise.pdf')
```

```python
fig2 = plt.figure(2)

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}

for jband, band in enumerate([90., 150., 220.]):
    A_sqrt = []
    alpha = []
    whitenoise = []
    noise_at_100mHz = []
    for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                    'w181', 'w188', 'w203', 'w204', 'w206']):
        group = '{:.1f}_{}'.format(band, wafer)

        par = fr["AverageASDDiffFitParams"][group]
        
        whitenoise.append(np.sqrt(par[0]))
        A_sqrt.append(np.sqrt(par[1]))
        alpha.append(par[2])
        
        noise_at_100mHz.append(atm_noise(0.1, par[1], par[2]))
        
    _ = plt.hist(noise_at_100mHz, histtype='step', bins=np.linspace(0, 2000, 21),
                 label='{:.0f} GHz (median = {:.0f} $\mu$K$\sqrt{{s}}$)'.format(band, np.median(noise_at_100mHz)))
    plt.xlabel('residual atmospheric noise ($\sqrt{A (0.1 Hz)^{-\\alpha}}$) [$\mu$K$\sqrt{s}$]')
    plt.ylabel('wafer-averaged observations')
plt.legend()
```

```python
plt.figure(10)
ax1 = fig1.add_subplot(1,2,1)
ax1 = fig1.add_subplot(1,2,2)
```

```python
noise_fnames = glob('/spt/user/adama/20190329_gainmatching/downsampled/*.g3')
```

```python
noise_params = {}
for fname in noise_fnames:
    print(fname)
    fr = list(core.G3File(fname))[1]
    print(fname)
    
    for jband, band in enumerate([90., 150., 220.]):
        for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                        'w181', 'w188', 'w203', 'w204', 'w206']):
            group = '{:.1f}_{}'.format(band, wafer)
            noise_params[group] = fr["AverageASDDiffFitParams"][group]
```

## Noise Calculations
One table in the paper is a bunch of numbers related to noise. This section calculates some relevant numbers.

```python
from mapping_speed import tes_noise
```

```python
correlation = 1.0
phonon_gamma = 0.5

nu = {90:93.5e9, 150:146.9e9, 220:220.0e9}
delta_nu = {90:23.3e9, 150:30.7e9, 220:46.4e9}
P_optical = {90:(2.31e-12+2.67e-12), 150:(4.5e-12+3.28e-12), 220:(6.29e-12+3.7e-12)}
P_sat = {90:10e-12, 150:15e-12, 220:20e-12}
Tc = {90:0.450, 150:0.450, 220:0.450}
Tload = 4.0
Tbath = 0.315
Rsh = 0.03
Rn = 2.0
Rfrac = 0.8
R0 = Rfrac*Rn
loopgain = 10
```

```python
readout_johnson_noise_i = {90:10.4e-12, 150:13.0e-12, 220:16.0e-12}
shot_noise         = {band: tes_noise.shot_noise(nu=nu[band],
                                                 power=P_optical[band]) for band in nu}
correlation_noise  = {band: tes_noise.correlation_noise(nu=nu[band],
                                                        power=P_optical[band],
                                                        delta_nu=delta_nu[band],
                                                        correlation=correlation) for band in nu}
G                  = {band: tes_noise.G(Tc=Tc[band],
                                        Psat=P_sat[band],
                                        Popt=P_optical[band],
                                        Tbath=Tbath) for band in Tc}
v_bias             = {band: tes_noise.Vbias_rms(P_sat[band], P_optical[band], R0) for band in P_sat}
dIdP               = {band: tes_noise.dIdP(Vbias_rms=v_bias[band]) for band in v_bias}
phonon_noise       = {band: tes_noise.tes_phonon_noise_P(Tbolo=Tc[band],
                                                         G=G[band],
                                                         gamma=phonon_gamma) for band in Tc}
johnson_noise_p    = {band: tes_noise.tes_johnson_noise_P(f=0,
                                                          Tc=Tc[band],
                                                          Psat=P_sat[band],
                                                          L=loopgain,
                                                          Popt=P_optical[band]) for band in Tc}
johnson_noise_i    = {band: tes_noise.tes_johnson_noise_I(f=0,
                                                          Tc=Tc[band],
                                                          R0=R0,
                                                          L=loopgain) for band in Tc}
readout_johnson_noise_p = {band: readout_johnson_noise_i[band] / dIdP[band] for band in dIdP}
readout_noise_i    = {band: np.sqrt(readout_johnson_noise_i[band]**2 - johnson_noise_i[band]**2)
                      for band in Tc}
readout_noise_p    = {band: np.sqrt(readout_johnson_noise_p[band]**2 - johnson_noise_p[band]**2)
                      for band in Tc}
total_noise        = {band: np.sqrt(shot_noise[band]**2 + \
                                    correlation_noise[band]**2 + \
                                    phonon_noise[band]**2 + \
                                    readout_johnson_noise_p[band]**2) for band in shot_noise}
```

```python
for band in nu.keys():
    print('{} GHz:'.format(band))
    print('Vbias = {:.1f} uV'.format(v_bias[band]*1e6))
    print('Shot noise = {:.1f} aW/rtHz'.format(shot_noise[band]*1e18))
    print('Correlation noise = {:.1f} aW/rtHz'.format(correlation_noise[band]*1e18))
    print('Phonon noise = {:.1f} aW/rtHz'.format(phonon_noise[band]*1e18))
    print('Johnson noise (NEP) = {:.1f} aW/rtHz'.format(johnson_noise_p[band]*1e18))
    print('Johnson noise (NEI) = {:.1f} pA/rtHz'.format(johnson_noise_i[band]*1e12))
    print('readout + Johnson noise (NEP) = {:.1f} aW/rtHz'.format(readout_johnson_noise_p[band]*1e18))
    print('readout + Johnson noise (NEI) = {:.1f} pA/rtHz'.format(readout_johnson_noise_i[band]*1e12))
    print('readout noise (NEP) = {:.1f} aW/rtHz'.format(readout_noise_p[band]*1e18))
    print('readout noise (NEI) = {:.1f} pA/rtHz'.format(readout_noise_i[band]*1e12))
    print('total noise = {:.1f} aW/rtHz'.format(total_noise[band]*1e18))
    print()
```

## Yield
Let's add up some numbers from Donna's spreadsheet on the trac wiki:

https://spt-trac.grid.uchicago.edu/trac_south_pole/attachment/wiki/SummerJournal2018-2019/compare_all_wafers_to_last_year_19Nov2018.xlsx

```python
n_good_dets_by_wafer = {'w172':1437, 'w174':1433, 'w176':1480, 'w177':1530, 'w180':1405,
                        'w181':1367, 'w188':1241, 'w203':1456, 'w204':1409, 'w206':1408}
n_good_dets = np.sum([n_good_dets_by_wafer[wafer] for wafer in n_good_dets_by_wafer])
n_total_dets = 10*(66*24 - 6*2) # subtract off channels that go to alignment pixels
```

```python
hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm_pole = pydfmux.load_session(open(hwm_fname, 'r'))['hardware_map']
```

```python
bolos = hwm_pole.query(pydfmux.Bolometer)
freqs = [b.channel_map.lc_channel.frequency \
         for b in bolos \
         if b.channel_map.lc_channel.frequency > 1.e5]
n_resonances = len(freqs)
```

```python
tuned_fnames = []
for dirstub in ['201908']:
    fnames = glob('/big_scratch/pydfmux_output/{}*/'
                  '*drop_bolos_*/data/TOTAL_DATA.pkl'.format(dirstub))
    tuned_fnames += [fname for fname in fnames if 'tweak' not in fname]
```

```python
ntuned = {}
for fname in tuned_fnames:
    with open(fname, 'rb') as f:
        print(fname)
        ntuned[fname] = 0
        
        d = pickle.load(f)
        for mod in d.keys():
            if type(d[mod])==dict and 'results_summary' in d[mod].keys():
                ntuned[fname] += d[mod]['results_summary']['ntuned']
```

```python
n_tuned_median = np.median(list(ntuned.values()))
```

```python
# A huge number of detectors are removed because they don't show a clear transition
# in response to HR-10. These detectors are split between many different exclusion
# lists, so let's combine them all to remove duplicates.
exclude_path = '/home/adama/SPT/hardware_maps_southpole/2019/global/exclude'
exclude_fnames = ['warmVcold_exclude_hr10.csv', 'exclude_hr10_dkan_20181213.csv',
                  'exclude_suspecthr10_dkan_20181213.csv']
exclude_names = []
for fn in exclude_fnames:
    df = pd.read_csv(os.path.join(exclude_path, fn))
    exclude_names.extend(list(df['name']))
exclude_names = np.unique(exclude_names)
```

```python
print('Total number of detectors = {}'.format(n_total_dets))
print('Detectors with good warm pinout = {} ({:.1f}%)'\
      .format(n_good_dets, 100 * n_good_dets / n_total_dets))
print('Detectors with matched resonances = {} ({:.1f}%)'\
      .format(n_resonances, 100 * n_resonances / n_total_dets))
print('Detectors removed by HR-10 exclusion = {}'\
      .format(len(exclude_names)))
print('Tuned bolometers = {:.0f} ({:.1f}%)'\
      .format(n_tuned_median, 100 * n_tuned_median / n_total_dets))
```

Some other useful statistics:

* Number of full modules removed = 10 or 660 channels
 * These are a combination of noisy SQUIDs, and modules with shorts to ground that generated thermal heating on the wafer.


## Sensitivity Plots
There are a lot of possibilities for this section. Our goal should really just be to show off how well SPT-3G is working, so it's worth putting in some extra effort to prototype a bunch of potential figures.


### Livetime
One possibility is a livetime plot to show how high our observing efficiency is and how smoothly noise is integrating down at high-ell. Below is an updated mock-up of this from my quick check in March.

```python
tstart = datetime(year=2019, month=1, day=1).timestamp()
tstop = datetime(year=2019, month=9, day=30).timestamp()
```

```python
# find all observations with obsids that fall between start and stop
obsids_all = []
dirnames_all = []
for source in ['ra0hdec-44.75', 'ra0hdec-52.25',
               'ra0hdec-59.75', 'ra0hdec-67.25']:
    dirnames = np.array(glob('/spt/data/bolodata/downsampled/{}/*'.format(source)))
    obsids = np.array([int(dirname.split('/')[-1]) for dirname in dirnames])
    times = np.array([obsid_to_g3time(obsid).time/core.G3Units.second
                      for obsid in obsids])
    obsids_all = np.append(obsids_all, obsids[(times < tstop) & (times > tstart)])
    dirnames_all = np.append(dirnames_all, dirnames[(times < tstop) & (times > tstart)])

obs_tstart = []
obs_tstop = []
for dirname in dirnames_all:
    print(dirname)
    f = core.G3File('{}/0000.g3'.format(dirname))
    fr = f.next()
    try:
        obs_tstart.append(fr["ObservationStart"].time/core.G3Units.second)
        obs_tstop.append(fr["ObservationStop"].time/core.G3Units.second)
    except KeyError:
        pass
        
obs_tstart = np.sort(np.array(obs_tstart))
obs_tstop = np.sort(np.array(obs_tstop))
obs_tall = np.sort(np.append(obs_tstart, obs_tstop))
```

```python
epoch_tstart = mdates.epoch2num(obs_tstart)
tlive = obs_tstop - obs_tstart

# bin_times = np.array([mdates.date2num(datetime(year=2019, month=jmonth, day=1)) for jmonth in np.arange(1,10)])
# bin_times = np.append(bin_times, mdates.date2num(datetime(year=2019, month=9, day=10)))
month_times = np.array([mdates.date2num(datetime(year=2019, month=jmonth, day=1)) for jmonth in np.arange(1,10)])
month_times = np.append(month_times, mdates.date2num(datetime(year=2019, month=9, day=21)))
bin_times = np.linspace(mdates.date2num(datetime(year=2019, month=1, day=1)),
                        mdates.date2num(datetime(year=2019, month=9, day=21)), 30)

month_lengths = bin_times[1:] - bin_times[:-1]
jmonths = np.digitize(mdates.epoch2num(obs_tstart), bin_times) - 1
times = np.linspace(tstart, tstop, 50)
sunset = mdates.date2num(datetime(year=2019, month=3, day=21))
```

```python
len(tlive)
```

```python
matplotlib.rcParams.update({'font.size': 13})
times = np.linspace(tstart, tstop, 50)
dts = np.array([datetime.utcfromtimestamp(ts) for ts in times])
# livetime = np.interp(times, obs_tall, tlive)

fig = plt.figure(figsize=(8,5))
plt.ylabel('efficiency', color='C1')
# plt.ylim([0,1])
times = np.linspace(tstart, tstop, 50)
_ = plt.hist(mdates.epoch2num(obs_tstart),
             weights=tlive / (3600*24) / month_lengths[jmonths],
             bins=bin_times,
             color='C1', alpha=0.5)
plt.gca().tick_params('y', colors='C1')

xfmt = mdates.DateFormatter('%b %Y')
plt.gca().xaxis.set_major_formatter(xfmt)
plt.xticks(bin_times[:-1], rotation=45)

ax2 = plt.gca().twinx()
plt.plot(epoch_tstart,
         np.cumsum(tlive) / (3600*24),
         color='C0', linewidth=1.5, linestyle='--',
         label='all 2019')
plt.ylabel('cumulative time on CMB [d]', color='C0')
plt.gca().tick_params('y', colors='C0')

plt.plot(epoch_tstart[epoch_tstart>sunset],
         np.cumsum(tlive[epoch_tstart>sunset]) / (3600*24),
         color='C0', linewidth=1.5,
         label='since sunset')
plt.ylabel('cumulative time on CMB [d]', color='C0')
plt.gca().tick_params('y', colors='C0')
plt.legend()
plt.grid()

plt.title('2019 Observing Performance')
plt.tight_layout()
plt.savefig('livetime_progress.png', dpi=120)
```

### Noise vs. time

```python
fields = ['ra0hdec-44.75', 'ra0hdec-52.25', 'ra0hdec-59.75', 'ra0hdec-67.25']
band_labels = {90:'95', 150:'150', 220:'220'}

for band in [90, 150, 220]:
    noise_dict = {}
    obsids_dict = {}
    for field in fields:
        d = list(core.G3File('poledata/some_analysis_results_{}GHz.g3'.format(band)))[0]
        noise = np.array([d["NoiseLevelsFromCoaddedTMaps"][field][obsid] \
                          for obsid in d["NoiseLevelsFromCoaddedTMaps"][field].keys()])
        noise = noise / (core.G3Units.microkelvin * core.G3Units.arcmin)
        obsids = [int(obsid) for obsid in d["NoiseLevelsFromCoaddedTMaps"][field].keys()]
        noise_dict[field] = noise
        obsids_dict[field] = obsids
        
    max_min_obsid = np.max([np.min(obsids_dict[field]) for field in obsids_dict])
    min_max_obsid = np.min([np.max(obsids_dict[field]) for field in obsids_dict])
    obsids_interp = np.linspace(max_min_obsid, min_max_obsid, 500)
    noise_interp = np.sum(np.vstack([np.interp(obsids_interp, obsids_dict[field], noise_dict[field])
                           for field in fields]), axis=0) / 4
    
    plt.semilogy(obsids_interp, noise_interp, label='{} GHz'.format(band_labels[band]))
    
plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('cumulative map noise\n[$\mu$K arcmin]')
```

### Instantaneous noise plot


What Wei measures in his data quality plots is the noise in the uniform-coverage portion of the map. The portion of the map that receives uniform coverage is smaller than the portion of the map that the focal plane covers at a uniform velocity, which we include for mapmaking. Thus, if we assume that the sensitivity achieved in the uniform-coverage portion of the map is equal to the sensitivity achieved everywhere in the constant-velocity portion of the scans, we will necessarily overestimate the NET.

We could imagine correcting for this effect in several ways. If the entire constant-velocity portion of the scan corresponded exactly to the uniform-coverage region of the map, then the NET would be given by
\begin{equation}
\textrm{NET} = \frac{N\sqrt{T}}{\sqrt{A}},
\end{equation}
where $N$ is the noise level in $\mu \textrm{K}~\textrm{arcmin}$, $T$ is the total integration time, and $A$ is the map area. If we want to restrict ourselves to just the uniform-coverage region, then the calculation is the same, but we need to use the area and time that corresponds to the uniform-coverage region. The area that corresponds to the uniform region is easy to calculate based on whatever our threshold is (I assume 90% of max weight). The time is a little bit trickier because different detectors each are in the uniform region for slightly different times. To account for this, we can scale the total time $T$ (which is assumed to be entirely in the constant velocity region) by the ratio of the weights in the uniform-coverage region to the total weights. In other words, the NET using purely uniform-coverage estimators is given by
\begin{equation}
\textrm{NET} = \frac{N}{\sqrt{A_{\textrm{UC}}}} \times \sqrt{T \frac{\sum w^i_{\textrm{UC}}}{\sum w^i_{\textrm{total}}}}.
\end{equation}

To recap, we need the noise $N$ from Wei's cutouts, the constant-velocity time $T$, and the area of the uniform-coverage region $A_\textrm{UC}$, and the ratio of the uniform-coverage weights to the total weights.

```python
# Let's calculate the duration of a field scan excluding turnarounds
# (i.e the relevant length for noise estimates)
scan_time = 0
rawdata_path = '/spt/data/bolodata/downsampled/ra0hdec-44.75/84783979'
d = list(core.G3File(os.path.join(rawdata_path, '0000.g3')))
for fname in glob(os.path.join(rawdata_path, '0*.g3')):
    print(fname)
    d = list(core.G3File(os.path.join(fname)))
    for fr in d:
        if fr.type == core.G3FrameType.Scan and \
            'Turnaround' not in fr:
            scan_time += (fr['DetectorSampleTimes'][-1].time -
                          fr['DetectorSampleTimes'][0].time)
```

```python
scan_time
```

```python
# load the coadded maps to get the weights and area of the uniform coverage region
coadd_path = '/spt/user/weiquan/map_quality/hi_res_maps/yearly/2019/'
coadd_fnames = {band: {field: os.path.join(coadd_path, 'coadded_maps_from_{}_{}GHz.g3.gz'.format(field, band))
                       for field in fields} for band in [90, 150, 220]}

coadd_TT_weights = {}
weights_ratio = {}
area_uniform_coverage = {}

for band in coadd_fnames.keys():
    if band not in coadd_TT_weights.keys():
        coadd_TT_weights[band] = {}
        weights_ratio[band] = {}
        area_uniform_coverage[band] = {}
        
    for field in fields:
        print('Loading: {}'.format(coadd_fnames[band][field]))
        coadd_data = list(core.G3File(coadd_fnames[band][field]))

        # get weights
        coadd_TT_weights[band][field] = coadd_data[2]['Wunpol'].TT + coadd_data[3]['Wunpol'].TT

        # calculate apodization mask
        apod_mask_uniform = make_border_apodization(coadd_TT_weights[band][field],
                                                    apod_type='tophat',
                                                    weight_threshold=0.9)

        weights_ratio[band][field] = np.sum(coadd_TT_weights[band][field] * apod_mask_uniform) / \
                                np.sum(coadd_TT_weights[band][field])
        area_uniform_coverage[band][field] = np.sum(apod_mask_uniform) * (apod_mask_uniform.res)**2
```

```python
# load the analysis summary files to get the noise levels
noise_summary_path = '/spt/user/weiquan/map_quality/hi_res_maps/monthly/all_months'
noise_summary_fnames = {band: os.path.join(noise_summary_path, 'all_analysis_results_{}GHz.g3'.format(band))
                                          for band in [90, 150, 220]}

obsids_nets = {}
noises = {}
nets = {}
for band in noise_summary_fnames.keys():
    if band not in obsids_nets.keys():
        obsids_nets[band] = {}
        noises[band] = {}
        nets[band] = {}
        
    noise_summary = list(core.G3File(noise_summary_fnames[band]))
        
    for field in fields:

        obsids_nets[band][field] = np.array([k for k in noise_summary[0]["NoiseLevelsFromIndividualTMaps"]\
                                                                     [field].keys()])
        noises[band][field] = np.array([noise_summary[0]["NoiseLevelsFromIndividualTMaps"][field][k] \
                           for k in noise_summary[0]["NoiseLevelsFromIndividualTMaps"][field].keys()])
        nets[band][field] = noises[band][field] / np.sqrt(area_uniform_coverage[band][field]) * \
                        np.sqrt(scan_time * weights_ratio[band][field])
```

```python
plt.figure(figsize=(10,5))
color = {90:'C0', 150:'C1', 220:'C2'}
for band in nets.keys():
    for field in fields:
        plt.plot(obsids_nets[band][field], nets[band][field] / \
                 (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec)), '.',
                 marker='.', color=color[band])
plt.ylim([0, 40])
plt.xlabel('observation ID')
plt.ylabel('instantaneous array NET [$\mu$K$\sqrt{s}$]')
plt.tight_layout()
```

```python
for band in nets:
    print('{} GHz'.format(band))
    for field in nets[band].keys():
        median_net = np.median(nets[band][field])
        print('{} NET = {:.2f} uK-rtsec'.format(field, median_net / \
                                                          (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec))))
```

### Combined livetime and noise plot

```python
matplotlib.rcParams.update({'font.size': 13})
times = np.linspace(tstart, tstop, 50)
dts = np.array([datetime.utcfromtimestamp(ts) for ts in times])

fig = plt.figure(figsize=(12,9))

ax_top = plt.subplot(3,1,1)
plt.ylabel('livetime fraction\non CMB')
times = np.linspace(tstart, tstop, 50)
_ = plt.hist(mdates.epoch2num(obs_tstart),
             weights=tlive / (3600*24) / month_lengths[jmonths],
             bins=bin_times,
             color='C1', alpha=0.3)
_ = plt.hist(mdates.epoch2num(obs_tstart),
             weights=tlive / (3600*24) / month_lengths[jmonths],
             bins=bin_times,
             color='C1', histtype='step')
plt.ylim([0,0.79])
plt.grid()


plt.subplot(3,1,2,sharex=ax_top)
for band in nets.keys():
    for field in fields:
        times_nets = np.array([mdates.epoch2num(obsid_to_g3time(ob).time/core.G3Units.sec) \
                               for ob in obsids_nets[band][field]])
        net_threshold = 6*core.G3Units.microkelvin * np.sqrt(core.G3Units.sec)
        plt.plot(times_nets[nets[band][field] > net_threshold],
                 nets[band][field][nets[band][field] > net_threshold] / \
                 (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec)), '.',
                 markersize=2, color=color[band])
plt.ylim([0, 39.7])
plt.xlabel('observation ID')
plt.ylabel('NET [$\mu$K$\sqrt{s}$]')
plt.tight_layout()
plt.grid()


band_labels = {90:'95 GHz', 150:'150 GHz', 220:'220 GHz'}
plt.subplot(3,1,3,sharex=ax_top)
for band in [90, 150, 220]:
    noise_dict = {}
    obsids_dict = {}
    for field in fields:
        d = list(core.G3File('poledata/some_analysis_results_{}GHz.g3'.format(band)))[0]
        noise = np.array([d["NoiseLevelsFromCoaddedTMaps"][field][obsid] \
                          for obsid in d["NoiseLevelsFromCoaddedTMaps"][field].keys()])
        noise = noise / (core.G3Units.microkelvin * core.G3Units.arcmin)
        obsids = [int(obsid) for obsid in d["NoiseLevelsFromCoaddedTMaps"][field].keys()]
        noise_dict[field] = noise
        obsids_dict[field] = obsids
        
    max_min_obsid = np.max([np.min(obsids_dict[field]) for field in obsids_dict])
    min_max_obsid = np.min([np.max(obsids_dict[field]) for field in obsids_dict])
    obsids_interp = np.linspace(max_min_obsid, min_max_obsid, 500)
    times_interp = np.array([mdates.epoch2num(obsid_to_g3time(ob).time/core.G3Units.sec) for ob in obsids_interp])
    noise_interp = np.sum(np.vstack([np.interp(obsids_interp, obsids_dict[field], noise_dict[field])
                           for field in fields]), axis=0) / 4
    
    plt.semilogy(times_interp, noise_interp, label=band_labels[band])


plt.legend()
plt.grid()
plt.ylabel('cumulative map\ndepth [$\mu$K arcmin]')

plt.tight_layout()
fig.subplots_adjust(hspace=0)

plt.savefig('livetime_noise_summary.pdf')
```

```python
livetime_eff = np.sum(tlive[epoch_tstart>sunset]) / (24*3600) / \
                (np.max(epoch_tstart[epoch_tstart>sunset]) - \
                 np.min(epoch_tstart[epoch_tstart>sunset]))
print('Livetime efficiency = {}'.format(livetime_eff))
```

### $1/f$ Noise in Noise Stares
The $1/f$ noise figure at the top of this note was kind of stupid. Let's make a simpler one that includes data from all the 2019 data so far, and just as the average sum and difference power spectra for each band. This is analogous to Figure 22 of 1403.4302.

```python
fnames_noise = glob('/spt/user/adama/20190911_noise_gainmatch_cal/fullrate/gainmatching_noise_*.g3')
```

```python
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout

f_knee_dict = {}
for fname in fnames_noise:
    print(fname)
    fr = list(core.G3File(fname))[1]
    
    for jband, band in enumerate([90., 150., 220.]):
        for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                        'w181', 'w188', 'w203', 'w204', 'w206']):
            group = '{:.1f}_{}'.format(band, wafer)
            
            if band not in f_knee_dict:
                f_knee_dict[band] = {}
            if wafer not in f_knee_dict[band]:
                f_knee_dict[band][wafer] = []
                
            try:
                ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
                asd_diff = np.array(fr['AverageASDDiff'][group])
                par_diff = fr["AverageASDDiffFitParams"][group]

                f_knee = bisect(knee_func, a=0.01, b=10.0, args=tuple(par_diff))
                f_knee_dict[band][wafer].append(f_knee)
            except:
                pass
```

```python
fnames = glob('/spt/user/adama/20190911_noise_gainmatch_cal/fullrate/gainmatching_noise_83521028.g3')
d = list(core.G3File(fnames))[1]

f_hi = 73
freq = np.array(d["AverageASDDiff"]['frequency']) / core.G3Units.Hz

fig = plt.figure(figsize=(12,4))
gridspec.GridSpec(4,3)


for jband, band in enumerate(f_knee_dict):
    plt.subplot2grid((4,3), (0,jband), colspan=1, rowspan=1)
    f_knees = np.hstack([f_knee_dict[band][wafer] for wafer in f_knee_dict[band].keys()])
    plt.hist(f_knees, bins=np.logspace(np.log10(1e-2), np.log10(73), 21),
             alpha=0.5, color='C0') #, normed=True)
    plt.hist(f_knees, bins=np.logspace(np.log10(1e-2), np.log10(73), 21),
             histtype='step', color='C0') #, normed=True)
#     plt.axis([1e-2, 73, 0, 11])
    plt.axis([1e-2, 73, 0, 220])
    plt.gca().set_xscale('log')
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    if jband == 0:
        plt.ylabel('$1/f$ knees of\nnoise stares')

# ax = plt.subplot(2,3,4)
plt.subplot2grid((4,3), (1,0), colspan=1, rowspan=3)
asd_diff = np.array(d["AverageASDDiff"]['90.0_w204'])
asd_sum = np.array(d["AverageASDSum"]['90.0_w204'])
plt.loglog(freq[freq<f_hi], asd_diff[freq<f_hi] / np.sqrt(2.))
plt.loglog(freq[freq<f_hi], asd_sum[freq<f_hi] / np.sqrt(2.))
print(np.mean(asd_diff[(freq>1) & (freq<5)]) / np.sqrt(2.))
plt.grid()
plt.axis([1e-2, 73, 200, 50000])
plt.ylabel('NET [$\mu$K $\sqrt{s}$]')

# plt.subplot(2,3,5)
plt.subplot2grid((4,3), (1,1), colspan=1, rowspan=3)
asd_diff = np.array(d["AverageASDDiff"]['150.0_w204'])
asd_sum = np.array(d["AverageASDSum"]['150.0_w204'])
plt.loglog(freq[freq<f_hi], asd_diff[freq<f_hi] / np.sqrt(2.))
plt.loglog(freq[freq<f_hi], asd_sum[freq<f_hi] / np.sqrt(2.))
print(np.mean(asd_diff[(freq>1) & (freq<5)]) / np.sqrt(2.))
plt.grid()
plt.axis([1e-2, 73, 200, 50000])
plt.gca().set_yticklabels([])
plt.xlabel('frequency [Hz]')

# plt.subplot(2,3,6)
plt.subplot2grid((4,3), (1,2), colspan=1, rowspan=3)
asd_diff = np.array(d["AverageASDDiff"]['220.0_w204'])
asd_sum = np.array(d["AverageASDSum"]['220.0_w204'])
plt.loglog(freq[freq<f_hi], asd_diff[freq<f_hi] / np.sqrt(2.))
plt.loglog(freq[freq<f_hi], asd_sum[freq<f_hi] / np.sqrt(2.))
print(np.mean(asd_diff[(freq>1) & (freq<5)]) / np.sqrt(2.))
plt.grid()
plt.axis([1e-2, 73, 200, 50000])
plt.gca().set_yticklabels([])

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig('lowf_noise_summary.pdf')
```

```python
def readout_noise(x, readout):
    return np.sqrt(readout)*np.ones(len(x))
def photon_noise(x, photon, tau):
    return np.sqrt(photon / (1 + 2*np.pi*((x*tau)**2)))
def atm_noise(x, A, alpha):
    return np.sqrt(A * (x)**(-1*alpha))
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout + (A * (x)**(-1*alpha)) + photon / (1 + 2*np.pi*((x*tau)**2)))
def horizon_model(x, readout, A, alpha):
    return np.sqrt(readout + (A * (x)**(-1*alpha)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout
```

```python

```
