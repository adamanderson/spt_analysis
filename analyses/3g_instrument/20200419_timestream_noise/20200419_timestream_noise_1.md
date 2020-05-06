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

# Timestream Noise
**Original date:** 19 April 2020  
**Name:** Adam Anderson

This note fits some timestream noise for the instrument paper.

```python
from spt3g import core, calibration
from spt3g.calibration.template_groups import get_template_groups
from spt3g.dfmux import HousekeepingForBolo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
```

```python
wafers = ['w172', 'w174', 'w176', 'w177', 'w180',
          'w181', 'w188', 'w203', 'w204', 'w206']
bands = [90, 150, 220]
```

```python
horizon_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_77863968_horizon_default_current.g3'
horizon_data = list(core.G3File(horizon_fname))

normal_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_81433244_default_current.g3'
normal_data = list(core.G3File(normal_fname))

bolo_tgroups = get_template_groups(horizon_data[0]["BolometerProperties"], 
                                            per_band = True,
                                            per_wafer = True,
                                            include_keys = True)
```

```python
print(horizon_data[1])
```

```python
def readout_noise(x, readout):
    return readout*np.ones(len(x))
def photon_noise(x, photon, tau):
    return photon / np.sqrt(1 + 2*np.pi*(x*tau))
def atm_noise(x, A, alpha):
    return A * (x)**(-1*alpha)
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha)) + photon**2 / (1 + 2*np.pi*(x*tau)))
def full_readout_model(x, readout, A, alpha):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha)))
```

```python
avg_asd_horizon = {}
avg_asd_normal = {}

for band in bands:
    plt.figure(figsize=(10,20))
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        nbolos = 0
        for bolo in group:
            if bolo in horizon_data[1]['ASD'] and \
               np.all(np.isfinite(horizon_data[1]['ASD'][bolo])):
                if groupname not in avg_asd_horizon:
                    avg_asd_horizon[groupname] = horizon_data[1]['ASD'][bolo]
                else:
                    avg_asd_horizon[groupname] += horizon_data[1]['ASD'][bolo]
            nbolos += 1
        avg_asd_horizon[groupname] /= nbolos

        nbolos = 0
        for bolo in group:
            if bolo in normal_data[1]['ASD'] and \
               np.all(np.isfinite(normal_data[1]['ASD'][bolo])):
                if groupname not in avg_asd_normal:
                    avg_asd_normal[groupname] = normal_data[1]['ASD'][bolo]
                else:
                    avg_asd_normal[groupname] += normal_data[1]['ASD'][bolo]
            nbolos += 1
        avg_asd_normal[groupname] /= nbolos

        plt.subplot(5,2,jwafer+1)
        plt.loglog(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz,
                   avg_asd_horizon[groupname])
        plt.loglog(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz,
                   avg_asd_normal[groupname])
        plt.axis([0.01, 75, 5, 500])
        plt.title('{} GHz: {}'.format(band, wafer))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('current noise [pA/rtHz]')
    plt.tight_layout()
#     plt.savefig('horizon_vs_intransition_{}.png'.format(band))
```

```python
groupname = '90.0_w172'

f_min = 0.01
f_max = 60

f_horizon = np.array(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz)
asd_horizon = np.array(avg_asd_horizon[groupname])
f_normal = np.array(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz)
asd_normal = np.array(avg_asd_normal[groupname])
    
par_horizon, cov = curve_fit(full_readout_model,
                             f_horizon[(f_horizon>f_min) & (f_horizon<f_max)],
                             asd_horizon[(f_horizon>f_min) & (f_horizon<f_max)],
                             bounds=([0, 0, 0],
                                     [np.inf, np.inf, np.inf]),
                     p0=[10, 10, 1])

# def noise_model_fixed(x, A, alpha, photon, tau):
#     return noise_model(x, par_horizon[0], A, alpha, photon, tau)
    
# par_normal, cov = curve_fit(noise_model_fixed,
#                             f_normal[(f_normal>f_min) & (f_normal<f_max)],
#                             asd_normal[(f_normal>f_min) & (f_normal<f_max)],
#                             bounds=([0, 0, 0, 0],
#                                     [np.inf, np.inf, np.inf, np.inf]),
#                             p0=[10, 1, 10, 0.01])

    
par_normal, cov = curve_fit(noise_model,
                            f_normal[(f_normal>f_min) & (f_normal<f_max)],
                            asd_normal[(f_normal>f_min) & (f_normal<f_max)],
                            bounds=([0, 0, 0, 0, 0],
                                    [np.inf, np.inf, np.inf, np.inf, np.inf]),
                            p0=[10, 10, 1, 10, 0.01])
```

```python
f_min = 0.01
f_max = 60

for band in bands:
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        
        f_horizon = np.array(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz)
        asd_horizon = np.array(avg_asd_horizon[groupname])
        f_normal = np.array(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz)
        asd_normal = np.array(avg_asd_normal[groupname])
        

        
#         par_normal, cov = curve_fit(noise_model,
#                             f_normal[(f_normal>f_min) & (f_normal<f_max)],
#                             asd_normal[(f_normal>f_min) & (f_normal<f_max)],
#                             bounds=([0, 0, 0, 0, 0],
#                                     [np.inf, np.inf, np.inf, np.inf, np.inf]),
#                             p0=[10, 10, 1, 10, 0.01])
        errs = np.diagonal(np.sqrt(cov))
        
        plt.figure(figsize=(8,5))
        plt.plot(f_horizon, avg_asd_horizon[groupname],
                 label='horizon noise stare')
        plt.plot(f_normal, avg_asd_normal[groupname],
                 label='in-transition noise stare')
        plt.plot(f_normal, noise_model(f_normal, *par_normal), 'k',
                 label='total noise model')
        plt.plot(f_normal, photon_noise(f_normal, par_normal[3], par_normal[4]), '--',
                 label='photon + phonon noise: $\\tau = ${:.1f} $\pm$ {:.1f} msec, '.format(par_normal[3]*1e3, errs[3]*1e3) + 
                       '$\sqrt{{ N_\gamma^2 + N_{{ph}}^2 }} = ${:.1f} $\pm$ {:.1f} pA / $\sqrt{{Hz}}$'.format(par_normal[2], errs[2]))
        plt.plot(f_normal, readout_noise(f_normal, par_normal[0]), '--C3',
                 label='readout noise: $N_{{ro}} = ${:.1f} pA / $\sqrt{{Hz}}$'.format(par_horizon[0]))
        
        
        plt.axis([0.01, 75, 5, 200])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('current noise [pA/rtHz]')
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.legend()
        plt.title('{}: {:.0f} GHz'.format(wafer, band))
        plt.tight_layout()
        plt.savefig('figures_fixed_ro/noisefits_{}_{}_fixed_ro.png'.format(wafer, band))
        plt.close()
```

## Constrained fits
### Constrained readout noise

```python
f_min = 0.01
f_max = 60

for band in bands:
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        
        f_horizon = np.array(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz)
        asd_horizon = np.array(avg_asd_horizon[groupname])
        f_normal = np.array(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz)
        asd_normal = np.array(avg_asd_normal[groupname])
        
        par_horizon, cov = curve_fit(full_readout_model,
                                     f_horizon[(f_horizon>f_min) & (f_horizon<f_max)],
                                     asd_horizon[(f_horizon>f_min) & (f_horizon<f_max)],
                                     bounds=([0, 0, 0],
                                             [np.inf, np.inf, np.inf]),
                             p0=[10, 10, 1])

        def noise_model_fixed(x, A, alpha, photon, tau):
            return noise_model(x, par_horizon[0], A, alpha, photon, tau)

        par_normal, cov = curve_fit(noise_model_fixed,
                                    f_normal[(f_normal>f_min) & (f_normal<f_max)],
                                    asd_normal[(f_normal>f_min) & (f_normal<f_max)],
                                    bounds=([0, 0, 0, 0],
                                            [np.inf, np.inf, np.inf, np.inf]),
                                    p0=[10, 1, 10, 0.01])
        
#         par_normal, cov = curve_fit(noise_model,
#                             f_normal[(f_normal>f_min) & (f_normal<f_max)],
#                             asd_normal[(f_normal>f_min) & (f_normal<f_max)],
#                             bounds=([0, 0, 0, 0, 0],
#                                     [np.inf, np.inf, np.inf, np.inf, np.inf]),
#                             p0=[10, 10, 1, 10, 0.01])
        errs = np.diagonal(np.sqrt(cov))
        
        plt.figure(figsize=(8,5))
        plt.plot(f_horizon, avg_asd_horizon[groupname],
                 label='horizon noise stare')
        plt.plot(f_normal, avg_asd_normal[groupname],
                 label='in-transition noise stare')
        plt.plot(f_normal, noise_model_fixed(f_normal, *par_normal), 'k',
                 label='total noise model')
        plt.plot(f_normal, photon_noise(f_normal, par_normal[2], par_normal[3]), '--',
                 label='photon + phonon noise: $\\tau = ${:.1f} $\pm$ {:.1f} msec, '.format(par_normal[3]*1e3, errs[3]*1e3) + 
                       '$\sqrt{{ N_\gamma^2 + N_{{ph}}^2 }} = ${:.1f} $\pm$ {:.1f} pA / $\sqrt{{Hz}}$'.format(par_normal[2], errs[2]))
        plt.plot(f_normal, readout_noise(f_normal, par_horizon[0]), '--C3',
                 label='readout noise: $N_{{ro}} = ${:.1f} pA / $\sqrt{{Hz}}$'.format(par_horizon[0]))
        
        
        plt.axis([0.01, 75, 5, 200])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('current noise [pA/rtHz]')
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.legend()
        plt.title('{}: {:.0f} GHz'.format(wafer, band))
        plt.tight_layout()
        plt.savefig('figures_fixed_ro/noisefits_{}_{}_fixed_ro.png'.format(wafer, band))
        plt.close()
```

## Power conversion using "correct" transfer function
Joshua Montgomery has supplied a more "correct" transfer function that accounts for some (all?) of the parasitic impedances in the system. We can use his conversion to get the noises into power units.


### Inspecting Joshua's TF

```python
with open('carrier_and_nuller_tf_adam.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
```

```python
tf_dict
```

```python
plt.plot(tf_dict['bias_freq'], tf_dict['tf_nuller'], '.', markersize=2)
```

```python
plt.plot(tf_dict['bias_freq'], tf_dict['tf_carrier'], '.', markersize=2)
plt.xlabel('bias frequency [Hz]')
plt.ylabel('tf_carrier')
plt.tight_layout()
plt.savefig('tf_carrier.png', dpi=150)
```

The primary reason we need the new transfer function calibration is in order to perform the power calibration. This implies that it should eliminate any frequency-dependence that exists in the Psat distribution. We can use this as a straightforward test of the validity of the correction. Note that this new transfer function replaces the pydfmux TF, so we should take care to apply it **before** the pydfmux one, and on normalized amplitudes.

To do the test, let's apply the TF on top of Psats that we measure from drop_bolos at pole.

```python
dropbolos_path = '/big_scratch/pydfmux_output/20200419/20200419_222800_drop_bolos_586B/data/TOTAL_DATA.pkl'
with open(dropbolos_path, 'rb') as f:
    dropbolos_data = pickle.load(f)
```

```python
carrier_amp = {}
nuller_amp = {}
current = {}
voltage = {}
bias_freq = {}
wafer_name = {}
observing_band = {}
for rmod in dropbolos_data:
    if 'post_drop' in dropbolos_data[rmod]:
        for ch in dropbolos_data[rmod]['post_drop']:
            boloname = dropbolos_data[rmod]['subtargets'][ch]['bolometer']
            carrier_amp[boloname] = dropbolos_data[rmod]['post_drop'][ch]['Cmag']
            nuller_amp[boloname] = (dropbolos_data[rmod]['post_drop'][ch]['Nmag']/2**23)*1e-3
            bias_freq[boloname] = dropbolos_data[rmod]['subtargets'][ch]['frequency']
            current[boloname] = dropbolos_data[rmod]['post_drop'][ch]['I']
            voltage[boloname] = dropbolos_data[rmod]['post_drop'][ch]['V']
            wafer_name[boloname] = dropbolos_data[rmod]['subtargets'][ch]['physical_name'].split('/')[0]
            observing_band[boloname] = dropbolos_data[rmod]['subtargets'][ch]['observing_band']
            
tf_carrier = {}
tf_nuller = {}
voltage_corr = {}
current_corr = {}
power_corr = {}
for bolo in tf_dict['name']:
    if bolo in carrier_amp:
        tf_carrier[bolo] = tf_dict['tf_carrier_meth3'][tf_dict['name'] == bolo][0]
        tf_nuller[bolo] = tf_dict['dan_nuller_corr'][tf_dict['name'] == bolo][0]
        voltage_corr[bolo] = carrier_amp[bolo] * 0.01 * tf_carrier[bolo]
        current_corr[bolo] = current[boloname] * tf_nuller[bolo]
        power_corr[bolo] = voltage_corr[bolo] * current_corr[bolo]
```

```python
plt.plot([bias_freq[bolo] for bolo in power_corr],
         [power_corr[bolo]/2*1e12 for bolo in power_corr], '.', markersize=2)
```

```python
plt.figure(figsize=(8,30))
for jwafer, wafer in enumerate(wafers[:7]):
    plt.subplot(7,1,jwafer+1)
    for band in bands:
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        plt.plot([bias_freq[bolo] for bolo in power_corr if bolo in group],
                 [power_corr[bolo]/2*1e12 \
                  for bolo in power_corr if bolo in group], '.', markersize=2)
    plt.ylim([0,10])
    plt.title(wafer)
    plt.ylabel('power after drop_bolos [pW]')
    plt.xlabel('bias frequency [Hz]')
    plt.tight_layout()
plt.savefig('psat_v_freq.png', dpi=150)
```

```python
plt.figure(figsize=(8,30))
for jwafer, wafer in enumerate(wafers[:7]):
    plt.subplot(7,1,jwafer+1)
    for band in bands:
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        plt.plot([bias_freq[bolo] for bolo in voltage if bolo in group],
                 [voltage[bolo]*current[bolo]/2*1e12 \
                  for bolo in voltage if bolo in group], '.', markersize=2)
    plt.ylim([0,12])
    plt.title(wafer)
    plt.ylabel('power after drop_bolos [pW]')
    plt.xlabel('bias frequency [Hz]')
    plt.tight_layout()
plt.savefig('psat_v_freq_uncorrected.png', dpi=150)
```

```python
plt.figure(figsize=(8,30))
for jwafer, wafer in enumerate(wafers[:7]):
    plt.subplot(7,1,jwafer+1)
    for band in bands:
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        plt.plot([bias_freq[bolo] for bolo in corr_carrier_amp if bolo in group],
                 [voltage[bolo]/current[bolo] \
                  for bolo in corr_carrier_amp\
                  if bolo in group], '.', markersize=2)
    plt.ylim([0,3])
    plt.title(wafer)
    plt.ylabel('resistance [Ohm]')
    plt.xlabel('bias frequency [Hz]')
    plt.tight_layout()
#     plt.savefig('psat_v_freq_uncorrected.png', dpi=150)
```

```python
plt.figure(figsize=(8,30))
for jwafer, wafer in enumerate(wafers[:7]):
    plt.subplot(7,1,jwafer+1)
    for band in bands:
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        plt.plot([bias_freq[bolo] for bolo in corr_carrier_amp if bolo in group],
                 [corr_carrier_amp[bolo]/current[bolo] \
                  for bolo in corr_carrier_amp\
                  if bolo in group], '.', markersize=2)
    plt.ylim([0,3])
    plt.title(wafer)
    plt.ylabel('resistance [Ohm]')
    plt.xlabel('bias frequency [Hz]')
    plt.tight_layout()
```

```python
plt.figure(figsize=(8,30))
for jwafer, wafer in enumerate(wafers[:7]):
    plt.subplot(7,1,jwafer+1)
    for band in bands:
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        plt.plot([bias_freq[bolo] for bolo in corr_carrier_amp if bolo in group],
                 [corr_carrier_amp[bolo]*nuller_amp[bolo]/2*1e12 \
                  for bolo in corr_carrier_amp\
                  if bolo in group], '.', markersize=2)
    plt.ylim([0,50])
    plt.title(wafer)
    plt.ylabel('power after drop_bolos [pW]')
    plt.xlabel('bias frequency [Hz]')
    plt.tight_layout()
    plt.savefig('psat_v_freq_nmag.png', dpi=150)
```

### Applying the correction

```python
# load raw data in order to get the housekeeping and voltage bias information
rawdata = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/0000.g3'))
```

```python
# compute the bias voltage and dI/dP
hkmap = rawdata[3]["DfMuxHousekeeping"]
wiringmap = rawdata[2]["WiringMap"]

amp_carrier_raw = {}
v_comb = {}
didp = {}
bolonames = [bolo for bolo in horizon_data[1]['ASD'].keys() \
             if bolo != 'frequency']
for bolo in bolonames:
    if bolo in tf_dict['name']:
        chanhk = HousekeepingForBolo(hkmap, wiringmap, bolo, False)
        amp_carrier_raw[bolo] = chanhk.carrier_amplitude
        v_comb[bolo] = amp_carrier_raw[bolo] * 0.01 * tf_dict['tf_carrier'][tf_dict['name'] == bolo][0]
        didp[bolo] = 2./v_comb[bolo]
        
avg_didp = {}
for band in bands:
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        avg_didp[groupname] = 2./np.mean([v_comb[bolo] for bolo in group \
                                          if bolo in v_comb])
```

```python
# inspecting the dI/dP
for jwafer, wafer in enumerate(wafers):
    plt.figure()
    for band in bands:
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        didp_plot = np.array([didp[bolo] for bolo in group \
                              if bolo in v_comb])
        plt.hist(didp_plot*1e-6, bins=np.linspace(0,1,51),
                 histtype='step', label='{} GHz'.format(band))
    plt.title(wafer)
    plt.legend()
```

```python
f_min = 0.01
f_max = 60

for band in bands:
    for jwafer, wafer in enumerate(wafers[3:4]):
        groupname = '{:.1f}_{}'.format(band, wafer)
        
        f_horizon = np.array(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz)
        asd_horizon = np.array(avg_asd_horizon[groupname])
        f_normal = np.array(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz)
        asd_normal = np.array(avg_asd_normal[groupname])
        
        plt.figure(figsize=(8,5))
        plt.plot(f_horizon, avg_asd_horizon[groupname]*1e-12 / avg_didp[groupname] / 1e-18,
                 label='horizon noise stare')
        plt.plot(f_normal, avg_asd_normal[groupname]*1e-12 / avg_didp[groupname] / 1e-18,
                 label='in-transition noise stare')
     
        plt.axis([0.01, 75, 9, 400])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('current noise [pA/rtHz]')
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.legend()
        plt.title('{}: {:.0f} GHz'.format(wafer, band))
        plt.grid()
        plt.tight_layout()
#         plt.savefig('figures_nep/noisefits_{}_{}.png'.format(wafer, band))
#         plt.close()
```

## Thinking more broadly about the TF

```python
dropbolos_path = '/big_scratch/pydfmux_output/20200419/20200419_222800_drop_bolos_586B/data/TOTAL_DATA.pkl'
with open(dropbolos_path, 'rb') as f:
    dropbolos_data = pickle.load(f)
    
with open('carrier_tf_adam.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
```

```python
carrier_amp = {}
nuller_amp = {}
current = {}
voltage = {}
bias_freq = {}
wafer_name = {}
observing_band = {}
for rmod in dropbolos_data:
    if 'post_drop' in dropbolos_data[rmod]:
        for ch in dropbolos_data[rmod]['post_drop']:
            boloname = dropbolos_data[rmod]['subtargets'][ch]['bolometer']
            carrier_amp[boloname] = dropbolos_data[rmod]['post_drop'][ch]['Cmag']
            nuller_amp[boloname] = (dropbolos_data[rmod]['post_drop'][ch]['Nmag']/2**23)*1e-3
            bias_freq[boloname] = dropbolos_data[rmod]['subtargets'][ch]['frequency']
            current[boloname] = dropbolos_data[rmod]['post_drop'][ch]['I']
            voltage[boloname] = dropbolos_data[rmod]['post_drop'][ch]['V']
            wafer_name[boloname] = dropbolos_data[rmod]['subtargets'][ch]['physical_name'].split('/')[0]
            observing_band[boloname] = dropbolos_data[rmod]['subtargets'][ch]['observing_band']
            
corr_carrier_amp = {}
new_tf = {}
for bolo in tf_dict['name']:
    if bolo in carrier_amp:
        new_tf[bolo] = tf_dict['tf_carrier'][tf_dict['name'] == bolo]
        corr_carrier_amp[bolo] = carrier_amp[bolo] * 0.01 * new_tf[bolo]
```

```python
plt.figure(figsize=(10,15))
for jwafer, wafer in enumerate(wafers):
    plt.subplot(5,2,jwafer+1)
    plt.plot([voltage[bolo] for bolo in voltage
              if wafer_name[bolo] == wafer and observing_band[bolo] == 220],
             [current[bolo] for bolo in current
              if wafer_name[bolo] == wafer and observing_band[bolo] == 220], '.', markersize=2)
    plt.axis([1e-6, 1e-5, 1e-6, 5e-6])
    plt.xlabel('voltage')
    plt.ylabel('current')
    plt.title('{}: 220 GHz using pydfmux TF'.format(wafer))
plt.tight_layout()
plt.savefig('IV_220.png', dpi=150)
```

```python
plt.figure(figsize=(10,15))
for jwafer, wafer in enumerate(wafers):
    plt.subplot(5,2,jwafer+1)
    plt.plot([corr_carrier_amp[bolo] for bolo in corr_carrier_amp
              if wafer_name[bolo] == wafer and observing_band[bolo] == 220],
             [current[bolo] for bolo in corr_carrier_amp
              if wafer_name[bolo] == wafer and observing_band[bolo] == 220], '.', markersize=2)
    plt.axis([1e-6, 1e-5, 1e-6, 5e-6])
    plt.xlabel('voltage')
    plt.ylabel('current')
    plt.title('{}: 220 GHz using pydfmux TF'.format(wafer))
plt.tight_layout()
plt.savefig('IV_220_new_tf.png', dpi=150)
```

## Using my LTD TF

```python

```
