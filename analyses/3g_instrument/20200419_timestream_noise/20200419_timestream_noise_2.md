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
**Original date:** 24 April 2020  
**Name:** Adam Anderson

This note fits some timestream noise for the instrument paper. My previous notebook on this topic became such a mess due to dead ends, that I decided to start over. Hence the `_2` suffix.

```python
from spt3g import core, calibration
from spt3g.calibration.template_groups import get_template_groups
from spt3g.dfmux import HousekeepingForBolo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from spt3g.dfmux.IceboardConversions import convert_TF
from mapping_speed import tes_noise
from scipy.constants import h, k
```

```python
wafers = ['w172', 'w174', 'w176', 'w177', 'w180',
          'w181', 'w188', 'w203', 'w204', 'w206']
bands = [90, 150, 220]

G = {'w172': {90: 99e-12,  150: 112e-12, 220: 112e-12},
     'w174': {90: 108e-12, 150: 151e-12, 220: 139e-12},
     'w176': {90: 102e-12, 150: 120e-12, 220: 120e-12},
     'w177': {90: 100e-12, 150: 116e-12, 220: 115e-12},
     'w181': {90: 111e-12, 150: 124e-12, 220: 122e-12},
     'w188': {90: 90e-12,  150: 109e-12, 220: 104e-12},
     'w204': {90: 103e-12, 150: 136e-12, 220: 157e-12},
     'w206': {90: 85e-12,  150: 113e-12, 220: 128e-12}} # all in W/K
Tc = {'w172': 0.423,
     'w174': 0.414,
     'w176': 0.493,
     'w177': 0.487,
     'w181': 0.469,
     'w188': 0.459,
     'w204': 0.432,
     'w206': 0.444} # all in K
band_centers = {90:93.5e9, 150:146.7e9, 220:219.9e9}
bandwidths = {90:23.2e9, 150:30.7e9, 220:46.4e9}
Popt = {90:4.6e-12, 150:7.7e-12, 220:8.9e-12}
optical_eff = {90:0.135, 150:0.192, 220:0.126}
```

```python
horizon_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_77863968_horizon_default_power.g3'
horizon_data = list(core.G3File(horizon_fname))

normal_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_81433244_default_power.g3'
normal_data = list(core.G3File(normal_fname))

bolo_tgroups = get_template_groups(horizon_data[0]["BolometerProperties"], 
                                            per_band = True,
                                            per_wafer = True,
                                            include_keys = True)
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
def noise_model_full(x, readout, A, alpha, photon, phonon, tau):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha)) + (photon**2 + phonon**2) / (1 + 2*np.pi*(x*tau)))
def full_readout_model(x, readout, A, alpha):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha)))

def dPdT(nu, delta_nu, eff):
    x = h*nu / (k * 2.73)
    dPdT = 2*k * eff * delta_nu * x**2 * np.exp(x) / (np.exp(x) - 1)**2
    return dPdT
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
        plt.axis([0.01, 75, 20, 500])
        plt.title('{} GHz: {}'.format(band, wafer))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('NEP [aW/rtHz]')
    plt.tight_layout()
#     plt.savefig('horizon_vs_intransition_{}.png'.format(band))
```

```python
f_min = 0.01
f_max = 60

NEP_phph = {}

for band in bands:
    NEP_phph[band] = {}
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
        if wafer in Tc and wafer in G:
            phonon_noise = tes_noise.tes_phonon_noise_P(Tc[wafer], G[wafer][band], 0.5)*1e18
        else:
            phonon_noise = 0

#         def noise_model_fixed(x, A, alpha, photon, tau):
#             return noise_model(x, par_horizon[0], A, alpha, photon, tau)
        def noise_model_fixed(x, A, alpha, photon, tau):
            return noise_model_full(x, par_horizon[0], A, alpha, photon, phonon_noise, tau)

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
                 label='total noise model: ' + \
                       '$\\tau = ${:.1f} $\pm$ {:.1f} msec, '.format(par_normal[3]*1e3, errs[3]*1e3))
        plt.plot(f_normal, photon_noise(f_normal, phonon_noise, par_normal[3]), '--',
                 label='phonon noise:' + 
                       '$ N_{{ph}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(phonon_noise))
        plt.plot(f_normal, photon_noise(f_normal, par_normal[2], par_normal[3]), '--',
                 label='photon noise: ' +
                       '$N_\gamma = ${:.1f} $\pm$ {:.1f} aW / $\sqrt{{Hz}}$'.format(par_normal[2], errs[2]))
        plt.plot(f_normal, readout_noise(f_normal, par_horizon[0]), '--',
                 label='readout noise: ' +
                       '$N_{{ro}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(par_horizon[0]))
        
        NEP_phph[band][wafer] = par_normal[2]
        
        plt.axis([0.01, 75, 20, 500])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('current noise [pA/rtHz]')
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.legend()
        plt.title('{}: {:.0f} GHz'.format(wafer, band))
        plt.tight_layout()
#         plt.savefig('figures_fixed_ro/noisefits_{}_{}_fixed_ro.png'.format(wafer, band), dpi=150)
        plt.close()
```

```python
kcmb_conversion_factors = {
  # Best guess for W28A2 based on a map using RCW28 calibration
  'W28A2': {
    90.0*core.G3Units.GHz: 4.858e-8*core.G3Units.K,
    150.0*core.G3Units.GHz: 3.536e-8*core.G3Units.K,
    220.0*core.G3Units.GHz: 6.560e-8*core.G3Units.K,
  },
  'RCW38': {
    90.0*core.G3Units.GHz: 4.0549662e-07*core.G3Units.K,
    150.0*core.G3Units.GHz: 2.5601153e-07*core.G3Units.K,
    220.0*core.G3Units.GHz: 2.8025804e-07*core.G3Units.K,
  },
  'MAT5A': {
    90.0*core.G3Units.GHz: 2.5738063e-07*core.G3Units.K, # center (608, 555)
    150.0*core.G3Units.GHz: 1.7319235e-07*core.G3Units.K,
    220.0*core.G3Units.GHz: 2.145164e-07*core.G3Units.K,
  },
}

pWperK = {}
for bolo in normal_data[0]["CalibratorResponse"].keys():
    try:
        band = normal_data[0]["BolometerProperties"][bolo].band
        cal = normal_data[0]["CalibratorResponse"][bolo]
        fluxcal = normal_data[0]["RCW38FluxCalibration"][bolo]
        skytrans = normal_data[0]["RCW38SkyTransmission"][str(int(band / core.G3Units.GHz))]
        intflux = normal_data[0]["RCW38IntegralFlux"][bolo]
        band = normal_data[0]["BolometerProperties"][bolo].band
        pWperK[bolo] = cal * fluxcal * skytrans * intflux / kcmb_conversion_factors['RCW38'][band] / \
                        (1e-12*core.G3Units.watt / core.G3Units.kelvin) * (-1)
    except:
        pass
```

```python
pWperK_median = {}
for band in bands:
    pWperK_median[band] = {}
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        
        pWperK_median[band][wafer] = np.median([pWperK[bolo] for bolo in group \
                                                             if bolo in pWperK])
```

```python
# plot of raw NEP with "dfmux calibration"
for band in NEP_phph:
    nep_plot = [NEP_phph[band][wafer] for wafer in NEP_phph[band]]
    wafer_plot = [wafer for wafer in NEP_phph[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.xlabel('wafer')
plt.ylabel('NEP [aW/rtHz]')
plt.legend()
plt.grid()
plt.title('median NEP with "dfmux calibration"')
plt.tight_layout()
plt.savefig('figures/nep_bywafer_dfmuxcal.png', dpi=200)
```

```python
# plot of pW/K factors derived from RCW38
for band in pWperK_median:
    pWperK_plot = [pWperK_median[band][wafer] for wafer in pWperK_median[band]]
    wafer_plot = [wafer for wafer in pWperK_median[band]]
    plt.plot(np.arange(len(pWperK_plot)), pWperK_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(pWperK_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylim([0, 0.2])
plt.xlabel('wafer')
plt.ylabel('calibration factor [pW / K]')
plt.title('median calibration factors from RCW38')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/pWperK_RCW38_bywafer_dfmuxcal.png', dpi=200)
```

```python
# plot of NEP scaled by ratio of optics model pW/K and data pW/K
# pWperK_optical_model = {90:0.115, 150:0.122, 220:0.054}
pWperK_optical_model = {band: np.mean([pWperK_median[band][wafer] \
                                       for wafer in pWperK_median[band]]) \
                        for band in pWperK_median}
band_centers = {90:93.5e9, 150:146.7e9, 220:219.9e9}
bandwidths = {90:23.2e9, 150:30.7e9, 220:46.4e9}
Popt = {90:4.6e-12, 150:7.7e-12, 220:8.9e-12}

for jband, band in enumerate(NEP_phph):
    nep_plot = [NEP_phph[band][wafer] * \
                (pWperK_optical_model[band] / pWperK_median[band][wafer]) \
                for wafer in NEP_phph[band]]
    wafer_plot = [wafer for wafer in NEP_phph[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band),
             color='C{}'.format(jband))
    
    photon_nep = np.sqrt(tes_noise.shot_noise(band_centers[band], Popt[band])**2 + \
                           tes_noise.correlation_noise(band_centers[band], Popt[band], bandwidths[band], 1)**2)
    plt.plot(np.arange(len(nep_plot)),
             np.ones(len(nep_plot))*photon_nep*1e18, '--',
             color='C{}'.format(jband))
    
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.xlabel('wafer')
plt.ylabel('NEP [aW/rtHz]')
plt.title('median NEP with RCW38 relative response removed')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nep_bywafer_relcorr.png', dpi=200)
```

## Applying Joshua's TF

```python
with open('carrier_and_nuller_tf_adam.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
```

```python
tf_freqs             = tf_dict['freq']
tf_carrier           = tf_dict['tf_carrier_meth3']
dan_nuller_corr      = tf_dict['dan_nuller_corr']
tf_nuller            = tf_dict['tf_nuller']
tf_nuller_pydfmux    = convert_TF(gain=0, target='nuller', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
tf_carrier_pydfmux   = convert_TF(gain=0, target='carrier', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
power_correction     = dan_nuller_corr * tf_carrier / (tf_carrier_pydfmux/0.01)
```

```python
power_corr_median = {}
for band in bands:
    power_corr_median[band] = {}
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        
        power_corr = []
        for bolo, pcorr in zip(tf_dict['name'], power_correction):
            if bolo in group:
                power_corr.append(pcorr)
        
        power_corr_median[band][wafer] = np.median(power_corr)
```

```python
plt.plot(tf_freqs, tf_carrier, '.')
plt.plot(tf_freqs, tf_carrier_pydfmux/0.01, '.')
```

```python
plt.plot(tf_freqs, dan_nuller_corr, '.')
plt.plot(tf_freqs, tf_nuller_pydfmux, '.')
```

```python
plt.plot(tf_freqs, tf_nuller_pydfmux/0.01, '.')
plt.plot(tf_freqs, tf_carrier_pydfmux/0.01, '.')
```

```python
plt.plot(tf_freqs, power_correction, '.')
```

```python
for band in power_corr_median:
    nep_plot = [power_corr_median[band][wafer]
                for wafer in power_corr_median[band]]
    wafer_plot = [wafer for wafer in power_corr_median[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
# plt.ylabel('NEP [aW/rtHz]')
plt.legend()
plt.grid()
plt.tight_layout()
```

```python
tes_noise.correlation_noise(95e8, 4.5e-12, 23.3e8, 1)
```

## Joshua Correction v3

```python
with open('carrier_and_nuller_tf_adam_20200428.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
```

```python
tf_dict
```

```python
tf_names             = np.array([bolo for bolo in tf_dict])
tf_freqs             = np.array([tf_dict[bolo]['bias_freq'] for bolo in tf_dict])
tf_carrier           = np.array([tf_dict[bolo]['carrier_tf'] for bolo in tf_dict])
dan_nuller_corr      = np.array([tf_dict[bolo]['dan_current_correction'] for bolo in tf_dict])
tf_nuller_pydfmux    = convert_TF(gain=0, target='nuller', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
tf_carrier_pydfmux   = convert_TF(gain=0, target='carrier', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
power_correction     = dan_nuller_corr * tf_carrier / (tf_carrier_pydfmux/0.01)
```

```python
plt.plot(tf_freqs, power_correction, '.')
```

```python
power_corr_median = {}
for band in bands:
    power_corr_median[band] = {}
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        
        power_corr = []
        for bolo, pcorr in zip(tf_names, power_correction):
            if bolo in group:
                power_corr.append(pcorr)
        
        power_corr_median[band][wafer] = np.median(power_corr)
```

```python
for band in power_corr_median:
    nep_plot = [power_corr_median[band][wafer]
                for wafer in power_corr_median[band]]
    wafer_plot = [wafer for wafer in power_corr_median[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
# plt.ylabel('NEP [aW/rtHz]')
plt.legend()
plt.grid()
plt.tight_layout()
```

## Joshua Correction v4

```python
with open('carrier_and_nuller_tf_adam_with_pcorrection.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
```

```python
tf_dict
```

```python
tf_names             = np.array([bolo for bolo in tf_dict])
tf_freqs             = np.array([tf_dict[bolo]['bias_freq'] for bolo in tf_dict])
tf_carrier           = np.array([tf_dict[bolo]['carrier_tf'] for bolo in tf_dict])
dan_nuller_corr      = np.array([tf_dict[bolo]['dan_current_correction'] for bolo in tf_dict])
tf_nuller_pydfmux    = convert_TF(gain=0, target='nuller', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
tf_carrier_pydfmux   = convert_TF(gain=0, target='carrier', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
power_calibration    = np.array([tf_dict[bolo]['power_calibration'] for bolo in tf_dict])
```

```python
plt.plot(tf_freqs, power_calibration, '.', markersize=2)
plt.xlabel('bias frequency [Hz]')
plt.ylabel('power calibration factor')
plt.title('montgojo calibration (carrier_and_nuller_tf_adam_with_pcorrection)')
plt.tight_layout()
plt.savefig('figures/power_cal_factor.png', dpi=200)
```

```python
power_corr_median = {}
for band in bands:
    power_corr_median[band] = {}
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        
        power_corr = []
        for bolo, pcorr in zip(tf_names, power_calibration):
            if bolo in group:
                power_corr.append(pcorr)
        
        if len(power_corr)>0:
            power_corr_median[band][wafer] = np.median(power_corr)
        
for band in power_corr_median:
    nep_plot = [power_corr_median[band][wafer]
                for wafer in power_corr_median[band]]
    wafer_plot = [wafer for wafer in power_corr_median[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylabel('$[pW_{Montreal}] / [pW_{dfmux}]$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/montreal_pW_cal_factor.png', dpi=200)
```

```python
f_min = 0.01
f_max = 60

NEP_phph = {}

for band in bands:
    NEP_phph[band] = {}
    for jwafer, wafer in enumerate(wafers):
        if wafer in Tc and wafer in G and wafer in power_corr_median[band]:
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

            phonon_noise = tes_noise.tes_phonon_noise_P(Tc[wafer], G[wafer][band], 0.5)*1e18 / \
                            power_corr_median[band][wafer]
            
            def noise_model_fixed(x, A, alpha, photon, tau):
                return noise_model_full(x, par_horizon[0], A, alpha, photon / power_corr_median[band][wafer],
                                        phonon_noise, tau)

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
                     label='total noise model: ' + \
                           '$\\tau = ${:.1f} $\pm$ {:.1f} msec, '.format(par_normal[3]*1e3, errs[3]*1e3))
            plt.plot(f_normal, photon_noise(f_normal, phonon_noise, par_normal[3]), '--',
                     label='phonon noise:' + 
                           '$ N_{{ph}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(phonon_noise))
            plt.plot(f_normal, photon_noise(f_normal, par_normal[2], par_normal[3]), '--',
                     label='photon noise: ' +
                           '$N_\gamma = ${:.1f} $\pm$ {:.1f} aW / $\sqrt{{Hz}}$'.format(par_normal[2], errs[2]))
            plt.plot(f_normal, readout_noise(f_normal, par_horizon[0]), '--',
                     label='readout noise: ' +
                           '$N_{{ro}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(par_horizon[0]))

            NEP_phph[band][wafer] = par_normal[2]

            plt.axis([0.01, 75, 20, 500])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('current noise [pA/rtHz]')
            plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
            plt.legend()
            plt.title('{}: {:.0f} GHz'.format(wafer, band))
            plt.tight_layout()
    #         plt.savefig('figures_fixed_ro/noisefits_{}_{}_fixed_ro.png'.format(wafer, band), dpi=150)
    #         plt.close()
```

```python
# plot of raw NEP with "dfmux calibration"
for band in NEP_phph:
    nep_plot = [NEP_phph[band][wafer] for wafer in NEP_phph[band]]
    wafer_plot = [wafer for wafer in NEP_phph[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.xlabel('wafer')
plt.ylabel('NEP [aW/rtHz]')
plt.legend()
plt.grid()
plt.title('median NEP in Montreal aW/rtHz')
plt.tight_layout()
plt.savefig('figures/nep_bywafer_montgojo_cal.png', dpi=200)
```

## Joshua Correction v5

```python
with open('carrier_and_nuller_tf_0p5nH_complex.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
```

```python
tf_names             = np.array([bolo for bolo in tf_dict])
tf_freqs             = np.array([tf_dict[bolo]['bias_freq'] for bolo in tf_dict])
tf_carrier           = np.array([tf_dict[bolo]['carrier_tf'] for bolo in tf_dict])
dan_nuller_corr      = np.array([tf_dict[bolo]['dan_current_correction'] for bolo in tf_dict])
tf_nuller_pydfmux    = convert_TF(gain=0, target='nuller', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
tf_carrier_pydfmux   = convert_TF(gain=0, target='carrier', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
power_calibration    = np.array([tf_dict[bolo]['power_calibration'] for bolo in tf_dict])
```

```python
plt.plot(tf_freqs, power_calibration, '.', markersize=2)
plt.xlabel('bias frequency [Hz]')
plt.ylabel('power calibration factor')
plt.title('montgojo calibration (carrier_and_nuller_tf_0p5nH_complex)')
plt.tight_layout()
plt.savefig('figures/power_cal_factor.png', dpi=200)
```

```python
power_corr_median = {}
for band in bands:
    power_corr_median[band] = {}
    for jwafer, wafer in enumerate(wafers):
        groupname = '{:.1f}_{}'.format(band, wafer)
        group = bolo_tgroups[groupname]
        
        power_corr = []
        for bolo, pcorr in zip(tf_names, power_calibration):
            if bolo in group:
                power_corr.append(pcorr)
        
        if len(power_corr)>0:
            power_corr_median[band][wafer] = np.median(power_corr)
        
for band in power_corr_median:
    nep_plot = [power_corr_median[band][wafer]
                for wafer in power_corr_median[band]]
    wafer_plot = [wafer for wafer in power_corr_median[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylabel('$[pW_{Montreal}] / [pW_{dfmux}]$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/montreal_pW_cal_factor.png', dpi=200)
```

```python
f_min = 0.01
f_max = 60

NEP_photon = {}
NEP_phonon = {}
NEP_readout = {}

for band in bands:
    NEP_photon[band] = {}
    NEP_phonon[band] = {}
    NEP_readout[band] = {}
    for jwafer, wafer in enumerate(wafers):
        if wafer in Tc and wafer in G and wafer in power_corr_median[band]:
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

            phonon_noise = tes_noise.tes_phonon_noise_P(Tc[wafer], G[wafer][band], 0.5)*1e18 / \
                            power_corr_median[band][wafer]
            
            def noise_model_fixed(x, A, alpha, photon, tau):
                return noise_model_full(x, par_horizon[0], A, alpha, photon / power_corr_median[band][wafer],
                                        phonon_noise / power_corr_median[band][wafer], tau)

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
                     label='total noise model: ' + \
                           '$\\tau = ${:.1f} $\pm$ {:.1f} msec, '.format(par_normal[3]*1e3, errs[3]*1e3))
            plt.plot(f_normal, photon_noise(f_normal, phonon_noise, par_normal[3]), '--',
                     label='phonon noise:' + 
                           '$ N_{{ph}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(phonon_noise))
            plt.plot(f_normal, photon_noise(f_normal, par_normal[2], par_normal[3]), '--',
                     label='photon noise: ' +
                           '$N_\gamma = ${:.1f} $\pm$ {:.1f} aW / $\sqrt{{Hz}}$'.format(par_normal[2], errs[2]))
            plt.plot(f_normal, readout_noise(f_normal, par_horizon[0]), '--',
                     label='readout noise: ' +
                           '$N_{{ro}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(par_horizon[0]))

            NEP_photon[band][wafer] = par_normal[2]
            NEP_phonon[band][wafer] = phonon_noise
            NEP_readout[band][wafer] = par_horizon[0]
            
            
            plt.axis([0.01, 75, 20, 500])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('current noise [pA/rtHz]')
            plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
            plt.legend()
            plt.title('{}: {:.0f} GHz'.format(wafer, band))
            plt.tight_layout()
    #         plt.savefig('figures_fixed_ro/noisefits_{}_{}_fixed_ro.png'.format(wafer, band), dpi=150)
    #         plt.close()
```

```python
# plot of raw NEP with "dfmux calibration"
plt.figure(1)
for jband, band in enumerate(NEP_photon):
    nep_plot = [NEP_photon[band][wafer] for wafer in NEP_photon[band]]
    wafer_plot = [wafer for wafer in NEP_photon[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
    
    photon_nep = np.sqrt(tes_noise.shot_noise(band_centers[band], Popt[band])**2 + \
                           tes_noise.correlation_noise(band_centers[band], Popt[band], bandwidths[band], 1)**2)
    plt.plot(np.arange(len(nep_plot)),
             np.ones(len(nep_plot))*photon_nep*1e18, '--',
             color='C{}'.format(jband))
    
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.xlabel('wafer')
plt.ylabel('NEP [aW/rtHz]')
plt.legend()
plt.grid()
plt.title('median NEP in Montreal aW/rtHz')
plt.tight_layout()
plt.savefig('figures/nep_photon_bywafer_montreal_cal.png', dpi=200)

plt.figure(2)
for jband, band in enumerate(NEP_phonon):
    nep_plot = [NEP_phonon[band][wafer]
                for wafer in NEP_phonon[band]]
    wafer_plot = [wafer for wafer in NEP_phonon[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylim([0,40])
plt.ylabel('phonon NEP [aW/rtHz]')
plt.title('phonon NEP')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nep_phonon_bywafer_montreal_cal.png', dpi=200)

plt.figure(3)
for jband, band in enumerate(NEP_readout):
    nep_plot = [NEP_readout[band][wafer]
                for wafer in NEP_readout[band]]
    wafer_plot = [wafer for wafer in NEP_readout[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylim([0,100])
plt.ylabel('readout NEP [aW/rtHz]')
plt.title('readout NEP from horizon noise stare')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nep_readout_bywafer_montreal_cal.png', dpi=200)
```

```python
# Popt from Daniel's 3g instrument paper script
Popt = {'w172': {90:5.5e-12, 150:8.2e-12, 220:8.0e-12},
        'w174': {90:4.3e-12, 150:8.9e-12, 220:8.4e-12},
        'w176': {90:2.6e-12, 150:5.4e-12, 220:7.3e-12},
        'w177': {90:3.0e-12, 150:6.2e-12, 220:6.9e-12},
        'w181': {90:3.8e-12, 150:7.7e-12, 220:7.6e-12},
        'w188': {90:1.8e-12, 150:5.0e-12, 220:5.1e-12},
        'w204': {90:5.4e-12, 150:9.3e-12, 220:9.3e-12},
        'w206': {90:3.6e-12, 150:6.8e-12, 220:7.3e-12}}

# plot of raw NEP with "dfmux calibration"
plt.figure(1)
for jband, band in enumerate(NEP_photon):
    photon_nep = [np.sqrt(tes_noise.shot_noise(band_centers[band], Popt[wafer][band] * power_corr_median[band][wafer])**2 + \
                          tes_noise.correlation_noise(band_centers[band], Popt[wafer][band] * power_corr_median[band][wafer], bandwidths[band], 1)**2)
                  for wafer in Popt if wafer in power_corr_median[band]]
    wafer_plot = [wafer for wafer in Popt if wafer in power_corr_median[band]]
    plt.plot(np.arange(len(photon_nep)),
             np.array(photon_nep)*1e18, 'o',
             color='C{}'.format(jband))
    
_ = plt.gca().set_xticks(np.arange(len(photon_nep)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.xlabel('wafer')
plt.ylabel('NEP [aW/rtHz]')
plt.legend()
plt.grid()
plt.title('median NEP in Montreal aW/rtHz')
plt.tight_layout()
```

## NET conversion
Given the general failure of most of our power calibration attempts and parasitic modeling, I am attempting here to convert NET into NEP using the pW/K implied by Brad's optical model. We know that our NET is rock solid: by construction it is insensitive to responsivity, and it is calibrated using on-sky quantities to eliminate any dependence on dfmux calibration. To get it into NEP, we have to assume a pW/K calibration factor, which makes an implicit assumption about optical efficiency. There is uncertainty here, but it is probably more tractable than the dfmux calibration quagmire.

```python
horizon_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_77863968_horizon_default_current.g3'
horizon_data = list(core.G3File(horizon_fname))

normal_fname = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                'gainmatching_noise_81433244_default_temperature.g3'
normal_data = list(core.G3File(normal_fname))

normal_fname_current = '/home/adama/SPT/spt3g_papers/2019/3g_instrument/docs/code/lowf/' + \
                       'gainmatching_noise_81433244_default_current.g3'
normal_data_current = list(core.G3File(normal_fname_current))

bolo_tgroups = get_template_groups(horizon_data[0]["BolometerProperties"], 
                                            per_band = True,
                                            per_wafer = True,
                                            include_keys = True)
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
        
        KperA = {}
        for bolo in group:
            if bolo in normal_data[1]['ASD'] and \
               np.all(np.isfinite(normal_data[1]['ASD'][bolo])):
                if groupname not in avg_asd_normal:
                    avg_asd_normal[groupname] = normal_data[1]['ASD'][bolo]
                else:
                    avg_asd_normal[groupname] += normal_data[1]['ASD'][bolo]
                KperA[bolo] = np.median(normal_data[1]['ASD'][bolo] / normal_data_current[1]['ASD'][bolo])
                nbolos += 1
        avg_asd_normal[groupname] /= nbolos

        nbolos = 0
        for bolo in group:
            if bolo in horizon_data[1]['ASD'] and \
               np.all(np.isfinite(horizon_data[1]['ASD'][bolo])) and \
               bolo in KperA:
                if groupname not in avg_asd_horizon:
                    avg_asd_horizon[groupname] = horizon_data[1]['ASD'][bolo] * KperA[bolo]
                else:
                    avg_asd_horizon[groupname] += horizon_data[1]['ASD'][bolo] * KperA[bolo]
                nbolos += 1
        avg_asd_horizon[groupname] /= nbolos
        

        plt.subplot(5,2,jwafer+1)
        plt.loglog(horizon_data[1]['ASD']['frequency'] / core.G3Units.Hz,
                   avg_asd_horizon[groupname])
        plt.loglog(normal_data[1]['ASD']['frequency'] / core.G3Units.Hz,
                   avg_asd_normal[groupname])
        plt.xlim([0.01, 75])
        plt.title('{} GHz: {}'.format(band, wafer))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('NEP [aW/rtHz]')
    plt.tight_layout()
#     plt.savefig('horizon_vs_intransition_{}.png'.format(band))
```

```python
f_min = 0.01
f_max = 60

NET_photon = {}
NEP_photon = {}
NEP_phonon = {}
NEP_readout = {}

for band in bands:
    NET_photon[band] = {}
    NEP_photon[band] = {}
    NEP_phonon[band] = {}
    NEP_readout[band] = {}
    for jwafer, wafer in enumerate(wafers):
        if wafer in Tc and wafer in G:
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

            phonon_noise = tes_noise.tes_phonon_noise_P(Tc[wafer], G[wafer][band], 0.5)*1e18

    #         def noise_model_fixed(x, A, alpha, photon, tau):
    #             return noise_model_full(x, par_horizon[0], A, alpha, photon, phonon_noise, tau)
            def noise_model_fixed(x, A, alpha, photon, tau):
                return noise_model_full(x, par_horizon[0], A, alpha, photon, phonon_noise, tau)

            par_normal, cov = curve_fit(noise_model_fixed,
                                        f_normal[(f_normal>f_min) & (f_normal<f_max)],
                                        asd_normal[(f_normal>f_min) & (f_normal<f_max)],
                                        bounds=([0, 0, 0, 0],
                                                [np.inf, np.inf, np.inf, np.inf]),
                                        p0=[10, 1, 10, 0.01])
            errs = np.diagonal(np.sqrt(cov))

            net_to_nep = dPdT(band_centers[band], bandwidths[band], optical_eff[band]) * np.sqrt(2) * 1e12
            plt.figure(figsize=(8,5))
            plt.plot(f_horizon, avg_asd_horizon[groupname],
                     label='horizon noise stare')
            plt.plot(f_normal, avg_asd_normal[groupname],
                     label='in-transition noise stare')
            plt.plot(f_normal, noise_model_fixed(f_normal, *par_normal), 'k',
                     label='total noise model: ' + \
                           '$\\tau = ${:.1f} $\pm$ {:.1f} msec, '.format(par_normal[3]*1e3, errs[3]*1e3))
            plt.plot(f_normal, photon_noise(f_normal, phonon_noise, par_normal[3]), '--',
                     label='phonon noise:' + 
                           '$ N_{{ph}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(phonon_noise))
            plt.plot(f_normal, photon_noise(f_normal, par_normal[2], par_normal[3]), '--',
                     label='photon noise: ' +
                           '$N_\gamma = ${:.1f} $\pm$ {:.1f} aW / $\sqrt{{Hz}}$'.format(par_normal[2]*net_to_nep,
                                                                                        errs[2]*net_to_nep))
            plt.plot(f_normal, readout_noise(f_normal, par_horizon[0]), '--',
                     label='readout noise: ' +
                           '$N_{{ro}} = ${:.1f} aW / $\sqrt{{Hz}}$'.format(par_horizon[0]*net_to_nep))

            NET_photon[band][wafer] = par_normal[2]
            NEP_photon[band][wafer] = par_normal[2]*net_to_nep
            NEP_phonon[band][wafer] = phonon_noise
            NEP_readout[band][wafer] = par_horizon[0]*net_to_nep

    #         plt.axis([0.01, 75, 20, 500])
            plt.xlim([0.01, 75])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('NET [uK rtsec]')
            plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
            plt.legend()
            plt.title('{}: {:.0f} GHz'.format(wafer, band))
            plt.tight_layout()
            plt.savefig('figures_fixed_ro/noisefits_{}_{}_fixed_ro.png'.format(wafer, band), dpi=150)
            plt.close()
```

```python
plt.figure(1)
for jband, band in enumerate(NEP_photon):
    nep_plot = [NEP_photon[band][wafer]
                for wafer in NEP_photon[band]]
    wafer_plot = [wafer for wafer in NEP_photon[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
    
    photon_nep = np.sqrt(tes_noise.shot_noise(band_centers[band], Popt[band])**2 + \
                           tes_noise.correlation_noise(band_centers[band], Popt[band], bandwidths[band], 1)**2)
    plt.plot(np.arange(len(nep_plot)),
             np.ones(len(nep_plot))*photon_nep*1e18, '--',
             color='C{}'.format(jband))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylabel('photon NEP [aW/rtHz]')
plt.title('photon NEP from NET assuming\noptical efficiencies of {} / {} / {}'.format(optical_eff[90],
                                                                              optical_eff[150],
                                                                              optical_eff[220],))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nep_photon_from_net.png', dpi=200)


plt.figure(2)
for jband, band in enumerate(NEP_phonon):
    nep_plot = [NEP_phonon[band][wafer]
                for wafer in NEP_phonon[band]]
    wafer_plot = [wafer for wafer in NEP_phonon[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylim([0,40])
plt.ylabel('phonon NEP [aW/rtHz]')
plt.title('phonon NEP')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nep_phonon_from_net.png', dpi=200)

plt.figure(3)
for jband, band in enumerate(NEP_readout):
    nep_plot = [NEP_readout[band][wafer]
                for wafer in NEP_readout[band]]
    wafer_plot = [wafer for wafer in NEP_readout[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylim([0,100])
plt.ylabel('readout NEP [aW/rtHz]')
plt.title('readout NEP from horizon noise stare')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nep_readout_from_net.png', dpi=200)
```

```python
for name, nep in {"photon": NEP_photon,
                  "phonon": NEP_phonon,
                  "readout": NEP_readout}.items():
    print(name)
    print('\t90 GHz\t150 GHz\t220 GHz')
    for wafer in nep[90]:
        print('{}\t{:.1f}\t{:.1f}\t{:.1f}'.format(wafer,
                                                  nep[90][wafer],
                                                  nep[150][wafer],
                                                  nep[220][wafer]))
    print()
```

```python
for band in NET_photon:
    nep_plot = [NET_photon[band][wafer]
                for wafer in NET_photon[band]]
    wafer_plot = [wafer for wafer in NET_photon[band]]
    plt.plot(np.arange(len(nep_plot)), nep_plot, 'o', label='{} GHz'.format(band))
_ = plt.gca().set_xticks(np.arange(len(nep_plot)))
_ = plt.gca().set_xticklabels(wafer_plot)
plt.ylim([0,1800])
plt.ylabel('NET [aW/rtHz]')
plt.legend()
plt.grid()
plt.tight_layout()
```

```python

```

## NET as a function of optical efficiency

```python
def NET_model(pixel_eff, band, nep_other=None, nep_other_relative=None):
    emissivity  = {220: np.array([0.050,0.100,0.100,0.143,0.040,0.010,0.119,0.119,
                                  0.036,0.010,0.035,0.010,0.010,0.010,0.085,1.000]),
                   150: np.array([0.050,0.050,0.100,0.098,0.180,0.010,0.082,0.082,
                                  0.025,0.004,0.024,0.007,0.007,0.007,0.060,1.000]),
                   90:  np.array([0.050,0.050,0.100,0.063,0.440,0.010,0.053,0.053,
                                  0.016,0.002,0.015,0.005,0.005,0.005,0.080,1.000])}
    reflection  = {220: np.array([0,0,0,0.09,0,0.05,0.09,0.09,0.09,0,0.012,0,0,0,0,0]),
                   150: np.array([0,0,0,0.01,0,0.05,0.01,0.01,0.01,0,0.012,0,0,0,0,0]),
                   90:  np.array([0,0,0,0.028,0,0.05,0.028,0.028,0.028,0,0.012,0,0,0,0,0])}
    T_element   = np.array([0.5,0.3,0.3,4,4,4,5,5,50,200,280,280,280,250,230,2.73])
    
    band_centers = {90:93e9, 150:148e9, 220:222e9}
    bandwidths = {90:27.9e9, 150:42.9e9, 220:51.1e9}
    
    eff_element = 1 - emissivity[band] - reflection[band]
    eff_element[0] = pixel_eff
    eff_element[1] = pixel_eff
    eff_element[-1] = 1.0
    eff_cumul = [eff_element[0]]
    for eff in eff_element[1:]:
        eff_cumul.append(eff_cumul[-1] * eff)
    
    power = (emissivity[band] + reflection[band]) * eff_cumul * h * band_centers[band] / \
            (np.exp(h/k * (band_centers[band]) / T_element) - 1) * bandwidths[band]
    total_power = np.sum(power)
        
    photon_nep = np.sqrt(tes_noise.shot_noise(band_centers[band], total_power)**2 + \
                         tes_noise.correlation_noise(band_centers[band], total_power, bandwidths[band], 1)**2)
    if nep_other is not None:
        nep = np.sqrt(photon_nep**2 + nep_other**2)
    elif nep_other_relative is not None:
        nep = np.sqrt(photon_nep**2 + (nep_other_relative*photon_nep)**2)
    net = nep / dPdT(band_centers[band], bandwidths[band], eff_cumul[-1]) * np.sqrt(2)

    return photon_nep, net, total_power
```

```python
eff_plot = np.linspace(0.5, 1)
nep_plot = np.array([NET_model(eff, 220, nep_other_relative=1)[0] for eff in eff_plot])
net_plot = np.array([NET_model(eff, 220, nep_other_relative=1)[1] for eff in eff_plot])
power_plot = np.array([NET_model(eff, 220, nep_other_relative=1)[2] for eff in eff_plot])

plt.figure(1, figsize=(6, 10))

plt.subplot(3,1,1)
plt.plot(eff_plot, np.sqrt(2)*nep_plot*1e18, label='total NEP')
plt.plot(eff_plot, nep_plot*1e18, '--', label='photon-only NEP')
plt.xlabel('efficiency of pixel')
plt.ylabel('NEP [aW / rtHz]')
plt.legend()
plt.grid()
plt.title('optical model with $NEP_{photon} = NEP_{other}$')

plt.subplot(3,1,2)
plt.plot(eff_plot, net_plot*1e6)
plt.xlabel('efficiency of pixel')
plt.ylabel('NET [uK rtsec]')
plt.grid()

plt.subplot(3,1,3)
plt.plot(eff_plot, power_plot*1e12)
plt.xlabel('efficiency of pixel')
plt.ylabel('optical power [pW]')
plt.grid()

plt.tight_layout()
plt.savefig('figures/optical_model_net.png', dpi=200)
```

```python
eff_plot = np.linspace(0.5, 1)
nep_plot = np.array([NET_model(eff, 150, nep_other_relative=0.5)[0] for eff in eff_plot])
net_plot = np.array([NET_model(eff, 150, nep_other_relative=0.5)[1] for eff in eff_plot])
power_plot = np.array([NET_model(eff, 150, nep_other_relative=0.5)[2] for eff in eff_plot])

plt.figure(1, figsize=(6, 10))

plt.subplot(3,1,1)
plt.plot(eff_plot, np.sqrt(1.5)*nep_plot*1e18, label='total NEP')
plt.plot(eff_plot, nep_plot*1e18, '--', label='photon-only NEP')
plt.xlabel('efficiency of pixel')
plt.ylabel('NEP [aW / rtHz]')
plt.legend()
plt.grid()
plt.title('optical model with $NEP_{photon} = NEP_{other}$')

plt.subplot(3,1,2)
plt.plot(eff_plot, net_plot*1e6)
plt.xlabel('efficiency of pixel')
plt.ylabel('NET [uK rtsec]')
plt.grid()

plt.subplot(3,1,3)
plt.plot(eff_plot, power_plot*1e12)
plt.xlabel('efficiency of pixel')
plt.ylabel('optical power [pW]')
plt.grid()

plt.tight_layout()
# plt.savefig('figures/optical_model_net.png', dpi=200)
```

```python
eff_plot = np.linspace(0.5, 1)
nep_plot = np.array([NET_model(eff, 90, nep_other_relative=0.5)[0] for eff in eff_plot])
net_plot = np.array([NET_model(eff, 90, nep_other_relative=0.5)[1] for eff in eff_plot])
power_plot = np.array([NET_model(eff, 90, nep_other_relative=0.5)[2] for eff in eff_plot])

plt.figure(1, figsize=(6, 10))

plt.subplot(3,1,1)
plt.plot(eff_plot, np.sqrt(1.5)*nep_plot*1e18, label='total NEP')
plt.plot(eff_plot, nep_plot*1e18, '--', label='photon-only NEP')
plt.xlabel('efficiency of pixel')
plt.ylabel('NEP [aW / rtHz]')
plt.legend()
plt.grid()
plt.title('optical model with $NEP_{photon} = NEP_{other}$')

plt.subplot(3,1,2)
plt.plot(eff_plot, net_plot*1e6)
plt.xlabel('efficiency of pixel')
plt.ylabel('NET [uK rtsec]')
plt.grid()

plt.subplot(3,1,3)
plt.plot(eff_plot, power_plot*1e12)
plt.xlabel('efficiency of pixel')
plt.ylabel('optical power [pW]')
plt.grid()

plt.tight_layout()
# plt.savefig('figures/optical_model_net.png', dpi=200)
```

```python
np.sqrt(1688**2 - 1211**2)
```

```python

```
