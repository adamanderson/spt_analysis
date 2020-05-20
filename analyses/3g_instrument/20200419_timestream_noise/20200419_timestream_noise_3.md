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
**Original date:** 29 April 2020  
**Name:** Adam Anderson

Our initial version of these studies tried to do the re-calibration on average over each wafer. It's not clear that this will help, but doing the re-calibration by bolometer instead makes the responsivity correction more straightforward.

```python
from spt3g import core, calibration
from spt3g.calibration.template_groups import get_template_groups
from spt3g.dfmux import HousekeepingForBolo
from spt3g.dfmux.unittransforms import bolo_bias_voltage_rms
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
with open('carrier_and_nuller_tf_0p5nH_complex.pkl', 'rb') as f:
    tf_dict = pickle.load(f)
    
tf_names             = np.array([bolo for bolo in tf_dict])
tf_freqs             = np.array([tf_dict[bolo]['bias_freq'] for bolo in tf_dict])
tf_carrier           = np.array([tf_dict[bolo]['carrier_tf'] for bolo in tf_dict])
dan_nuller_corr      = np.array([tf_dict[bolo]['dan_current_correction'] for bolo in tf_dict])
tf_nuller_pydfmux    = convert_TF(gain=0, target='nuller', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
tf_carrier_pydfmux   = convert_TF(gain=0, target='carrier', frequency=list(tf_freqs),\
                                  custom_TF='spt3g_filtering_2017_full')
power_calibration    = np.array([tf_dict[bolo]['power_calibration'] for bolo in tf_dict])
power_calibration_dict = {bolo: tf_dict[bolo]['power_calibration'] for bolo in tf_dict}
```

```python
horizon_fname = '/sptlocal/user/adama/instrument_paper_2019/nep_calibration/' + \
                'gainmatching_noise_77863968_horizon_default_power_biasinfo.g3'
horizon_data = list(core.G3File(horizon_fname))

normal_fname = '/sptlocal/user/adama/instrument_paper_2019/nep_calibration/' + \
                'gainmatching_noise_81433244_default_power_biasinfo.g3'
normal_data = list(core.G3File(normal_fname))

bolo_tgroups = get_template_groups(normal_data[0]["BolometerProperties"], 
                                   per_band = True,
                                   per_wafer = True,
                                   include_keys = True)
```

```python
# get bias frequencies
d = list(core.G3File('/spt/data/bolodata/fullrate/noise/77863968/0000.g3'))
bias_freq_horizon = {}
for bolo in horizon_data[0]["BolometerProperties"].keys():
    try:
        chan = HousekeepingForBolo(d[3]["DfMuxHousekeeping"], d[2]['WiringMap'], bolo)
        bias_freq_horizon[bolo] = chan.carrier_frequency
    except:
        pass
    
# get bias frequencies
d = list(core.G3File('/spt/data/bolodata/fullrate/noise/81433244/0000.g3'))
bias_freq_normal = {}
vbias_normal = {}
for bolo in horizon_data[0]["BolometerProperties"].keys():
    try:
        chan = HousekeepingForBolo(d[3]["DfMuxHousekeeping"], d[2]['WiringMap'], bolo)
        bias_freq_normal[bolo] = chan.carrier_frequency
        vbias_normal[bolo] = normal_data[1]['VBiasRMS'][bolo]
    except:
        pass
    
del(d)
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

## A Look At RCW38 Responsivity

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
pWperKperV = {}
bolo_bands = {}
bolo_wafers = {}
for bolo in normal_data[0]["CalibratorResponse"].keys():
    if bolo in vbias_normal and vbias_normal[bolo] != 0:
        try:
            band = normal_data[0]["BolometerProperties"][bolo].band
            cal = normal_data[0]["CalibratorResponse"][bolo]
            fluxcal = normal_data[0]["RCW38FluxCalibration"][bolo]
            skytrans = normal_data[0]["RCW38SkyTransmission"][str(int(band / core.G3Units.GHz))]
            intflux = normal_data[0]["RCW38IntegralFlux"][bolo]
            band = normal_data[0]["BolometerProperties"][bolo].band
            pWperK[bolo] = cal * fluxcal * skytrans * intflux / kcmb_conversion_factors['RCW38'][band] / \
                            (1e-12*core.G3Units.watt / core.G3Units.kelvin) * (-1)
            pWperKperV[bolo] = pWperK[bolo] / vbias_normal[bolo]
            bolo_bands[bolo] = band
            bolo_wafers[bolo] = normal_data[0]["BolometerProperties"][bolo].wafer_id
        except:
            pass
    
pWperK_rescaled = {}


bias_freq_plot    = np.array([bias_freq_normal[bolo] / core.G3Units.Hz \
                              for bolo in pWperK])
vbias_plot        = np.array([vbias_normal[bolo] / core.G3Units.volt \
                              for bolo in pWperK])
pWperK_plot       = np.array([pWperK[bolo] \
                              for bolo in pWperK])
pWperKperV_plot   = np.array([pWperKperV[bolo] \
                              for bolo in pWperK])
bolo_bands_plot   = np.array([bolo_bands[bolo] \
                              for bolo in pWperK])
bolo_wafers_plot  = np.array([bolo_wafers[bolo] \
                              for bolo in pWperK])
pWperKperV_rescaled = {}
pWperK_rescaled = {}
for wafer in wafers:
    scalings = {90: 1,
                150: np.median(pWperKperV_plot[(bolo_bands_plot==90*core.G3Units.GHz) & \
                                         (bolo_wafers_plot==wafer) & \
                                         (bias_freq_plot>2.3e6) & \
                                         np.isfinite(pWperKperV_plot)]) / \
                     np.median(pWperKperV_plot[(bolo_bands_plot==150*core.G3Units.GHz) & \
                                         (bolo_wafers_plot==wafer) & \
                                         (bias_freq_plot<2.7e6) & \
                                         np.isfinite(pWperKperV_plot)]),
                220: np.median(pWperKperV_plot[(bolo_bands_plot==150*core.G3Units.GHz) & \
                                         (bolo_wafers_plot==wafer) & \
                                         (bias_freq_plot>3.4e6) & \
                                         np.isfinite(pWperKperV_plot)]) / \
                     np.median(pWperKperV_plot[(bolo_bands_plot==220*core.G3Units.GHz) & \
                                         (bolo_wafers_plot==wafer) & \
                                         (bias_freq_plot<3.8e6) & \
                                         np.isfinite(pWperKperV_plot)])}
    scalings[220] = scalings[150] * scalings[220]

    for bolo in pWperK:
        if bolo_wafers[bolo] == wafer:
            pWperKperV_rescaled[bolo] = pWperKperV[bolo] * \
                                        scalings[bolo_bands[bolo]/core.G3Units.GHz]
            pWperK_rescaled[bolo] = pWperKperV[bolo] * vbias_normal[bolo] * \
                                    scalings[bolo_bands[bolo]/core.G3Units.GHz]
```

```python
# raw pW/K plot
plt.figure(figsize=(8,6))
for band in [90,150,220]:
    plt.plot(bias_freq_plot[(bolo_bands_plot==band*core.G3Units.GHz)],
             pWperK_plot[(bolo_bands_plot==band*core.G3Units.GHz)], '.')
plt.axis([1.5e6, 5.5e6, 0,0.25])
plt.xlabel('bias frequency [Hz]')
plt.ylabel('$pW_{dfmux}$ / K from RCW38')
plt.grid()
plt.title('all wafers')
plt.tight_layout()
plt.savefig('figures/pWperK_raw_dfmux_all.png', dpi=200)
```

```python
# raw pW/K/Vb plot
plt.figure(figsize=(8,6))
for band in [90,150,220]:
    plt.plot(bias_freq_plot[(bolo_bands_plot==band*core.G3Units.GHz)],
             pWperKperV_plot[(bolo_bands_plot==band*core.G3Units.GHz)], '.')
plt.axis([1.5e6, 5.5e6, 0, 1e5])
plt.xlabel('bias frequency [Hz]')
plt.ylabel('$pW_{dfmux}$ / K / $V_b$ from RCW38')
plt.grid()
plt.title('all wafers')
plt.tight_layout()
plt.savefig('figures/pWperKperV_raw_dfmux_all.png', dpi=200)
```

```python
bias_freq_plot   = np.array([bias_freq_normal[bolo] / core.G3Units.Hz \
                             for bolo in pWperK_rescaled])
vbias_plot       = np.array([vbias_normal[bolo] / core.G3Units.Hz \
                             for bolo in pWperK_rescaled])
pWperK_plot      = np.array([pWperK_rescaled[bolo] \
                             for bolo in pWperK_rescaled])
pWperKperV_plot  = np.array([pWperKperV_rescaled[bolo] \
                             for bolo in pWperK_rescaled])
bolo_bands_plot  = np.array([bolo_bands[bolo] \
                             for bolo in pWperK_rescaled])
bolo_wafers_plot = np.array([bolo_wafers[bolo] \
                             for bolo in pWperK_rescaled])

plt.figure(figsize=(8,15))
for jwafer, wafer in enumerate(wafers):
    plt.subplot(5,2,jwafer+1)
    norm = np.median(pWperKperV_plot[(bolo_bands_plot==150*core.G3Units.GHz) & \
                                     (bolo_wafers_plot==wafer)])
    for band in [90,150,220]:
        plt.plot(bias_freq_plot[(bolo_bands_plot==band*core.G3Units.GHz) & \
                                (bolo_wafers_plot==wafer)] / 1e6,
                 pWperKperV_plot[(bolo_bands_plot==band*core.G3Units.GHz) & \
                             (bolo_wafers_plot==wafer)],
                 '.', markersize=2)
    plt.axis([1.5, 5.5, 0, 50000])
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('$pW_{dfmux}$ / K / $V_b$ (rescaled)')
    plt.grid()
    plt.title(wafer)
    plt.tight_layout()
plt.savefig('figures/pWperKperV_rescaled_all.png', dpi=200)

plt.figure(figsize=(8,15))
for jwafer, wafer in enumerate(wafers):
    plt.subplot(5,2,jwafer+1)
    for band in [90,150,220]:
        plt.plot(bias_freq_plot[(bolo_bands_plot==band*core.G3Units.GHz) & \
                                (bolo_wafers_plot==wafer)] / 1e6,
                 pWperK_plot[(bolo_bands_plot==band*core.G3Units.GHz) & \
                             (bolo_wafers_plot==wafer)], '.', markersize=2)
    plt.axis([1.5, 5.5, 0, 0.15])
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('$pW_{dfmux}$ / K (rescaled)')
    plt.grid()
    plt.title(wafer)
    plt.tight_layout()
plt.savefig('figures/pWperK_rescaled_all.png', dpi=200)
```

## Rescaling NEP by pW/K from RCW38

```python
with open('77863968_fit_params_vbias_corr.pkl', 'rb') as f:
    data_fits = pickle.load(f)
```

```python
phph_noise             = np.array([data_fits['in-transition'][bolo][2] \
                                   for bolo in data_fits['in-transition'] \
                                   if bolo in pWperK_rescaled])
phph_noise_rescaled    = np.array([data_fits['in-transition'][bolo][2] / \
                                   pWperK_rescaled[bolo] \
                                   for bolo in data_fits['in-transition'] \
                                   if bolo in pWperK_rescaled])
bias_freq              = np.array([bias_freq_normal[bolo] / core.G3Units.Hz \
                                   for bolo in data_fits['in-transition'] \
                                   if bolo in pWperK_rescaled])
wafers_plot            = np.array([bolo_wafers[bolo] \
                                   for bolo in data_fits['in-transition'] \
                                   if bolo in pWperK_rescaled])
bands_plot             = np.array([bolo_bands[bolo] / core.G3Units.GHz \
                                   for bolo in data_fits['in-transition'] \
                                   if bolo in pWperK_rescaled])
```

```python
plt.figure(figsize=(12,30))
for jwafer, wafer in enumerate(wafers):
    plt.subplot(10,3,3*jwafer + 1)
    for band in [90, 150, 220]:
        median150 = np.median(phph_noise[(wafers_plot==wafer) & \
                                         (bands_plot==150)])
        plt.plot(bias_freq[(wafers_plot==wafer) & (bands_plot==band)] / 1e6,
                 phph_noise[(wafers_plot==wafer) & (bands_plot==band)],
                 '.', markersize=2)
    plt.axis([1.5, 5.5, 0, 200])
    plt.xlabel('bias frequency [MHz]')
    plt.ylabel('photon + phonon $NEP_{dfmux}$ [aW / $\sqrt{Hz}$]')
    plt.tight_layout()
    
    plt.subplot(10,3,3*jwafer + 2)
    for band in [90, 150, 220]:
        median150 = np.median(phph_noise_rescaled[(wafers_plot==wafer) & \
                                                  (bands_plot==150)])
        plt.plot(bias_freq[(wafers_plot==wafer) & (bands_plot==band)] / 1e6,
                 phph_noise_rescaled[(wafers_plot==wafer) & (bands_plot==band)]/median150,
                 '.', markersize=2)
    plt.axis([1.5, 5.5, 0, 2])
    plt.title(wafer)
    plt.xlabel('bias frequency [MHz]')
    plt.ylabel('photon + phonon $NEP$ (rescaled)')
    plt.tight_layout()
    
    plt.subplot(10,3,3*jwafer + 3)
    for band in [90, 150, 220]:
        median150 = np.median(phph_noise_rescaled[(wafers_plot==wafer) & \
                                                  (bands_plot==150)])
        phph_plot = phph_noise_rescaled[(wafers_plot==wafer) & (bands_plot==band)]/median150
        plt.hist(phph_plot[np.isfinite(phph_plot)],
                 bins=np.linspace(0, 2, 51), histtype='step',
                 label='{} GHz'.format(band))
    plt.legend()
    plt.xlabel('photon + phonon $NEP$ (rescaled)')
    plt.ylabel('bolometers')
    plt.tight_layout()
plt.savefig('figures/NEP_rescaled_all.png', dpi=200)

```

## Scratch work
