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
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
```

```python
cal_datapath = '/sptgrid/analysis/calibration/calibrator/'
nep_datapath = '/sptgrid/analysis/calibration/noise/'
obsids_cal   = {220: [111526789, 111554041, 111555584, 111556717]}
obsids_noise = {220: [111526522, 111554114, 111555660, 111556790]}
bps = list(core.G3File('/spt/data/bolodata/fullrate/calibrator/'
                       '93508217/offline_calibration.g3'))[0]["BolometerProperties"]

cal_response = {}
cal_rfrac = {}
cal_bololist = {}
nep = {}
nep_bololist = {}

for wafer in ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']:
    cal_response[wafer] = {}
    cal_rfrac[wafer] = {}
    cal_bololist[wafer] = {}
    nep[wafer] = {}
    nep_bololist[wafer] = {}
    
    for band in obsids_cal:
        cal_response[wafer][band] = {}
        cal_rfrac[wafer][band] = {}
        cal_bololist[wafer][band] = {}
        nep[wafer][band] = {}
        nep_bololist[wafer][band] = {}
    
        for obsid_cal, obsid_noise in zip(obsids_cal[band], obsids_noise[band]):
            cal_data = list(core.G3File('{}/{}.g3'.format(cal_datapath, obsid_cal)))[0]
            cal_response[wafer][band][obsid_cal] = []
            cal_rfrac[wafer][band][obsid_cal] = []
            cal_bololist[wafer][band][obsid_cal] = []

            nep_data = list(core.G3File('{}/{}.g3'.format(nep_datapath, obsid_noise)))[0]
            nep[wafer][band][obsid_noise] = []
            nep_bololist[wafer][band][obsid_noise] = []

            for bolo in cal_data["CalibratorResponse"].keys():
                if bps[bolo].band/core.G3Units.GHz == band and \
                   bps[bolo].wafer_id == wafer and \
                   np.isfinite(cal_data["CalibratorResponse"][bolo]) and\
                   np.isfinite(cal_data["CalibratorResponseRfrac"][bolo]):
                    cal_response[wafer][band][obsid_cal].append(cal_data["CalibratorResponse"][bolo])
                    cal_rfrac[wafer][band][obsid_cal].append(cal_data["CalibratorResponseRfrac"][bolo])
                    cal_bololist[wafer][band][obsid_cal].append(bolo)

            cal_response[wafer][band][obsid_cal] = np.array(cal_response[wafer][band][obsid_cal])
            cal_rfrac[wafer][band][obsid_cal] = np.array(cal_rfrac[wafer][band][obsid_cal])
            cal_bololist[wafer][band][obsid_cal] = np.array(cal_bololist[wafer][band][obsid_cal])

            for bolo in nep_data["NEP_10.0Hz_to_15.0Hz"].keys():
                if bps[bolo].band/core.G3Units.GHz == band and \
                   bps[bolo].wafer_id == wafer and \
                   np.isfinite(nep_data["NEP_10.0Hz_to_15.0Hz"][bolo]):
                    nep[wafer][band][obsid_noise].append(nep_data["NEP_10.0Hz_to_15.0Hz"][bolo])
                    nep_bololist[wafer][band][obsid_noise].append(bolo)

            nep[wafer][band][obsid_noise] = np.array(nep[wafer][band][obsid_noise])
            nep_bololist[wafer][band][obsid_noise] = np.array(nep_bololist[wafer][band][obsid_noise])
```

```python
max_response = {90: 4, 150: 15, 220: 15}

for jband, band in enumerate(list(cal_response[wafer].keys())):
    plt.figure(jband, figsize=(10,16))
    for jwafer, wafer in enumerate(cal_response):
        plt.subplot(5, 2, jwafer+1)
        for obsid in cal_response[wafer][band]:
            plt.hist(cal_response[wafer][band][obsid]/(core.G3Units.watt*1e-15),
                     bins=np.linspace(0,max_response[band],51),
                     histtype='stepfilled', alpha=0.5,
                     label='rfrac = {:.2f}; median resp. = {:.2f}'\
                                 .format(np.mean(cal_rfrac[wafer][band][obsid]),
                                         np.median(cal_response[wafer][band][obsid]/(core.G3Units.watt*1e-15))))
        plt.title('{}: {}'.format(wafer, band))
        plt.legend()
        plt.xlabel('(nominal) calibrator response [fW]')
        plt.ylabel('bolometers')
        plt.tight_layout()
        plt.savefig('{}_cal_hist_winter.png'.format(band), dpi=150)
```

```python
for jband, band in enumerate(list(cal_response[wafer].keys())):
    plt.figure(jband, figsize=(10,16))
    for jwafer, wafer in enumerate(cal_response):
        plt.subplot(5, 2, jwafer+1)
        for obsid_nep, obsid_cal in zip(nep[wafer][band], cal_response[wafer][band]):
            nep_aWperrtHz = nep[wafer][band][obsid_nep]/\
                            (core.G3Units.watt*1e-18 / np.sqrt(core.G3Units.Hz))
            plt.hist(nep_aWperrtHz,
                     bins=np.linspace(0,300,51),
                     histtype='stepfilled', alpha=0.5,
                     label='rfrac = {:.2f}; median NEP = {:.2f}'\
                                 .format(np.mean(cal_rfrac[wafer][band][obsid_cal]),
                                         np.median(nep_aWperrtHz[(nep_aWperrtHz>1) & (nep_aWperrtHz<300)])))
        plt.title('{}: {}'.format(wafer, band))
        plt.legend()
        plt.xlabel('NEP [aW / $\sqrt{Hz}$]')
        plt.ylabel('bolometers')
        plt.tight_layout()
        plt.savefig('{}_nep_winter.png'.format(band), dpi=150)
```

```python
for jband, band in enumerate(list(cal_response[wafer].keys())):
    plt.figure(jband, figsize=(10,16))
    for jwafer, wafer in enumerate(cal_response):
        plt.subplot(5, 2, jwafer+1)
        for obsid_cal, obsid_noise in zip(obsids_cal[band], obsids_noise[band]):
            cal_sn = []
            for jbolo, bolo in enumerate(cal_bololist[wafer][band][obsid_cal]):
                if bolo in nep_bololist[wafer][band][obsid_noise]:
                    cal_sn.append((cal_response[wafer][band][obsid_cal][jbolo]) / \
                                  (nep[wafer][band][obsid_noise][nep_bololist[wafer][band][obsid_noise]==bolo][0]/\
                                       (1 / np.sqrt(core.G3Units.Hz))))

            cal_sn = np.array(cal_sn)
            plt.hist(cal_sn[(cal_sn>0) & (cal_sn<150)],
                     bins=np.linspace(0,100,51),
                     histtype='stepfilled', alpha=0.5,
                     label='rfrac = {:.2f}; median S/N = {:.2f}'\
                                 .format(np.mean(cal_rfrac[wafer][band][obsid_cal]),
                                         np.median(cal_sn)))
        plt.title('{}: {}'.format(wafer, band))
        plt.legend()
        plt.xlabel('calibrator S/N [$\sqrt{Hz}$]')
        plt.ylabel('bolometers')
        plt.tight_layout()
        plt.savefig('{}_cal_sn_winter.png'.format(band), dpi=150)
```

```python

```
