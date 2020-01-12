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

# Analysis of cal vs. rfrac data

```python
import numpy as np
from spt3g import core, calibration
import matplotlib.pyplot as plt
from functools import reduce
import os
import pickle
```

## First test: calibrator response only

```python
datapath = '/spt/user/production/calibration/calibrator/'
obsids = {'w206': [93507113, 93508217, 93509295, 93510380, 93511444, 93512566]} #,
#           'w174': [93708436, 93709515, 93710641, 93713130, 93714249, 93715328]}
#'w174': [93513647, 93514727, 93515794, 93516849]}
```

```python
bps = list(core.G3File('/spt/data/bolodata/fullrate/calibrator/'
                       '93508217/offline_calibration.g3'))[0]["BolometerProperties"]
```

```python
cal_response = {}
cal_rfrac = {}
bololist = {}

for wafer in obsids.keys():
    cal_response[wafer] = {}
    cal_rfrac[wafer] = {}
    bololist[wafer] = {}
    
    for obsid in obsids[wafer]:
        cal_data = list(core.G3File('{}/{}.g3'.format(datapath, obsid)))[0]
        cal_response[wafer][obsid] = []
        cal_rfrac[wafer][obsid] = []
        bololist[wafer][obsid] = []

        for bolo in cal_data["CalibratorResponse"].keys():
            if bps[bolo].band/core.G3Units.GHz == 90 and \
               bps[bolo].wafer_id == wafer and \
               np.isfinite(cal_data["CalibratorResponse"][bolo]) and\
               np.isfinite(cal_data["CalibratorResponseRfrac"][bolo]):
                cal_response[wafer][obsid].append(cal_data["CalibratorResponse"][bolo])
                cal_rfrac[wafer][obsid].append(cal_data["CalibratorResponseRfrac"][bolo])
                bololist[wafer][obsid].append(bolo)
        
        cal_response[wafer][obsid] = np.array(cal_response[wafer][obsid])
        cal_rfrac[wafer][obsid] = np.array(cal_rfrac[wafer][obsid])
        bololist[wafer][obsid] = np.array(bololist[wafer][obsid])
```

```python
for jwafer, wafer in enumerate(cal_response):
    plt.figure(jwafer)
    
    for obsid in cal_response[wafer]:
        plt.hist(cal_response[wafer][obsid]/(core.G3Units.watt*1e-15),
                 bins=np.linspace(0,3,21),
                 histtype='stepfilled', alpha=0.5,
                 label='rfrac = {:.2f}'.format(np.mean(cal_rfrac[wafer][obsid])))
    plt.title(wafer)
    plt.legend()
    plt.xlabel('(nominal) calibrator response [fW]')
    plt.ylabel('bolometers')
    plt.tight_layout()
    plt.savefig('{}_cal_hist.png'.format(wafer), dpi=150)
```

```python
for jwafer,wafer in enumerate(cal_rfrac.keys()):
    common_bolos = reduce(np.intersect1d, bololist[wafer].values())
    random_bolos = np.random.choice(common_bolos, 10, replace=False)

    plt.figure(jwafer)
    for bolo in random_bolos:
        rfracs    = np.array([cal_rfrac[wafer][obsid][bololist[wafer][obsid]==bolo] \
                              for obsid in cal_rfrac[wafer].keys()])
        responses = np.array([cal_response[wafer][obsid][bololist[wafer][obsid]==bolo] \
                              for obsid in cal_response[wafer].keys()])
        plt.plot(rfracs,
                 responses/(core.G3Units.watt*1e-15), 
                 'o-', label='{}'.format(bolo))
    plt.title(wafer)
    plt.legend()
    plt.ylabel('rfrac')
    plt.ylabel('(nominal) calibrator response [fW]')
    plt.tight_layout()
    plt.savefig('{}_cal_by_bolo.png'.format(wafer), dpi=150)
```

```python
wafer = 'w206'
obsid_0 = 93507113
common_bolos = reduce(np.intersect1d, bololist[wafer].values())

cal_response_0 = np.array([cal_response[wafer][obsid_0][bololist[wafer][obsid_0]==bolo] \
                           for bolo in common_bolos])
for obsid in cal_response[wafer].keys():
    if obsid != obsid_0:
        responses  = np.array([cal_response[wafer][obsid][bololist[wafer][obsid]==bolo] \
                               for bolo in common_bolos])
        plt.hist((responses - cal_response_0)/(core.G3Units.watt*1e-15),
                 bins=np.linspace(-0.5, 1.5, 51), histtype='step',
                 label='{:.2f}'.format(np.mean(responses - cal_response_0)))
plt.legend()
```

## Second test: better S/N estimates


### Calibrator S/N

```python
cal_datapath = '/spt/user/production/calibration/calibrator/'
nep_datapath = '/spt/user/production/calibration/noise/'
obsids_cal = {'w206': [94405616, 94406496, 94407376, 94408259, 94409175],
              'w174': [94410720, 94411599, 94412465, 94413333, 94415122]}
obsids_noise = {'w206': [94405689, 94406569, 94407449, 94408332, 94409248],
                'w174': [94410793, 94411672, 94412538, 94413407, 94415195]}
band = '90 GHz'
bps = list(core.G3File('/spt/data/bolodata/fullrate/calibrator/'
                       '93508217/offline_calibration.g3'))[0]["BolometerProperties"]

cal_response = {}
cal_rfrac = {}
cal_bololist = {}
nep = {}
nep_bololist = {}

for wafer in obsids_cal.keys():
    cal_response[wafer] = {}
    cal_rfrac[wafer] = {}
    cal_bololist[wafer] = {}
    nep[wafer] = {}
    nep_bololist[wafer] = {}
    
    for obsid_cal, obsid_noise in zip(obsids_cal[wafer], obsids_noise[wafer]):
        cal_data = list(core.G3File('{}/{}.g3'.format(cal_datapath, obsid_cal)))[0]
        cal_response[wafer][obsid_cal] = []
        cal_rfrac[wafer][obsid_cal] = []
        cal_bololist[wafer][obsid_cal] = []
        
        nep_data = list(core.G3File('{}/{}.g3'.format(nep_datapath, obsid_noise)))[0]
        nep[wafer][obsid_noise] = []
        nep_bololist[wafer][obsid_noise] = []

        for bolo in cal_data["CalibratorResponse"].keys():
            if bps[bolo].band/core.G3Units.GHz == 90 and \
               bps[bolo].wafer_id == wafer and \
               np.isfinite(cal_data["CalibratorResponse"][bolo]) and\
               np.isfinite(cal_data["CalibratorResponseRfrac"][bolo]):
                cal_response[wafer][obsid_cal].append(cal_data["CalibratorResponse"][bolo])
                cal_rfrac[wafer][obsid_cal].append(cal_data["CalibratorResponseRfrac"][bolo])
                cal_bololist[wafer][obsid_cal].append(bolo)
        
        cal_response[wafer][obsid_cal] = np.array(cal_response[wafer][obsid_cal])
        cal_rfrac[wafer][obsid_cal] = np.array(cal_rfrac[wafer][obsid_cal])
        cal_bololist[wafer][obsid_cal] = np.array(cal_bololist[wafer][obsid_cal])
        
        for bolo in nep_data["NEP_10.0Hz_to_15.0Hz"].keys():
            if bps[bolo].band/core.G3Units.GHz == 90 and \
               bps[bolo].wafer_id == wafer and \
               np.isfinite(nep_data["NEP_10.0Hz_to_15.0Hz"][bolo]):
                nep[wafer][obsid_noise].append(nep_data["NEP_10.0Hz_to_15.0Hz"][bolo])
                nep_bololist[wafer][obsid_noise].append(bolo)
        
        nep[wafer][obsid_noise] = np.array(nep[wafer][obsid_noise])
        nep_bololist[wafer][obsid_noise] = np.array(nep_bololist[wafer][obsid_noise])
```

```python
for jwafer,wafer in enumerate(cal_rfrac.keys()):
    common_bolos = reduce(np.intersect1d, cal_bololist[wafer].values())
    random_bolos = np.random.choice(common_bolos, 10, replace=False)

    plt.figure(jwafer)
    for bolo in random_bolos:
        rfracs    = np.array([cal_rfrac[wafer][obsid][cal_bololist[wafer][obsid]==bolo] \
                              for obsid in cal_rfrac[wafer].keys()])
        responses = np.array([cal_response[wafer][obsid][cal_bololist[wafer][obsid]==bolo] \
                              for obsid in cal_response[wafer].keys()])
        plt.plot(rfracs,
                 responses/(core.G3Units.watt*1e-15), 
                 'o-', label='{}'.format(bolo))
    plt.title('{}: {}'.format(wafer, band))
    plt.legend()
    plt.xlabel('rfrac')
    plt.ylabel('(nominal) calibrator response [fW]')
    plt.tight_layout()
    plt.savefig('{}_cal_by_bolo_2.png'.format(wafer), dpi=150)
```

```python
for jwafer, wafer in enumerate(cal_response):
    plt.figure(jwafer)
    
    for obsid in cal_response[wafer]:
        plt.hist(cal_response[wafer][obsid]/(core.G3Units.watt*1e-15),
                 bins=np.linspace(0,3,51),
                 histtype='stepfilled', alpha=0.5,
                 label='rfrac = {:.2f}'.format(np.mean(cal_rfrac[wafer][obsid])))
    plt.title('{}: {}'.format(wafer, band))
    plt.legend()
    plt.xlabel('(nominal) calibrator response [fW]')
    plt.ylabel('bolometers')
    plt.tight_layout()
    plt.savefig('{}_cal_hist_2.png'.format(wafer), dpi=150)
```

```python
for jwafer, wafer in enumerate(nep):
    plt.figure(jwafer)
    
    for obsid_nep, obsid_cal in zip(nep[wafer], cal_response[wafer]):
        plt.hist(nep[wafer][obsid_nep]/(core.G3Units.watt*1e-18 / np.sqrt(core.G3Units.Hz)),
                 bins=np.linspace(0,200,51),
                 histtype='stepfilled', alpha=0.5,
                 label='rfrac = {:.2f}'.format(np.mean(cal_rfrac[wafer][obsid_cal])))
    plt.title('{}: {}'.format(wafer, band))
    plt.legend()
    plt.xlabel('NEP [aW / $\sqrt{Hz}$]')
    plt.ylabel('bolometers')
    plt.tight_layout()
    plt.savefig('{}_nep_2.png'.format(wafer), dpi=150)
```

```python
for jwafer, wafer in enumerate(cal_response):
    plt.figure(jwafer)
    
    for obsid_cal, obsid_noise in zip(obsids_cal[wafer], obsids_noise[wafer]):
        cal_sn = []
        for jbolo, bolo in enumerate(cal_bololist[wafer][obsid_cal]):
            if bolo in nep_bololist[wafer][obsid_noise]:
                cal_sn.append((cal_response[wafer][obsid_cal][jbolo]) / \
                              (nep[wafer][obsid_noise][nep_bololist[wafer][obsid_noise]==bolo][0]/\
                                   (1 / np.sqrt(core.G3Units.Hz))))
                
        plt.hist(cal_sn,
                 bins=np.linspace(0,50,51),
                 histtype='stepfilled', alpha=0.5,
                 label='rfrac = {:.2f}; median S/N = {:.2f}'\
                             .format(np.mean(cal_rfrac[wafer][obsid_cal]),
                                     np.median(cal_sn)))
    plt.title('{}: {}'.format(wafer, band))
    plt.legend()
    plt.xlabel('calibrator S/N [$\sqrt{Hz}$]')
    plt.ylabel('bolometers')
    plt.tight_layout()
    plt.savefig('{}_cal_sn_2.png'.format(wafer), dpi=150)
```

```python
for jwafer, wafer in enumerate(cal_response):
    cal_common_bolos = reduce(np.intersect1d, cal_bololist[wafer].values())
    nep_common_bolos = reduce(np.intersect1d, nep_bololist[wafer].values())
    common_bolos = np.intersect1d(cal_common_bolos, nep_common_bolos)
    random_bolos = np.random.choice(common_bolos, 10, replace=False)
    
    plt.figure(jwafer)
    for bolo in random_bolos:
        rfracs    = np.array([cal_rfrac[wafer][obsid][cal_bololist[wafer][obsid]==bolo] \
                              for obsid in cal_rfrac[wafer].keys()])
        responses = np.array([cal_response[wafer][obsid][cal_bololist[wafer][obsid]==bolo] \
                              for obsid in cal_response[wafer].keys()])
        neps = np.array([nep[wafer][obsid][nep_bololist[wafer][obsid]==bolo] \
                              for obsid in nep[wafer].keys()])
        plt.plot(rfracs, responses / neps * np.sqrt(core.G3Units.sec), 
                 'o-', label='{}'.format(bolo))
    plt.legend()
    plt.title('{}: {}'.format(wafer, band))
    plt.xlabel('rfrac')
    plt.ylabel('calibrator S/N [arb.]')
    plt.tight_layout()
    plt.savefig('{}_cal_sn_by_bolo_2.png'.format(wafer), dpi=150)
```

### Tuning yield
Joshua questioned whether we would latch bolometers by dropping deeper into the transition to gain responsivity. Let's check the detector yield on the wafer/band combos that we tuned. These data are also from the 12/29/2019 test.

```python
pydfmux_dir = '/scratch/pydfmux_output/20191229'
tuning_dirs = {'w206': ['20191229_153549_drop_bolos_tweak_bolos',
                        '20191229_155521_drop_bolos_tweak_bolos',
                        '20191229_161007_drop_bolos_tweak_bolos',
                        '20191229_162444_drop_bolos_tweak_bolos',
                        '20191229_163928_drop_bolos_tweak_bolos'],
               'w174': ['20191229_170540_drop_bolos_tweak_bolos',
                        '20191229_172026_drop_bolos_tweak_bolos',
                        '20191229_173504_drop_bolos_tweak_bolos',
                        '20191229_174928_drop_bolos_tweak_bolos',
                        '20191229_181840_drop_bolos_tweak_bolos']}
```

```python
ntuned = {}
for wafer in tuning_dirs:
    ntuned[wafer] = []
    
    for tuning_dir in tuning_dirs[wafer]:
        tuning_data = pickle.load(open(os.path.join(pydfmux_dir, tuning_dir, 'data/TOTAL_DATA.pkl'), 'rb'))
        
        ntuned[wafer].append(0)
        for mod in tuning_data:
            if mod != 'output_directory' and mod != 'temps': # these are not modules
                for chan in tuning_data[mod]['subtargets']:
                    if wafer in tuning_data[mod]['subtargets'][chan]['physical_name'] and \
                       tuning_data[mod]['subtargets'][chan]['observing_band'] == 90 and \
                       tuning_data[mod]['subtargets'][chan]['state'] == 'tuned':
                        ntuned[wafer][-1] += 1
#                 chans = list(tuning_data[mod]['subtargets'].keys())
#                 if len(chans) != 0 and \
#                    wafer in tuning_data[mod]['subtargets'][chans[0]]['physical_name'] and \
#                    tuning_data[mod]['subtargets'][chans[0]]['observing_band'] == 90:
#                     ntuned[wafer][-1] += 1
```

## w206: 150 GHz, 220 GHz

```python
cal_datapath = '/spt/user/production/calibration/calibrator/'
nep_datapath = '/spt/user/production/calibration/noise/'
obsids_cal   = {'w206': {150: [94501727, 94502604, 94503470, 94504342, 94505245],
                         220: [94506129, 94506993, 94507851, 94508736, 94509601]}}
obsids_noise = {'w206': {150: [94501800, 94502678, 94503543, 94504415, 94505318],
                         220: [94506202, 94507066, 94507924, 94508809, 94509674]}}
bps = list(core.G3File('/spt/data/bolodata/fullrate/calibrator/'
                       '93508217/offline_calibration.g3'))[0]["BolometerProperties"]

cal_response = {}
cal_rfrac = {}
cal_bololist = {}
nep = {}
nep_bololist = {}

for wafer in obsids_cal.keys():
    cal_response[wafer] = {}
    cal_rfrac[wafer] = {}
    cal_bololist[wafer] = {}
    nep[wafer] = {}
    nep_bololist[wafer] = {}
    
    for band in obsids_cal[wafer]:
        cal_response[wafer][band] = {}
        cal_rfrac[wafer][band] = {}
        cal_bololist[wafer][band] = {}
        nep[wafer][band] = {}
        nep_bololist[wafer][band] = {}
    
        for obsid_cal, obsid_noise in zip(obsids_cal[wafer][band], obsids_noise[wafer][band]):
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
for jwafer, wafer in enumerate(cal_response):
    for jband, band in enumerate(list(cal_response[wafer].keys())):
        plt.figure(jwafer + 100*jband)

        for obsid in cal_response[wafer][band]:
            plt.hist(cal_response[wafer][band][obsid]/(core.G3Units.watt*1e-15),
                     bins=np.linspace(0,15,51),
                     histtype='stepfilled', alpha=0.5,
                     label='rfrac = {:.2f}'.format(np.mean(cal_rfrac[wafer][band][obsid])))
        plt.title('{}: {}'.format(wafer, band))
        plt.legend()
        plt.xlabel('(nominal) calibrator response [fW]')
        plt.ylabel('bolometers')
        plt.tight_layout()
        plt.savefig('{}_{}_cal_hist_3.png'.format(wafer, band), dpi=150)
```

```python
for jwafer, wafer in enumerate(nep):
    for jband, band in enumerate(list(cal_response[wafer].keys())):
        plt.figure(jwafer + 100*jband)

        for obsid_nep, obsid_cal in zip(nep[wafer][band], cal_response[wafer][band]):
            plt.hist(nep[wafer][band][obsid_nep]/(core.G3Units.watt*1e-18 / np.sqrt(core.G3Units.Hz)),
                     bins=np.linspace(0,300,51),
                     histtype='stepfilled', alpha=0.5,
                     label='rfrac = {:.2f}'.format(np.mean(cal_rfrac[wafer][band][obsid_cal])))
        plt.title('{}: {}'.format(wafer, band))
        plt.legend()
        plt.xlabel('NEP [aW / $\sqrt{Hz}$]')
        plt.ylabel('bolometers')
        plt.tight_layout()
        plt.savefig('{}_{}_nep_3.png'.format(wafer, band), dpi=150)
```

```python
for jwafer, wafer in enumerate(cal_response):
    for jband, band in enumerate(list(cal_response[wafer].keys())):
        plt.figure(jwafer + 100*jband)

        for obsid_cal, obsid_noise in zip(obsids_cal[wafer][band], obsids_noise[wafer][band]):
            cal_sn = []
            for jbolo, bolo in enumerate(cal_bololist[wafer][band][obsid_cal]):
                if bolo in nep_bololist[wafer][band][obsid_noise]:
                    cal_sn.append((cal_response[wafer][band][obsid_cal][jbolo]) / \
                                  (nep[wafer][band][obsid_noise][nep_bololist[wafer][band][obsid_noise]==bolo][0]/\
                                       (1 / np.sqrt(core.G3Units.Hz))))

            plt.hist(cal_sn,
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
        plt.savefig('{}_{}_cal_sn_3.png'.format(wafer, band), dpi=150)
```

## Full array tests

```python
cal_datapath = '/spt/user/production/calibration/calibrator/'
nep_datapath = '/spt/user/production/calibration/noise/'
obsids_cal   = {90: [94510526, 94511474, 94512391, 94513312, 94514261],
                150: [94719976, 94720953, 94721928, 94723114, 94724218],
                220: [95115583, 95116529, 95117446, 95118373, 95119272]}
obsids_noise = {90: [94510599, 94511547, 94512464, 94513385, 94514334],
                150: [94720049, 94721026, 94722001, 94723187, 94724291],
                220: [95115656, 95116602, 95117519, 95118446, 95119345]}
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
        plt.savefig('{}_cal_hist_4.png'.format(band), dpi=150)
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
        plt.savefig('{}_nep_4.png'.format(band), dpi=150)
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
        plt.savefig('{}_cal_sn_4.png'.format(band), dpi=150)
```

```python

```
