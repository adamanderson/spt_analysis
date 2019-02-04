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
from spt3g import core, dfmux, calibration
import numpy as np
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.constants as const
```

The purpose of this note is to check whether the voltage bias applied to detectors when the stage temperature is elevated by the PID control loop is actually lower than in normal operation. The reason to want to check this is that our noise does not appear to be substantially better with the elevated stage temperature. It may be that we are simply trading off loopgain for readout noise, and the NET is basically unchanged.

```python
with open('vbias.pkl', 'rb') as f:
    data = pickle.load(f)
```

```python
metadata = {63084862: {'UC head temp': 0.280, 'elevation': 'high'}, # 1 Jan overnight
            63098693: {'UC head temp': 0.300, 'elevation': 'high'},
            63218920: {'UC head temp': 0.300, 'elevation': 'high'},
            63227942: {'UC head temp': 0.300, 'elevation': 'high'},
            63305224: {'UC head temp': 0.300, 'elevation': 'low'},
            63380372: {'UC head temp': 0.300, 'elevation': 'low'},
            63640406: {'UC head temp': 0.300, 'elevation': 'low'},
            63650590: {'UC head temp': 0.300, 'elevation': 'low'},
            63661173: {'UC head temp': 0.275, 'elevation': 'low'},
            63689042: {'UC head temp': 0.300, 'elevation': 'high'},
            63728180: {'UC head temp': 0.300, 'elevation': 'high'},
            64576620: {'UC head temp': 0.269, 'elevation': 'low'},
            64591411: {'UC head temp': 0.300, 'elevation': 'low'},
            64606397: {'UC head temp': 0.300, 'elevation': 'low'},
            64685912: {'UC head temp': 0.300, 'elevation': 'high'},
            64701072: {'UC head temp': 0.300, 'elevation': 'high'}}
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']
```

```python
plt.figure(figsize=(16,16))
jobs = 1
for obs in data:
    plt.subplot(4,4,jobs)
    if 'vbiasrms' in data[obs]:
        vbias = np.array([data[obs]['vbiasrms'][bolo]*1e6
                          for bolo in data[obs]['vbiasrms'].keys()])
        _ = plt.hist(vbias, bins=np.linspace(0,6,41))
        plt.xlim([0, 6])
    jobs += 1
```

Split by band and wafer.

```python
obsids_plot = {}
vbias_plot = {}
 
for obs in data:
    if 'vbiasrms' in data[obs]:
        bps = data[obs]['boloprops']
        bands = np.array([bps[bolo].band / core.G3Units.GHz for bolo in data[obs]['vbiasrms'].keys()])
        wafers = np.array([bps[bolo].wafer_id for bolo in data[obs]['vbiasrms'].keys()])
        vbias = np.array([data[obs]['vbiasrms'][bolo]*1e6 for bolo in data[obs]['vbiasrms'].keys()])
            
        for wafer in wafer_list:
            if wafer not in obsids_plot:
                obsids_plot[wafer] = {}
            if wafer not in vbias_plot:
                vbias_plot[wafer] = {}
            for band in [90, 150, 220]:
                if band not in obsids_plot[wafer]:
                    obsids_plot[wafer][band] = np.array([])
                if band not in vbias_plot[wafer]:
                    vbias_plot[wafer][band] = np.array([])
                obsids_plot[wafer][band] = np.append(obsids_plot[wafer][band], obs)
                vbias_plot[wafer][band] = np.append(vbias_plot[wafer][band],
                                                    np.median(vbias[(bands==band) & (wafers==wafer) & (vbias>0.1)]))
                
for jfig, band in enumerate([90, 150, 220]):
    plt.figure(jfig, figsize=(8,6))
    for wafer in wafer_list:
        oplot = [o for o in obsids_plot[wafer][band] if metadata[o]['UC head temp']==0.300]
        vplot = [v for o, v in zip(obsids_plot[wafer][band], vbias_plot[wafer][band])
                 if metadata[o]['UC head temp']==0.300]
        plt.plot(oplot, vplot, 'o-', label=wafer)
    plt.legend()
    plt.xlabel('observation ID')
    plt.ylabel('voltage bias [uVrms]')
```

Split by stage temperature.

```python
plt.figure(figsize=(24,8))
for jfig, band in enumerate([90, 150, 220]):
    for jwafer, wafer in enumerate(wafer_list):
        plt.subplot(2,5,jwafer+1)
        oplot = [o for o in obsids_plot[wafer][band] if metadata[o]['UC head temp']==0.300 and \
                 metadata[o]['elevation']=='low']
        vplot = [v for o, v in zip(obsids_plot[wafer][band], vbias_plot[wafer][band])
                 if metadata[o]['UC head temp']==0.300 and metadata[o]['elevation']=='low']
        plt.plot(oplot, vplot, 'o-', label='300mK, {}'.format(band), color='C{}'.format(jfig))
        
        oplot = [o for o in obsids_plot[wafer][band] if metadata[o]['UC head temp']<0.300 and \
                 metadata[o]['elevation']=='low']
        vplot = [v for o, v in zip(obsids_plot[wafer][band], vbias_plot[wafer][band])
                 if metadata[o]['UC head temp']<0.300 and metadata[o]['elevation']=='low']
        plt.plot(oplot, vplot, 'o--', label='270mK, {}'.format(band), color='C{}'.format(jfig))
        plt.ylim([0,5])
        plt.title(wafer)
        plt.xlabel('observation ID')
        plt.ylabel('voltage bias [uVrms]')
        plt.legend()
    plt.xlabel('observation ID')
    plt.ylabel('voltage bias [uVrms]')
    plt.tight_layout()
    plt.savefig('VbiasVobsid.png'.format(band),dpi=200)
```

```python
R = 1.5
Tc = 0.450
Tb280 = 0.280
n = 3.0

Tb = np.linspace(0.280, 0.400, 100)
Popt = {90:4, 150:8, 220:9}
NEPph = {}
bandwidth = {90:25e9, 150:35e9, 220:45e9}
for band in [90, 150, 220]:
    NEPph[band] = np.sqrt(2.* const.h * band*1e9 * Popt[band]*1e-12 + \
                          2.*(Popt[band]*1e-12)**2.0 / bandwidth[band]) * 1e18
Psat280_byband = {90:[6,8,10,12,14], 150:[10,12,14,16,18], 220:[11,13,15,17,19]}
In = {90:12, 150:14, 220:17}

plt.figure(1, figsize=(15,4))
for jband, band in enumerate([90,150,220]):
    for Psat280 in Psat280_byband[band]:
        plt.subplot(1,3,jband+1)
        Vbias = np.sqrt(R * (Psat280 / (Tc**n - Tb280**n) * (Tc**n - Tb**n) - Popt[band]))
        plt.plot(Tb*1e3, Vbias, label='Psat = {} pW'.format(Psat280))
        plt.title('Typical {} GHz, Popt = {} pW'.format(band, Popt[band]))
        plt.xlabel('base temperature [mK]')
        plt.ylabel('voltage bias [Vrms]')
    plt.plot([280, 280], [0, 4.0], 'k--')
    plt.plot([310, 310], [0, 4.0], 'k--')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('figures/Vbias_vs_baseT.png', dpi=200)

plt.figure(2, figsize=(15,4))
for jband, band in enumerate([90,150,220]):
    for Psat280 in Psat280_byband[band]:
        plt.subplot(1,3,jband+1)
        Vbias280 = np.sqrt(R * (Psat280 - Popt[band]))
        Vbias = np.sqrt(R * (Psat280 / (Tc**n - Tb280**n) * (Tc**n - Tb**n) - Popt[band]))
        plt.plot(Tb*1e3, Vbias - Vbias280, label='Psat = {} pW'.format(Psat280))
        plt.title('Typical {} GHz, Popt = {} pW'.format(band, Popt[band]))
        plt.xlabel('base temperature [mK]')
        plt.ylabel('change in voltage bias [Vrms]')
    plt.plot([280, 280], [-3.0, 0], 'k--')
    plt.plot([310, 310], [-3.0, 0], 'k--')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('figures/deltaVbias_vs_baseT.png', dpi=200)

plt.figure(3, figsize=(15,4))
for jband, band in enumerate([90,150,220]):
    for Psat280 in Psat280_byband[band]:
        plt.subplot(1,3,jband+1)
        Vbias = np.sqrt(R * (Psat280 / (Tc**n - Tb280**n) * (Tc**n - Tb**n) - Popt[band]))
        plt.plot(Tb*1e3,  In[band] * Vbias / np.sqrt(2.0), label='Psat = {} pW'.format(Psat280))
        plt.title('Typical {} GHz, Popt = {} pW'.format(band, Popt[band]))
        plt.xlabel('base temperature [mK]')
        plt.ylabel('readout NEP [aW/rtHz]')
    plt.plot([280, 280], [0, 50], 'k--')
    plt.plot([310, 310], [0, 50], 'k--')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('figures/NEPreadout_vs_baseT.png', dpi=200)
    
plt.figure(4, figsize=(15,4))
for jband, band in enumerate([90,150,220]):
    for Psat280 in Psat280_byband[band]:
        plt.subplot(1,3,jband+1)
        Vbias = np.sqrt(R * (Psat280 / (Tc**n - Tb280**n) * (Tc**n - Tb**n) - Popt[band]))
        plt.plot(Tb*1e3,  np.sqrt((In[band] * Vbias / np.sqrt(2.0))**2.0 + \
                                  NEPph[band]**2.0), label='Psat = {} pW'.format(Psat280))
        plt.title('Typical {} GHz, Popt = {} pW'.format(band, Popt[band]))
        plt.xlabel('base temperature [mK]')
        plt.ylabel('total NEP [aW/rtHz]')
    plt.plot(Tb*1e3, NEPph[band]*np.ones(Tb.shape), 'k--', label='photon NEP')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('figures/NEPtotal_vs_baseT.png', dpi=200)
    
plt.figure(5, figsize=(15,4))
for jband, band in enumerate([90,150,220]):
    for Psat280 in Psat280_byband[band]:
        plt.subplot(1,3,jband+1)
        Vbias = np.sqrt(R * (Psat280 / (Tc**n - Tb280**n) * (Tc**n - Tb**n) - Popt[band]))
        plt.plot(Tb*1e3,  np.sqrt((In[band] * Vbias / np.sqrt(2.0))**2.0 + \
                                  NEPph[band]**2.0) / NEPph[band], label='Psat = {} pW'.format(Psat280))
        plt.title('Typical {} GHz, Popt = {} pW'.format(band, Popt[band]))
        plt.xlabel('base temperature [mK]')
        plt.ylabel('NEP / NEP_photon')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('figures/NEPphoton_ratio_vs_baseT.png', dpi=200)
    
plt.figure(6, figsize=(15,4))
for jband, band in enumerate([90,150,220]):
    for Psat280 in Psat280_byband[band]:
        plt.subplot(1,3,jband+1)
        Vbias = np.sqrt(R * (Psat280 / (Tc**n - Tb280**n) * (Tc**n - Tb**n) - Popt[band]))
        NEP_total = np.sqrt((In[band] * Vbias / np.sqrt(2.0))**2.0 + \
                                  NEPph[band]**2.0)
        plt.plot(Tb*1e3, NEP_total / NEP_total[0], label='Psat = {} pW'.format(Psat280))
        plt.title('Typical {} GHz, Popt = {} pW'.format(band, Popt[band]))
        plt.xlabel('base temperature [mK]')
        plt.ylabel('NEP(T) / NEP(280mK)')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('figures/NEP280mKratio_vs_baseT.png', dpi=200)
```

```python
NEPph
```

```python
np.sqrt(1.1170770428799998e-33)*1e18
```

```python
np.sqrt(25**2 - (18*0.8)**2.) / 25.
```

```python
np.sqrt(25**2 - (18)**2.)/0.8 / 25.
```

```python
A
```
