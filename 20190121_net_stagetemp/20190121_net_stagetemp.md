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
from adama_utils import plot_noise, plot_noise_comparison
from spt3g import core, dfmux, calibration
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
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
            64701072: {'UC head temp': 0.300, 'elevation': 'high'},
            64716070: {'UC head temp': 0.270, 'elevation': 'high'},
            65041359: {'UC head temp': 0.270, 'elevation': 'high'},
            65106264: {'UC head temp': 0.270, 'elevation': 'high'},
            65118448: {'UC head temp': 0.270, 'elevation': 'high'},
            65134617: {'UC head temp': 0.310, 'elevation': 'high'},
            65146903: {'UC head temp': 0.310, 'elevation': 'high'}}
obsids = list(metadata.keys())
```

```python
for obsid in obsids:
    new_noise_file = 'noise_corrected/{}_processed_noise.g3'.format(obsid)
    offlinecal_fname = '/spt/data/bolodata/downsampled/noise/' + \
                       '{}/offline_calibration.g3' .format(obsid)
    if os.path.exists(offlinecal_fname) and \
       os.path.exists(new_noise_file):
        d = [fr for fr in core.G3File(new_noise_file)]
        bps = [fr for fr in core.G3File(offlinecal_fname)][0]["BolometerProperties"]
        _ = plot_noise(d[0], bps, obsid, bywafer=False,
                       filestub='NET_{}_el'.format(metadata[obsid]['elevation']))
    plt.close('all')
   
data = {}
for obsid in obsids:
    new_noise_file = 'noise_corrected/{}_processed_noise.g3'.format(obsid)
    offlinecal_fname = '/spt/data/bolodata/downsampled/noise/' + \
                       '{}/offline_calibration.g3' .format(obsid)
    if os.path.exists(offlinecal_fname) and \
       os.path.exists(new_noise_file):
        d = [fr for fr in core.G3File(new_noise_file)]
        bps = [fr for fr in core.G3File(offlinecal_fname)][0]["BolometerProperties"]
        data[obsid] = plot_noise(d[0], bps, obsid, bywafer=True,
                                 filestub='NET_bywafer_{}_el'.format(metadata[obsid]['elevation']))
    plt.close('all')
```

```python
# useful definitions for summary statistics
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206']
net_low = {90:300, 150:200, 220:800}
```

```python
for band in [90, 150, 220]:
    for el in ['low', 'high']:
        print('{} ELEVATION \ {} GHz\t'.format(el.upper(), band) + '\t'.join(wafer_list) + '\tall')
        for obsid in data:
            if metadata[obsid]['elevation'] == el:
                all_nets = np.array([])
                dstring = '{} ({:.3f}):\t'.format(obsid, metadata[obsid]['UC head temp'])
                for wafer in wafer_list:
                    nets = data[obsid][wafer][band]
                    dstring = dstring + '{:.1f}\t'.format(np.median(data[obsid][wafer][band]))
                    all_nets = np.append(all_nets, nets[nets>net_low[band]])
                dstring = dstring + '{:.1f}'.format(np.median(all_nets))
                print(dstring)
        print('')
        
# write to file with formatting for wiki
with open('net_per_bolo_wiki.txt', 'w') as f:
    for band in [90, 150, 220]:
        for el in ['low', 'high']:
            f.write('{|border="1" width=900px\n')
            wafer_list_str = ['\'\'\'' + w + '\'\'\'' for w in wafer_list]
            f.write('!colspan="12"| ' + '{} ELEVATION / '.format(el.upper()) +'{} GHz \n|-\n'.format(band))
            f.write('| \'\'\'obsid (UC head temp)\'\'\'\n|' + '\n| '.join(wafer_list_str) + '\n| \'\'\'all\'\'\'\n|-')
            for obsid in data:
                if metadata[obsid]['elevation'] == el:
                    all_nets = np.array([])
                    dstring = '\n| {} ({:.3f})'.format(obsid, metadata[obsid]['UC head temp'])
                    for wafer in wafer_list:
                        nets = data[obsid][wafer][band]
                        dstring = dstring + '\n|{:.1f}'.format(np.median(data[obsid][wafer][band]))
                        all_nets = np.append(all_nets, nets[nets>net_low[band]])
                    dstring = dstring + '\n|{:.1f}'.format(np.median(all_nets)) + '\n|-\n'
                    f.write(dstring)
            f.write('|}\n\n\n')
```

```python
for el in ['low', 'high']:
    print('{} ELEVATION'.format(el.upper()))
    for band in [90, 150, 220]:
        print('{} GHz\t\t\t'.format(band) + '\t'.join(wafer_list) + '\tall')
        for obsid in data:
            if metadata[obsid]['elevation'] == el:
                dstring = '{} ({:.3f}):\t'.format(obsid, metadata[obsid]['UC head temp'])
                all_nets = np.array([])
                for wafer in wafer_list:
                    nets = data[obsid][wafer][band]
                    total_net = np.sqrt(1.0 / np.sum(1.0 / nets[nets>net_low[band]]**2.0))
                    dstring = dstring + '{:.1f}\t'.format(total_net)
                    all_nets = np.append(all_nets, nets[nets>net_low[band]])
                total_net = np.sqrt(1.0 / np.sum(1.0 / all_nets**2.0))
                dstring = dstring + '{:.1f}'.format(total_net)
                print(dstring)
        print('')
        
        
# write to file with formatting for wiki
with open('net_per_wafer_wiki.txt', 'w') as f:
    for band in [90, 150, 220]:
        for el in ['low', 'high']:
            f.write('{|border="1" width=900px\n')
            wafer_list_str = ['\'\'\'' + w + '\'\'\'' for w in wafer_list]
            f.write('!colspan="12"| ' + '{} ELEVATION / '.format(el.upper()) +'{} GHz \n|-\n'.format(band))
            f.write('| \'\'\'obsid (UC head temp)\'\'\'\n|' + '\n| '.join(wafer_list_str) + '\n| \'\'\'all\'\'\'\n|-')
            for obsid in data:
                if metadata[obsid]['elevation'] == el:
                    all_nets = np.array([])
                    dstring = '\n| {} ({:.3f})'.format(obsid, metadata[obsid]['UC head temp'])
                    for wafer in wafer_list:
                        nets = data[obsid][wafer][band]
                        total_net = np.sqrt(1.0 / np.sum(1.0 / nets[nets>net_low[band]]**2.0))
                        dstring = dstring + '\n|{:.1f}'.format(total_net)
                        all_nets = np.append(all_nets, nets[nets>net_low[band]])
                        total_net = np.sqrt(1.0 / np.sum(1.0 / all_nets**2.0))
                    dstring = dstring + '\n|{:.1f}'.format(np.median(total_net)) + '\n|-\n'
                    f.write(dstring)
            f.write('|}\n\n\n')
```

```python
# overlay observations by wafer and band
obsid_pairs = [[64576620, 63084862],
               [64576620, 63661173],
               [64576620, 64591411],
               [63084862, 64591411],
               [64576620, 64606397]]
labels = [['64576620 (head = 269mK)', '63084862 (head = 280mK)'],
          ['64576620 (head = 269mK)', '63661173 (head = 275mK)'],
          ['64576620 (head = 269mK)', '64591411 (head = 300mK)'],
          ['63084862 (head = 280mK)', '64591411 (head = 300mK)'],
          ['64576620 (head = 269mK)', '64606397 (head = 300mK)']]

for jpair, obsids in enumerate(obsid_pairs):
    print(obsids)
    frame_list = []
    boloprops_list = []
    for oid in obsids:
        d = [fr for fr in core.G3File('noise_corrected/{}_processed_noise.g3'
                                      .format(oid))][0]
        frame_list.append(d)
        bps = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/{}/offline_calibration.g3'
                                       .format(oid))][0]["BolometerProperties"]
        boloprops_list.append(bps)
    plot_noise_comparison(frame_list, boloprops_list, obsids, bywafer=True, legend_labels=labels[jpair],
                          filestub='NETvTemp')
plt.close('all')
```

```python
len(all_nets)
```

```python
26 / 3.
```

```python
print('\'\'\'')
```

```python
# W203
for obsid in obsids:
    new_noise_file = 'noise_corrected/{}_processed_noise.g3'.format(obsid)
    if os.path.exists(new_noise_file):
        d = [fr for fr in core.G3File(new_noise_file)]
        bps = [fr for fr in core.G3File('/spt/data/bolodata/fullrate/noise/{}/offline_calibration.g3'
                                        .format(obsid))][0]["BolometerProperties"]
        _ = plot_noise(d[0], bps, obsid, bywafer=False, wafer='w203',
                       filestub='NET_{}_{}_el'.format(metadata[obsid]['elevation'], 'w203'))
    plt.close('all')
```

```python
d = [fr for fr in core.G3File('noise_corrected/{}_processed_noise.g3'.format(63098693))]
data = plot_noise(d[0], bps, 63098693, bywafer=False,
                  filestub='NET_{}_{}_el'.format(metadata[63098693]['elevation'], 'w203'))
```

```python
matplotlib.rcParams.update({'font.size': 13})
for jband, band in enumerate([90, 150, 220]):
    noise = data[band]
    plt.hist(noise[noise>200], bins=np.linspace(0, 2500, 61),
             histtype='stepfilled',
             alpha=0.5, color='C{}'.format(jband), linewidth=1.5)
    plt.hist(noise[noise>200], bins=np.linspace(0, 2500, 61),
             label='{} GHz'.format(band), histtype='step',
             color='C{}'.format(jband), linewidth=1.5)
plt.xlim([0, 2500])
plt.legend(frameon=False)
plt.xlabel('NET [$\mu K \sqrt{s}$]')
plt.ylabel('bolometers')
plt.setp(list(plt.gca().spines.values()), linewidth=1.5)
plt.gca().xaxis.set_tick_params(width=1.5)
plt.gca().yaxis.set_tick_params(width=1.5)
plt.tight_layout()
plt.savefig('net_update.png', dpi=200)
```

## Some plots by wafer for the March F2F

```python
# only works with matplotlib version in v3 clustertools

for obsid in data:
    ...:     plt.figure(1, figsize=(10,4))
    ...:     for band in [90, 150, 220]:
    ...:         all_nets = np.array([])
    ...:         wafers_plot = []
    ...:         nets_plot = []
    ...:         nets_1sigmalow = []
    ...:         nets_1sigmahigh = []
    ...:         try:
    ...:             for wafer in wafer_list:
    ...:                 nets = data[obsid][wafer][band]
    ...:                 nets_plot.append(np.median(nets))
    ...:                 nets_1sigmalow.append(np.median(nets) - np.percentile(nets, 16))
    ...:                 nets_1sigmahigh.append(np.percentile(nets, 84) - np.median(nets))
    ...:                 wafers_plot.append(wafer)
    ...:             plt.errorbar(nets_plot, wafers_plot, xerr=np.array([nets_1sigmalow, nets_1sigmahigh]), linestyle='None', marker='o', capsize=2, capthick=2)
    ...:         except:
    ...:             pass
    ...:     plt.xlim([0, 2500])
    ...:     plt.xlabel('NET [$\mu$K$\sqrt{s}$]')
    ...:     plt.ylabel('wafer')
    ...:     plt.title(obsid)
    ...:     plt.tight_layout()
    ...:     plt.savefig('net_wafer_summary_{}.png'.format(obsid), dpi=200)
    ...:     plt.close()
```

```python
import pickle
with open('net_skim.pkl', 'wb') as f:
    pickle.dump(data, f)
```

```python
data_sept_2018 = {}
obsids_sept_2018 = [fname.split('/')[-1] for fname in glob('/spt/data/bolodata/downsampled/noise/53*')]
for obsid in obsids_sept_2018:
    noise_file = '/spt/user/production/calibration/noise/{}.g3' .format(obsid)
    offlinecal_fname = '/spt/data/bolodata/downsampled/noise/' + \
                       '{}/offline_calibration.g3' .format(obsid)
    if os.path.exists(offlinecal_fname) and \
       os.path.exists(noise_file):
        d = [fr for fr in core.G3File(noise_file)]
        bps = [fr for fr in core.G3File(offlinecal_fname)][0]["BolometerProperties"]
        data_sept_2018[obsid] = plot_noise(d[0], bps, obsid, bywafer=False,
                                 filestub='NET_')
    plt.close('all')
    

```

```python
new_noise_file = 'noise_corrected/63689042_processed_noise.g3'.format(obsid)
offlinecal_fname = '/spt/data/bolodata/fullrate/noise/63689042/offline_calibration.g3'
bps = [fr for fr in core.G3File(offlinecal_fname)][0]["BolometerProperties"]
d = [fr for fr in core.G3File(new_noise_file)]
data_2019 = plot_noise(d[0], bps, 63098693, bywafer=False,
                       filestub='NET_63689042_')
```

```python
matplotlib.rcParams.update({'font.size': 13})
for jband, band in enumerate([90, 150, 220]):
    noise_2018 = data_sept_2018['53614387'][band]
    noise_2019 = data_2019[band]
#     plt.hist(noise[noise>200], bins=np.linspace(0, 2500, 61),
#              histtype='stepfilled',
#              alpha=0.5, color='C{}'.format(jband), linewidth=1.5)
    plt.hist(noise_2018[noise_2018>200], bins=np.linspace(0, 2500, 61),
             label='{} GHz (Sept. 2018)'.format(band), histtype='step',
             color='C{}'.format(jband), linewidth=1.5, linestyle='--')
    plt.hist(noise_2019[noise_2019>200], bins=np.linspace(0, 2500, 61),
             label='{} GHz (Jan. 2019)'.format(band), histtype='step',
             color='C{}'.format(jband), linewidth=1.5)
plt.xlim([0, 2500])
plt.legend(frameon=False)
plt.xlabel('NET [$\mu K \sqrt{s}$]')
plt.ylabel('bolometers')
plt.setp(list(plt.gca().spines.values()), linewidth=1.5)
plt.gca().xaxis.set_tick_params(width=1.5)
plt.gca().yaxis.set_tick_params(width=1.5)
plt.tight_layout()
plt.savefig('net_2018_vs_2019.png', dpi=200)
```

```python
matplotlib.rcParams.update({'font.size': 13})
for jband, band in enumerate([90, 150, 220]):
    noise_2018 = data_sept_2018['53614387'][band]
    noise_2019 = data_2019[band]
#     plt.hist(noise[noise>200], bins=np.linspace(0, 2500, 61),
#              histtype='stepfilled',
#              alpha=0.5, color='C{}'.format(jband), linewidth=1.5)
    plt.hist(noise_2018[noise_2018>200], bins=np.linspace(0, 2500, 121),
             label='{} GHz (Sept. 2018)'.format(band), histtype='step',
             color='C{}'.format(jband), linewidth=1.5, linestyle='--')
    plt.hist(noise_2019[noise_2019>200], bins=np.linspace(0, 2500, 121),
             label='{} GHz (Jan. 2019)'.format(band), histtype='step',
             color='C{}'.format(jband), linewidth=1.5)
plt.xlim([200, 1250])
plt.legend(frameon=False)
plt.xlabel('NET [$\mu K \sqrt{s}$]')
plt.ylabel('bolometers')
plt.setp(list(plt.gca().spines.values()), linewidth=1.5)
plt.gca().xaxis.set_tick_params(width=1.5)
plt.gca().yaxis.set_tick_params(width=1.5)
plt.tight_layout()
plt.savefig('net_2018_vs_2019_zoom.png', dpi=200)
```

```python

```
