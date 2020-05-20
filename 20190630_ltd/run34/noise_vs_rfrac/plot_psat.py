import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pydfmux.analysis.analyze_IV as AIV

fnames_dark = glob('/daq/pydfmux_output/20181014/20181014_192147_drop_bolos/data/*pkl')
fnames_optical = glob('/daq/pydfmux_output/20181013/20181013_040726_drop_bolos/data/*pkl')

psat_dark = dict()
for fname in fnames_dark:
    d = AIV.find_Psat(fname, physical_names=True)
    for bolo in d:
        psat_dark[bolo] = d[bolo]

psat_optical = dict()
for fname in fnames_optical:
    d = AIV.find_Psat(fname, physical_names=True)
    for bolo in d:
        psat_optical[bolo] = d[bolo]

poptical = dict()
for bolo in psat_dark:
    if bolo in psat_optical:
        poptical[bolo] = psat_dark[bolo] - psat_optical[bolo]

plt.figure()
mean_poptical = dict()
for jband, band in enumerate([90, 150, 220]):
    popt_plot = np.array([poptical[bolo] for bolo in poptical
                          if int(bolo.split('.')[1]) == band])
    plt.hist(popt_plot*1e12, bins=np.linspace(0,10,31),
             histtype='stepfilled', label='{} GHz'.format(band), alpha=0.5,
             color='C{}'.format(jband))
    plt.hist(popt_plot*1e12, bins=np.linspace(0,10,31),
             histtype='step', color='C{}'.format(jband))
    mean_poptical[band] = np.median(popt_plot[np.isfinite(popt_plot)]*1e12)
plt.legend()
plt.xlim([0,10])
plt.title('Poptical = {:.1f} / {:.1f} / {:.1f} pW (90 / 150 / 220)'.format(mean_poptical[90],
                                                               mean_poptical[150],
                                                               mean_poptical[220]))
plt.tight_layout()
plt.savefig('Poptical_coldload.png', dpi=200)


plt.figure()
for jband, band in enumerate([90, 150, 220]):
    popt_plot = np.array([psat_dark[bolo] for bolo in psat_dark
                          if int(bolo.split('.')[1]) == band])
    plt.hist(popt_plot*1e12, bins=np.linspace(0,20,31),
             histtype='stepfilled', label='{} GHz'.format(band), alpha=0.5,
             color='C{}'.format(jband))
    plt.hist(popt_plot*1e12, bins=np.linspace(0,20,31),
             histtype='step', color='C{}'.format(jband))
plt.legend()
plt.xlim([0,20])
plt.title('dark Psat (UC stage @ 350mK)')
plt.xlabel('Psat [pW]')
plt.tight_layout()
plt.savefig('Psat_dark.png', dpi=200)


plt.figure()
for jband, band in enumerate([90, 150, 220]):
    popt_plot = np.array([psat_optical[bolo] for bolo in psat_optical
                          if int(bolo.split('.')[1]) == band])
    plt.hist(popt_plot*1e12, bins=np.linspace(0,20,31),
             histtype='stepfilled', label='{} GHz'.format(band), alpha=0.5,
             color='C{}'.format(jband))
    plt.hist(popt_plot*1e12, bins=np.linspace(0,20,31),
             histtype='step', color='C{}'.format(jband))
plt.legend()
plt.xlim([0,20])
plt.title('optical Psat (UC stage @ 350mK)')
plt.xlabel('Psat [pW]')
plt.tight_layout()
plt.savefig('Psat_optical.png', dpi=200)
