from spt3g import core, calibration, dfmux
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os.path
import pickle

fnames = np.sort(glob('opteff_stdproc_*.g3'))
obsid_list = []
opteff_quantiles = {90: {}, 150: {}, 220: {}}
opteff_all = {}

for fname in fnames:
    d = [fr for fr in core.G3File(fname)]
    print(d[0])
    obsid = os.path.splitext(fname)[0].split('_')[-1]
    print(obsid)

    bp = d[0]["BolometerProperties"]

    # histograms by observation
    plt.figure()
    for band in [90, 150, 220]:
        opteff = np.array([d[0]["opteff"][bolo] for bolo in d[0]["opteff"].keys() \
                      if bp[bolo].band / core.G3Units.GHz==band])
        plt.hist(opteff[np.isfinite(opteff)], bins=np.linspace(0,0.6,61),
                 label='{} GHz'.format(band),
                 histtype='step')
    plt.legend()
    plt.xlabel('optical efficiency')
    plt.title(obsid)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.splitext(fname)[0] + '.png', dpi=200)
    plt.close()


    # make list of all optical efficiencies
    for bolo in d[0]['opteff'].keys():
        if bolo not in opteff_all.keys():
            opteff_all[bolo] = []
        opteff_all[bolo].append(d[0]['opteff'][bolo])


    # focal plane plots
    for band in [90, 150, 220]:
        for pol in ['x', 'y']:
            bp = d[0]["BolometerProperties"]

            plt.figure()
            xpos = [bp[bolo].x_offset / core.G3Units.deg
                    for bolo in d[0]['opteff'].keys()
                    if bp[bolo].band/core.G3Units.GHz==band and \
                        bp[bolo].physical_name.split('.')[-1]==pol]
            ypos = [bp[bolo].y_offset / core.G3Units.deg
                    for bolo in d[0]['opteff'].keys()
                    if bp[bolo].band/core.G3Units.GHz==band and \
                        bp[bolo].physical_name.split('.')[-1]==pol]
            opteff = np.array([d[0]['opteff'][bolo]
                      for bolo in d[0]['opteff'].keys()
                      if bp[bolo].band/core.G3Units.GHz==band and \
                        bp[bolo].physical_name.split('.')[-1]==pol])

            plt.scatter(xpos, ypos, c=opteff,
                        vmin=np.percentile(opteff,5),
                        vmax=np.percentile(opteff,90))
            plt.colorbar()
            plt.xlabel('x offset [deg]')
            plt.ylabel('y offset [deg]')
            plt.title('{}: {} GHz {}-pol'.format(obsid, band, pol))
            plt.tight_layout()
            plt.savefig('opteff_focalplane_{}_{}_{}.png'.format(obsid, band, pol), dpi=200)
            plt.close()

    for band in [90, 150, 220]:
        opteff_quantiles[band][obsid] = (np.percentile(opteff[np.isfinite(opteff) & (opteff>0.02) & (opteff<0.8)], 16), 
                                         np.median(opteff[np.isfinite(opteff) & (opteff>0.02) & (opteff<0.8)]),
                                         np.percentile(opteff[np.isfinite(opteff) & (opteff>0.02) & (opteff<0.8)], 84))

# quantiles by observation
plt.figure(figsize=(6,10))
rgbcolors = ['b', 'g', 'r']
for jband, band in enumerate([90, 150, 220]):
    obslist = np.sort(list(opteff_quantiles[band].keys()))
    ymed = np.array([opteff_quantiles[band][obs][1] for obs in obslist])
    yerrlo = np.array([opteff_quantiles[band][obs][0] for obs in obslist])
    yerrhi = np.array([opteff_quantiles[band][obs][2] for obs in obslist])

    plt.subplot(3,1,jband+1)
    plt.errorbar(np.arange(len(fnames)),
                 ymed,
                 yerr=np.array([ymed-yerrlo, yerrhi-ymed]),
                 label='{} GHz'.format(band),
                 linestyle='None',
                 marker='o',
                 color=rgbcolors[jband])
    plt.xlim([-1, len(fnames)+1])
    plt.xticks(np.arange(len(fnames)), obslist,
               rotation='vertical')
    plt.xlabel('observation id')
    plt.ylabel('optical efficiency')
    plt.title('{} GHz: +/- 1 sigma quantiles'.format(band))
    plt.tight_layout()
plt.savefig(os.path.splitext(fname)[0] + '_quantiles.png'.format(band), dpi=200)
plt.close()

for band in [90, 150, 220]:
    for pol in ['x', 'y']:
        xpos = [bp[bolo].x_offset / core.G3Units.deg
                for bolo in opteff_all.keys()
                if bp[bolo].band/core.G3Units.GHz==band and \
                        bp[bolo].physical_name.split('.')[-1]==pol]
        ypos = [bp[bolo].y_offset / core.G3Units.deg
                for bolo in opteff_all.keys()
                if bp[bolo].band/core.G3Units.GHz==band and \
                        bp[bolo].physical_name.split('.')[-1]==pol]
        opteff_median = [np.median(opteff_all[bolo])
                         for bolo in opteff_all.keys()
                         if bp[bolo].band/core.G3Units.GHz==band and \
                             bp[bolo].physical_name.split('.')[-1]==pol]
        opteff_median = np.array(opteff_median)
        plt.figure()
        plt.scatter(xpos, ypos, c=opteff_median,
                    vmin=np.percentile(opteff_median,5),
                    vmax=np.percentile(opteff_median,90))
        plt.colorbar()
        plt.xlabel('x offset [deg]')
        plt.ylabel('y offset [deg]')
        plt.title('median optical efficiency')
        plt.tight_layout()
        plt.savefig('opteff_focalplane_median_{}_{}.png'.format(band, pol), dpi=200)
        plt.close()


with open('opteff_all_obs.pkl', 'wb') as f:
    pickle.dump(opteff_all, f)

with open('opteff_all_obs_py2.pkl', 'wb') as f:
    pickle.dump(opteff_all, f, protocol=2)
