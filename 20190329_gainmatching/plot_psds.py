from spt3g import core, calibration, dfmux
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram

d = [fr for fr in core.G3File('gain_match_test.g3')]
pairnames = np.random.choice(d[4]['PairDiffTimestreams'].keys(), 10)
for pairname in pairnames:
    ts = d[4]['PairDiffTimestreams'][pairname]
    bolo1 = pairname.split('_')[0]
    bolo2 = pairname.split('_')[1]

    plt.figure()
    ff, psd = periodogram(ts, fs=152.5)
    plt.loglog(ff, np.sqrt(psd) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6,
               label='{} - {}'.format(bolo1, bolo2))
    ts = d[4]['GainMatchCoeff'][bolo1] * np.array(d[4]['CalTimestreams'][bolo1]) + \
         d[4]['GainMatchCoeff'][bolo2] * np.array(d[4]['CalTimestreams'][bolo2])
    ff, psd = periodogram(ts, fs=152.5)
    plt.loglog(ff, np.sqrt(psd) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6,
               label='{} + {}'.format(bolo1, bolo2))
    plt.grid()
    plt.legend()
    plt.ylim([1e2, 1e8])
    plt.savefig('figures/{}.png'.format(pairname), dpi=200)
    plt.close()

wafer_psds_sum = {}
wafer_psds_diff = {}
wafer_nbolos = {}
boloprops = d[0]['BolometerProperties']
for pairname in d[4]['PairDiffTimestreams'].keys():
    bolo1 = pairname.split('_')[0]
    bolo2 = pairname.split('_')[1]
    wafer = boloprops[bolo1].wafer_id
    band = boloprops[bolo1].band / core.G3Units.GHz
    if wafer not in wafer_psds_sum:
        ff, psd = periodogram(d[4]['PairDiffTimestreams'][pairname], fs=152.5)
        wafer_psds_diff[wafer] = {90.:{}, 150.:{}, 220.:{}}
        wafer_psds_sum[wafer] = {90.:{}, 150.:{}, 220.:{}}
        wafer_nbolos[wafer] = {90.:0, 150.:0, 220.:0}
    ts_diff = np.array(d[4]['PairDiffTimestreams'][pairname]) / np.sqrt(2.)
    ts_sum = np.array(d[4]['PairSumTimestreams'][pairname]) / np.sqrt(2.)
    ff, psd_diff = periodogram(ts_diff, fs=152.5)
    ff, psd_sum = periodogram(ts_sum, fs=152.5)
    wafer_psds_sum[wafer][band][pairname] = np.sqrt(psd_sum) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6 / np.sqrt(2.) # rt(2) for uK rtHz to uK rtsec
    wafer_psds_diff[wafer][band][pairname] = np.sqrt(psd_diff) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6 / np.sqrt(2.) # rt(2) for uK rtHz to uK rtsec
    wafer_nbolos[wafer][band] += 1

print(wafer_nbolos)
for wafer in wafer_psds_sum:
    for band in [90., 150., 220.]:
        psds_diff = np.vstack([wafer_psds_diff[wafer][band][pair] for pair in wafer_psds_diff[wafer][band].keys()])
        psds_sum = np.vstack([wafer_psds_sum[wafer][band][pair] for pair in wafer_psds_sum[wafer][band].keys()])

        plt.figure()
        plt.loglog(ff, np.mean(psds_sum, axis=0), label='pair sum')
        plt.loglog(ff, np.mean(psds_diff, axis=0), label='pair difference')
        plt.title('{} {} GHz: mean PSDs'.format(wafer, int(band)))
        plt.grid()
        plt.legend()
        plt.ylim([1e2, 1e6])
        plt.savefig('figures/{}_{}_mean_psd.png'.format(wafer, int(band)), dpi=200)
        plt.close()

        plt.figure()
        plt.loglog(ff, np.median(psds_sum, axis=0), label='pair sum')
        plt.loglog(ff, np.median(psds_diff, axis=0), label='pair difference')
        plt.title('{} {} GHz: median PSDs'.format(wafer, int(band)))
        plt.grid()
        plt.legend()
        plt.ylim([1e2, 1e6])
        plt.savefig('figures/{}_{}_median_psd.png'.format(wafer, int(band)), dpi=200)
        plt.close()
