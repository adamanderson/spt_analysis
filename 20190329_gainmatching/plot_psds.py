from spt3g import core, calibration, dfmux
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram

d = [fr for fr in core.G3File('gain_match_test.g3')]
pairnames = np.random.choice(d[3]['PairDiffTimestreams'].keys(), 10)
for pairname in pairnames:
    ts = d[3]['PairDiffTimestreams'][pairname]
    bolo1 = pairname.split('_')[0]
    bolo2 = pairname.split('_')[1]

    plt.figure()
    ff, psd = periodogram(ts, fs=152.5)
    plt.loglog(ff, np.sqrt(psd) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6,
               label='{} - {}'.format(bolo1, bolo2))
    ts = np.array(d[3]['CalTimestreams'][bolo1]) + \
         d[3]['GainMatchCoeff'][pairname]*np.array(d[3]['CalTimestreams'][bolo2])
    ff, psd = periodogram(ts, fs=152.5)
    plt.loglog(ff, np.sqrt(psd) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6,
               label='{} + {}'.format(bolo1, bolo2))
    plt.grid()
    plt.legend()
    plt.ylim([1e2, 1e8])
    plt.savefig('{}.png'.format(pairname), dpi=200)
    plt.close()

wafer_psds_sum = {}
wafer_psds_diff = {}
wafer_nbolos = {}
boloprops = d[0]['BolometerProperties']
for pairname in d[3]['PairDiffTimestreams'].keys():
    bolo1 = pairname.split('_')[0]
    bolo2 = pairname.split('_')[1]
    wafer = boloprops[bolo1].wafer_id
    print(pairname)
    print(wafer)
    if wafer not in wafer_psds_sum:
        ff, psd = periodogram(d[3]['PairDiffTimestreams'][pairname], fs=152.5)
        wafer_psds_diff[wafer] = {}
        wafer_psds_sum[wafer] = {}
        wafer_nbolos[wafer] = 0
    ts_diff = np.array(d[3]['PairDiffTimestreams'][pairname]) / np.sqrt(2.)
    ts_sum = (np.array(d[3]['CalTimestreams'][bolo1]) + \
              d[3]['GainMatchCoeff'][pairname]*np.array(d[3]['CalTimestreams'][bolo2])) / np.sqrt(2.)
    ff, psd_diff = periodogram(ts_diff, fs=152.5)
    ff, psd_sum = periodogram(ts_sum, fs=152.5)
    wafer_psds_sum[wafer][pairname] = np.sqrt(psd_sum) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6
    wafer_psds_diff[wafer][pairname] = np.sqrt(psd_diff) / (1. / np.sqrt(core.G3Units.Hz)) * 1e6
    wafer_nbolos[wafer] += 1

print(wafer_nbolos)
for wafer in wafer_psds_sum:
    psds_diff = np.vstack([wafer_psds_diff[wafer][pair] for pair in wafer_psds_diff[wafer].keys()])
    psds_sum = np.vstack([wafer_psds_sum[wafer][pair] for pair in wafer_psds_sum[wafer].keys()])
    
    plt.figure()
    plt.loglog(ff, np.mean(psds_sum, axis=0), label='pair sum')
    plt.loglog(ff, np.mean(psds_diff, axis=0), label='pair difference')
    plt.title('{}: mean PSDs'.format(wafer))
    plt.grid()
    plt.legend()
    plt.ylim([1e2, 1e8])
    plt.savefig('{}_mean_psd.png'.format(wafer), dpi=200)
    plt.close()

    plt.figure()
    plt.loglog(ff, np.median(psds_sum, axis=0), label='pair sum')
    plt.loglog(ff, np.median(psds_diff, axis=0), label='pair difference')
    plt.title('{}: median PSDs'.format(wafer))
    plt.grid()
    plt.legend()
    plt.ylim([1e2, 1e8])
    plt.savefig('{}_median_psd.png'.format(wafer), dpi=200)
    plt.close()
