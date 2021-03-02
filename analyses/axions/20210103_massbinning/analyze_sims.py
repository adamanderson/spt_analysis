import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import os

datadir = '/sptlocal/user/adama/axions/20210103_massbinning/'
freq_range = '0.01000-2.00000'
freq_amp_pairs = [('0.01000', '0.00000'), ('0.01000', '0.20000'), ('0.01000', '1.00000'),
                  ('0.50000', '0.20000'), ('0.50000', '1.00000')]
# freqs = ['0.01000', '0.50000'] #, '0.452', '0.894', '1.337', '1.558', '1.779', '2.000']
# signal_amps = ['0.20000', '1.00000']
best_fit_amp = {}
ul_amp = {}

for freq, signal_amp in freq_amp_pairs:
    fnames = glob(os.path.join(datadir,
                            'massbinning_ul_fitfreq={}_amp={}_freq={}-*.pkl'.format(freq_range, signal_amp, freq)))

    if freq not in best_fit_amp:
        best_fit_amp[freq] = {}
    if signal_amp not in best_fit_amp[freq]:
        best_fit_amp[freq][signal_amp] = {}
    if freq not in ul_amp:
        ul_amp[freq] = {}
    if signal_amp not in ul_amp[freq]:
        ul_amp[freq][signal_amp] = {}
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            for jexpt in d['results']:
                for fitfreq, amp in d['results'][jexpt]['A_upperlimit'].items():
                    if fitfreq not in ul_amp[freq][signal_amp]:
                        ul_amp[freq][signal_amp][fitfreq] = []
                    ul_amp[freq][signal_amp][fitfreq].append(float(amp[0]))

                for fitfreq, amp in d['results'][jexpt]['A_fit'].items():
                    if fitfreq not in best_fit_amp[freq][signal_amp]:
                        best_fit_amp[freq][signal_amp][fitfreq] = []
                    best_fit_amp[freq][signal_amp][fitfreq].append(float(amp))

    # plot median and +/- 1 sigma limits + histogram of limits
    for jfreq, freq in enumerate(ul_amp):
        ulfreqs = list(ul_amp[freq][signal_amp].keys())
        mean_amp_ul = np.array([np.mean(ul_amp[freq][signal_amp][ulfreqs]) for ulfreqs in ul_amp[freq][signal_amp]])
        down1sigma_amp_ul = np.array([np.percentile(ul_amp[freq][signal_amp][ulfreqs], 16) for ulfreqs in ul_amp[freq][signal_amp]])
        up1sigma_amp_ul = np.array([np.percentile(ul_amp[freq][signal_amp][ulfreqs], 84) for ulfreqs in ul_amp[freq][signal_amp]])
        
        plt.figure()
        plt.gca().fill_between(ulfreqs, down1sigma_amp_ul, up1sigma_amp_ul, alpha=0.5, color='C2')
        plt.plot(ulfreqs, mean_amp_ul)
        plt.xlabel('frequency [1/d]')
        plt.ylabel('amplitude [deg]')
        plt.title('95% C.L. upper limit with {} deg signal at {} 1/d'.format(signal_amp, freq))
        plt.tight_layout()
        plt.savefig('figures/A_ul_freq={}_amp={}_linx.png'.format(freq, signal_amp), dpi=200)
        plt.gca().set_xscale('log')
        plt.savefig('figures/A_ul_freq={}_amp={}_logx.png'.format(freq, signal_amp), dpi=200)
        plt.close()

        # histogram of limits
        plt.figure()
        ulfreqs = list(ul_amp[freq][signal_amp].keys())
        ulfreq = ulfreqs[20]
        ul_amplitudes = ul_amp[freq][signal_amp][fitfreq]
        _ = plt.hist(ul_amplitudes, bins=np.linspace(0, 0.4, 31))
        plt.title('95% C.L. upper limit at {:.4f} 1/d'.format(ulfreq))
        plt.xlabel('amplitude [deg]')
        plt.tight_layout()
        plt.savefig('figures/A_ul_hist_freq={}_amp={}_fitfreq={}.png'.format(freq, signal_amp, fitfreq), dpi=200)
        plt.close()


        fitfreqs = list(best_fit_amp[freq][signal_amp].keys())
        mean_amp_fit = np.array([np.mean(best_fit_amp[freq][signal_amp][fitfreq]) for fitfreq in best_fit_amp[freq][signal_amp]])
        down1sigma_amp_fit = np.array([np.percentile(best_fit_amp[freq][signal_amp][fitfreq], 16) for fitfreq in best_fit_amp[freq][signal_amp]])
        up1sigma_amp_fit = np.array([np.percentile(best_fit_amp[freq][signal_amp][fitfreq], 84) for fitfreq in best_fit_amp[freq][signal_amp]])

        plt.figure()
        plt.gca().fill_between(fitfreqs, down1sigma_amp_fit, up1sigma_amp_fit, alpha=0.5, color='C2')
        plt.plot(fitfreqs, mean_amp_fit)
        plt.xlabel('frequency [1/d]')
        plt.ylabel('amplitude [deg]')
        plt.title('best-fit amplitude with {} deg signal at {} 1/d'.format(signal_amp, freq))
        plt.tight_layout()
        plt.savefig('figures/A_fit_freq={}_amp={}_linx.png'.format(freq, signal_amp), dpi=200)
        plt.gca().set_xscale('log')
        plt.savefig('figures/A_fit_freq={}_amp={}_logx.png'.format(freq, signal_amp), dpi=200)
        plt.close()

        # histogram of limits
        plt.figure()
        fitfreqs = list(best_fit_amp[freq][signal_amp].keys())
        fitfreq = fitfreqs[20]
        fit_amplitudes = best_fit_amp[freq][signal_amp][fitfreq]
        _ = plt.hist(fit_amplitudes, bins=np.linspace(0, 0.4, 31))
        plt.title('best-fit amplitude at {:.4f} 1/d'.format(ulfreq))
        plt.xlabel('amplitude [deg]')
        plt.tight_layout()
        plt.savefig('figures/A_fit_hist_freq={}_amp={}_fitfreq={}.png'.format(freq, signal_amp, fitfreq), dpi=200)
        plt.close()