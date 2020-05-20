import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

#fname = '20181009_181035_xtalk_noise_hkdata.pkl'
#fname = '20181010_222318_xtalk_noise_hkdata.pkl'
#fname = '20181010_230926_xtalk_noise_hkdata.pkl'
#fname = '20181011_141536_xtalk_noise_hkdata.pkl'
#fname = '20181011_152113_xtalk_noise_hkdata.pkl'
fname = '20181011_154807_xtalk_noise_hkdata.pkl'

with open(fname, 'rb') as f:
    d = pickle.load(f, encoding='latin1')

amps_list = [0.001, 0.005, 0.01]
    
for dtype in d:
    for jamp, amp in enumerate(amps_list):
        if dtype != 'all_modules':
            for jmod, rmod in enumerate(list(d[dtype][amp].keys())):
                plt.figure(jmod)
                noise_fname = glob('{}/data/*pkl'.format(d[dtype][amp][rmod]['dump_info']))[0]
                with open(noise_fname, 'rb') as f:
                    dnoise = pickle.load(f, encoding='latin1')
                noise = np.array([dnoise[chan]['noise']['i_phase']['median_noise']
                                  for chan in dnoise])
                freqs = np.array([dnoise[chan]['frequency']
                                  for chan in dnoise])
                plt.plot(freqs, noise, 'o', label='amp. = {}'.format(amp), markersize=2)

                plt.figure(100 + jmod)
                plt.hist(noise, bins=np.linspace(0, 50, 26),
                         label='amp. = {}'.format(amp), histtype='stepfilled',
                         alpha=0.5, color='C{}'.format(jamp))
                plt.hist(noise, bins=np.linspace(0, 50, 26),
                         histtype='step',
                         color='C{}'.format(jamp))
        else:
            # plt.figure(10)
            noise_fname = glob('{}/data/*pkl'.format(d[dtype][amp]['dump_info']))[0]
            with open(noise_fname, 'rb') as f:
                dnoise = pickle.load(f, encoding='latin1')
            all_chans = list(dnoise.keys())
            all_rmods = np.unique(['/'.join(chan.split('/')[:3]) for chan in all_chans])
                
            for jmod, rmod in enumerate(all_rmods):
                plt.figure(10 + jmod)
                noise = np.array([dnoise[chan]['noise']['i_phase']['median_noise']
                                  for chan in dnoise if rmod in chan])
                freqs = np.array([dnoise[chan]['frequency']
                                  for chan in dnoise if rmod in chan])
                plt.plot(freqs, noise, 'o', label='amp. = {}'.format(amp), markersize=2)

                plt.figure(100 + 10 + jmod)
                plt.hist(noise, bins=np.linspace(0, 50, 26),
                         label='amp. = {}'.format(amp), histtype='stepfilled',
                         alpha=0.5, color='C{}'.format(jamp))
                plt.hist(noise, bins=np.linspace(0, 50, 26),
                         histtype='step',
                         color='C{}'.format(jamp))

    if dtype != 'all_modules':
        for jmod, rmod in enumerate(list(d[dtype][amp].keys())):
            plt.figure(jmod)
            plt.legend()
            plt.xlabel('frequency [Hz]')
            plt.ylabel('noise [pA/rtHz]')
            plt.axis([1.5e6, 5.5e6, 0, 100])
            plt.tight_layout()
            plt.savefig('{}_{}.png'.format(dtype, rmod.replace('/', '_')), dpi=200)
            plt.close()

            plt.figure(100 + jmod)
            plt.legend()
            plt.xlabel('noise [pA/rtHz]')
            plt.xlim([0, 50])
            plt.tight_layout()
            plt.savefig('{}_{}_hist.png'.format(dtype, rmod.replace('/', '_')), dpi=200)
            plt.close()

    else:
        for jmod, rmod in enumerate(all_rmods):
            plt.figure(10 + jmod)
            plt.legend()
            plt.xlabel('frequency [Hz]')
            plt.ylabel('noise [pA/rtHz]')
            plt.axis([1.5e6, 5.5e6, 0, 100])
            plt.tight_layout()
            plt.savefig('{}_{}.png'.format(dtype, rmod.replace('/', '_')), dpi=200)
            plt.close()
            
            plt.figure(100 + 10 + jmod)
            plt.legend()
            plt.xlabel('noise [pA/rtHz]')
            plt.xlim([0, 50])
            plt.tight_layout()
            plt.savefig('{}_{}_hist.png'.format(dtype, rmod.replace('/', '_')), dpi=200)
            plt.close()
