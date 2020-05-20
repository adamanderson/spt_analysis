import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

fname = '20181014_141329_noise_v_rfrac_hkdata.pkl'
#fname = '20181012_225752_noise_v_rfrac_hkdata.pkl'
#fname = '20181012_170513_noise_v_rfrac_hkdata.pkl'
#fname = '20181012_143442_noise_v_rfrac_hkdata.pkl'
#fname = '20181012_132943_noise_v_rfrac_hkdata.pkl'
#fname = '20181009_232714_noise_v_rfrac_hkdata.pkl'
stub_name = 'Fnal_run34_w206_noiseVrfrac'

with open(fname, 'rb') as f:
    d = pickle.load(f, encoding='latin1')

    
for jamp, amp in enumerate([1.0, 0.95, 0.85, 0.75]): #np.sort(list(d.keys())):
    noise_fname = glob('{}/data/*pkl'.format(d[amp]))[0]
    
    with open(noise_fname, 'rb') as f:
        dnoise = pickle.load(f, encoding='latin1')

    pstrings = np.unique(['/'.join(bolo.split('/')[:3]) for bolo in dnoise])
    
    for jpstring, pstring in enumerate(pstrings):
        noise = np.array([dnoise[chan]['noise']['i_phase']['median_noise']
                          for chan in dnoise if pstring in chan])
        freqs = np.array([dnoise[chan]['frequency']
                          for chan in dnoise if pstring in chan])

        plt.figure(10 + jpstring)
        plt.plot(freqs, noise, 'o', label='rfrac = {}'.format(amp), markersize=4)

        plt.figure(100 + jpstring)
        plt.hist(noise, bins=np.linspace(0, 50, 31), histtype='stepfilled',
                 label='rfrac = {}'.format(amp), alpha=0.5, color='C{}'.format(jamp))
        plt.hist(noise, bins=np.linspace(0, 50, 31), histtype='step', color='C{}'.format(jamp))

for jpstring, pstring in enumerate(pstrings):
    plt.figure(10 + jpstring)
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('noise [pA/rtHz]')
    plt.axis([1.5e6, 5.5e6, 0, 50])
    plt.legend()
    plt.title(pstring)
    plt.tight_layout()
    plt.savefig('figures/{}_{}.png'.format(stub_name, pstring.replace('/', '_')))

    plt.figure(100 + jpstring)
    plt.xlabel('noise [pA/rtHz]')
    plt.xlim([0, 50])
    plt.legend()
    plt.title(pstring)
    plt.tight_layout()
    plt.savefig('figures/{}_hist_{}.png'.format(stub_name, pstring.replace('/', '_')))

