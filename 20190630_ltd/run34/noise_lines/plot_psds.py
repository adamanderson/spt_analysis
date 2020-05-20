import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
from scipy.signal import welch
from glob import glob

fname = '../noise_vs_rfrac/20181009_232714_noise_v_rfrac_hkdata.pkl'
stub_name = 'Fnal_run34_w206_intransition'

with open(fname, 'rb') as f:
    d = pickle.load(f, encoding='latin1')

for rfrac in [0.95, 0.90, 0.85, 0.80, 0.75]:
    noise_fname = glob('{}/data/*pkl'.format(d[rfrac]))[0]
    with open(noise_fname, 'rb') as f:
        dnoise = pickle.load(f, encoding='latin1')

    pstrings = np.unique(['/'.join(bolo.split('/')[:3]) for bolo in dnoise])

    for pstring in pstrings:
        plt.figure()
        for bolo in dnoise.keys():
            if pstring in bolo:
                tod = dnoise[bolo]['noise']['tod_i']
                freq, psd = welch(tod, fs=152.5, nperseg=300)
                asd = np.sqrt(psd)*1e12
                plt.plot(freq, asd, linewidth=0.5)
        plt.axis([0, 75, 0, 150])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('noise [pA/rtHz]')
        plt.title('{}: rfrac = {}'.format(pstring, rfrac))
        plt.tight_layout()
        plt.savefig('figures/{}_{}_{}.png'.format(stub_name, pstring.replace('/', '_'), rfrac), dpi=200)
        plt.close()
