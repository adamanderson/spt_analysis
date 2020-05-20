import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
from scipy.signal import welch
from glob import glob

datapath = '/daq/pydfmux_output/20180925/20180925_220033_measure_noise/data'
with open(os.path.join(datapath, '648_BOLOS_INFO_AND_MAPPING.pkl'), 'rb') as f:
    d = pickle.load(f)

modpstrings = np.unique(['/'.join(b.split('/')[:3]) for b in d.keys()])

for pstring in modpstrings:
    print(pstring)
    
    plt.figure()

    for bolo in d.keys():
        if pstring in bolo:
            tod = d[bolo]['noise']['tod_i']
            freq, psd = welch(tod, fs=152.5)
            asd = np.sqrt(psd)*1e12
            plt.plot(freq, asd, linewidth=0.5)
    plt.ylim([0,150])
    plt.savefig('figures/{}.png'.format(pstring.replace('/', '_')), dpi=200)
    plt.close()
