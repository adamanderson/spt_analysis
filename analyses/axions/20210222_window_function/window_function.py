import numpy as np 
import matplotlib.pyplot as plt
import pickle

def window_function(f, tn):
    return np.sum( np.exp(-2.*np.pi*1j * f * tn) )

# Unit frequency
# regular window function
t_regular = np.arange(0, 433)
t_regular_2x = np.arange(0, 2*len(t_regular))
freqs = np.linspace(-5, 5, 10001)
psd_window_regular = [np.abs(window_function(f, t_regular))**2 for f in freqs]
psd_window_regular_2x = [np.abs(window_function(f, t_regular_2x))**2 for f in freqs]

# randomized window function
t_random = t_regular + np.random.normal(0, 0.1, len(t_regular))
psd_window_random = [np.abs(window_function(f, t_random))**2 for f in freqs]

# plotting
plt.figure(1)
plt.semilogy(freqs, psd_window_regular, label='uniform sampling (N={})'.format(len(t_regular)))
#plt.semilogy(freqs, psd_window_regular_2x, label='uniform sampling (N={})'.format(len(t_regular_2x)))
plt.semilogy(freqs, psd_window_random, label='uniform sampling, random jitter')
plt.legend()
plt.xlabel('frequency')
plt.ylabel('PSD')
plt.tight_layout()
plt.savefig('window_psd_test.png', dpi=200)

# Realistic frequencies
with open('obsids_1500d_2019.pkl', 'rb') as f:
    obsids = pickle.load(f)
freqs_spt = np.linspace(-20, 20, 100001)

# SPT sampling
psd_window_2019 = [np.abs(window_function(f, obsids/(3600*24)))**2 for f in freqs_spt]

# Uniform sampling in SPT time period
obsids_uniform = np.linspace(np.min(obsids), np.max(obsids), len(obsids))
psd_window_2019_regular = [np.abs(window_function(f, obsids_uniform/(3600*24)))**2 for f in freqs_spt]

# plotting
plt.figure(2)
plt.semilogy(freqs_spt, psd_window_2019_regular, label='uniform sampling over SPT time')
plt.semilogy(freqs_spt, psd_window_2019, label='SPT sampling')
plt.legend()
plt.xlabel('frequency [1/d]')
plt.ylabel('PSD')
plt.tight_layout()
plt.savefig('window_psd_spt_log.png', dpi=200)

plt.figure(3)
plt.plot(freqs_spt, psd_window_2019_regular, label='uniform sampling over SPT time')
plt.plot(freqs_spt, psd_window_2019, label='SPT sampling')
plt.legend()
plt.xlabel('frequency [1/d]')
plt.ylabel('PSD')
plt.tight_layout()
plt.savefig('window_psd_spt_lin.png', dpi=200)

plt.show()