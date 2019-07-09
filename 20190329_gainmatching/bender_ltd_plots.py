import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration

# some useful noise functions
def readout_noise(x, readout):
    return np.sqrt(readout)*np.ones(len(x))
def photon_noise(x, photon, tau):
    return np.sqrt(photon / (1 + 2*np.pi*((x*tau)**2)))
def atm_noise(x, A, alpha):
    return np.sqrt(A * (x)**(-1*alpha))
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout + (A * (x)**(-1*alpha)) + photon / (1 + 2*np.pi*((x*tau)**2)))
def horizon_model(x, readout, A, alpha):
    return np.sqrt(readout + (A * (x)**(-1*alpha)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout
def horizon_knee_func(x, readout, A, alpha):
    return (A * (x)**(-1*alpha)) - readout

fr = list(core.G3File('horizon_noise_77863968_bender_ltd.g3'))[1]

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}
band_labels = {90:'95', 150:'150', 220:'220'}

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,4))
fig.subplots_adjust(wspace=0)
for jband, band in enumerate([90., 150., 220.]):
    
    group = '{:.1f}_w180'.format(band)

    ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
    ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
    asd_diff = np.array(fr['AverageASDDiff'][group]) / np.sqrt(2.)
    asd_sum = np.array(fr['AverageASDSum'][group]) / np.sqrt(2.)

    par = fr["AverageASDDiffFitParams"][group]
    ax[jband].loglog(ff_sum[ff_sum<75], asd_sum[ff_sum<75],
                     label='pair sum (measured)', color='0.6')
    ax[jband].loglog(ff_diff[ff_diff<75], asd_diff[ff_diff<75],
                     label='pair difference (measured)', color='k')
    ax[jband].loglog(ff_sum, atm_noise(ff_sum, par[1], par[2]) / np.sqrt(2.),
                     'C0--', label='low-frequency noise')
    ax[jband].loglog(ff_sum, readout_noise(ff_sum, par[0]) / np.sqrt(2.),
                     'C2--', label='white noise')
    ax[jband].loglog(ff_sum, horizon_model(ff_sum, *list(par)) / np.sqrt(2.),
                     'C3--', label='total noise model')

    ax[jband].set_title('{} GHz'.format(band_labels[band]))
    ax[jband].set_xlabel('frequency [Hz]')
    ax[jband].grid()
ax[0].set_ylabel('current noise [pA/$\sqrt{Hz}$]')

plt.ylim([5,1000])
plt.legend()
plt.tight_layout()
plt.savefig('w180_horizon_noise_ltd.pdf')
