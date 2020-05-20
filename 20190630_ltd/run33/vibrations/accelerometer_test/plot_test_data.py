from scipy.signal import periodogram, welch
from python_mma8451.read_accelerometer import read_file, read_for_time
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy.optimize import curve_fit

dirname = 'accelerometer_figures'
fnames = glob('*accelerometer*dat')
# for fname in fnames:
#     data, times, rate = read_file(fname)
#     fname_stub = fname.split('_')[-1].rstrip('.dat')
#     axes = ['x', 'y', 'z']

#     for jaxis in range(3):
#         plt.figure(jaxis+1)
#         # f, psd = periodogram(data[:,jaxis], fs=rate, window='hanning')
#         f, psd = welch(data[:,jaxis], fs=rate, nperseg=2048, window='hanning')
#         plt.semilogy(f, np.sqrt(psd))
#         plt.xlabel('frequency [Hz]')
#         plt.ylabel('acceleration ASD [g / rtHz]')
#         plt.title('{} axis: {:.1f} ug / rtHz'.format(axes[jaxis],
#                                                      1e6*np.mean(np.sqrt(psd[(f>3) & (f<15)]))))
#         plt.xlim([0, 100])
#         plt.tight_layout()
#         plt.savefig('{}/{}_axis_freq_{}.png'.format(dirname, axes[jaxis], fname_stub), dpi=200)
#         plt.close()
        
#     for jaxis in range(3):
#         plt.figure(10 + jaxis+1)
#         time = np.arange(len(data[:,jaxis])) / rate
#         plt.plot(time, data[:,jaxis])
#         plt.xlabel('time [sec]')
#         plt.ylabel('acceleration [g]')
#         plt.title('{} axis'.format(axes[jaxis]))
#         plt.xlim([0, 100])
#         plt.tight_layout()
#         plt.savefig('{}/{}_axis_time_{}.png'.format(dirname, axes[jaxis], fname_stub), dpi=200)
#         plt.close()

freqs = []
voltages = []
for fname in fnames:
    if 'V' in fname:
        data, times, rate = read_file(fname)

        # extract only the middle 1/3 of the data because of power-cycling the vibration
        nsamples = len(times)
        times = times[int(nsamples/3.):int(nsamples*2./3.)]
        data = data[int(nsamples/3.):int(nsamples*2./3.),:]
        
        fname_stub = fname.split('_')[-1].rstrip('.dat')
        voltage = float(fname_stub.rstrip('V'))
        voltages.append(voltage)
        axes = ['x', 'y', 'z']

        for jaxis in range(3):
            plt.figure(jaxis)
            f, psd = welch(data[:,jaxis], fs=rate, nperseg=2048, window='hanning')
            if jaxis == 0:
                freq_range = (f<100) & (f>18)
                freqs.append(f[freq_range][np.argmax(psd[freq_range])])
            plt.semilogy(f, np.sqrt(psd), label=fname_stub)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('acceleration ASD [g / rtHz]')
            plt.title('{} axis: {:.1f} ug / rtHz'.format(axes[jaxis],
                                                         1e6*np.mean(np.sqrt(psd[(f>3) & (f<15)]))))
            plt.legend()
            plt.xlim([0, 100])
            plt.tight_layout()
            plt.savefig('{}/{}_axis_freq_all.png'.format(dirname, axes[jaxis]), dpi=200)

def f_cal(x, a, b, c): return a*(x**b) + c
popt, _ = curve_fit(f_cal, voltages, freqs, [freqs[0] / voltages[0], 1, 0.1])
            
plt.figure(10)
plt.plot(voltages, freqs, 'o')
vplot = np.linspace(1,7,100)
plt.plot(vplot, f_cal(vplot, *popt))
plt.title('f = {:.2f} * V^{:.2f} + {:.2f}'.format(popt[0], popt[1], popt[2]))
plt.xlabel('voltage [V]')
plt.ylabel('frequency [Hz]')
plt.tight_layout()
plt.savefig('freqVvoltage.png', dpi=200)
