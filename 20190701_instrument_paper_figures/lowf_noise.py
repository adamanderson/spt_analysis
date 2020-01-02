from glob import glob
from scipy.optimize import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from spt3g import core

fnames_noise = glob('/spt/user/adama/20190911_noise_gainmatch_cal/fullrate/gainmatching_noise_*.g3')

def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout

f_knee_dict = {}
for fname in fnames_noise:
    print(fname)
    fr = list(core.G3File(fname))[1]
    
    for jband, band in enumerate([90., 150., 220.]):
        for jwafer, wafer in enumerate(['w172', 'w174', 'w176', 'w177', 'w180',
                                        'w181', 'w188', 'w203', 'w204', 'w206']):
            group = '{:.1f}_{}'.format(band, wafer)
            
            if band not in f_knee_dict:
                f_knee_dict[band] = {}
            if wafer not in f_knee_dict[band]:
                f_knee_dict[band][wafer] = []
                
            try:
                ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
                asd_diff = np.array(fr['AverageASDDiff'][group])
                par_diff = fr["AverageASDDiffFitParams"][group]

                f_knee = bisect(knee_func, a=0.01, b=10.0, args=tuple(par_diff))
                f_knee_dict[band][wafer].append(f_knee)
            except:
                pass

fnames = glob('/spt/user/adama/20190911_noise_gainmatch_cal/fullrate/gainmatching_noise_83521028.g3')
d = list(core.G3File(fnames))[1]

f_hi = 73
freq = np.array(d["AverageASDDiff"]['frequency']) / core.G3Units.Hz

fig = plt.figure(figsize=(12,4))
gridspec.GridSpec(4,3)
band_labels = {90:'95', 150:'150', 220:'220'}


for jband, band in enumerate(f_knee_dict):
    plt.subplot2grid((4,3), (0,jband), colspan=1, rowspan=1)
    f_knees = np.hstack([f_knee_dict[band][wafer] for wafer in f_knee_dict[band].keys()])
    plt.hist(f_knees, bins=np.logspace(np.log10(1e-2), np.log10(73), 21),
             alpha=0.5, color='C0') #, normed=True)
    plt.hist(f_knees, bins=np.logspace(np.log10(1e-2), np.log10(73), 21),
             histtype='step', color='C0') #, normed=True)
#     plt.axis([1e-2, 73, 0, 11])
    plt.axis([1e-2, 73, 0, 220])
    plt.gca().set_xscale('log')
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.title('{} GHz'.format(band_labels[band]))
    
    if jband == 0:
        plt.ylabel('$1/f$ knees of\nnoise stares')


# ax = plt.subplot(2,3,4)
plt.subplot2grid((4,3), (1,0), colspan=1, rowspan=3)
asd_diff = np.array(d["AverageASDDiff"]['90.0_w204'])
asd_sum = np.array(d["AverageASDSum"]['90.0_w204'])
plt.loglog(freq[freq<f_hi], asd_diff[freq<f_hi] / np.sqrt(2.))
plt.loglog(freq[freq<f_hi], asd_sum[freq<f_hi] / np.sqrt(2.))
print(np.mean(asd_diff[(freq>1) & (freq<5)]) / np.sqrt(2.))
plt.grid()
plt.axis([1e-2, 73, 200, 50000])
plt.ylabel('NET [$\mu$K $\sqrt{s}$]')

# plt.subplot(2,3,5)
plt.subplot2grid((4,3), (1,1), colspan=1, rowspan=3)
asd_diff = np.array(d["AverageASDDiff"]['150.0_w204'])
asd_sum = np.array(d["AverageASDSum"]['150.0_w204'])
plt.loglog(freq[freq<f_hi], asd_diff[freq<f_hi] / np.sqrt(2.))
plt.loglog(freq[freq<f_hi], asd_sum[freq<f_hi] / np.sqrt(2.))
print(np.mean(asd_diff[(freq>1) & (freq<5)]) / np.sqrt(2.))
plt.grid()
plt.axis([1e-2, 73, 200, 50000])
plt.gca().set_yticklabels([])
plt.xlabel('frequency [Hz]')

# plt.subplot(2,3,6)
plt.subplot2grid((4,3), (1,2), colspan=1, rowspan=3)
asd_diff = np.array(d["AverageASDDiff"]['220.0_w204'])
asd_sum = np.array(d["AverageASDSum"]['220.0_w204'])
plt.loglog(freq[freq<f_hi], asd_diff[freq<f_hi] / np.sqrt(2.))
plt.loglog(freq[freq<f_hi], asd_sum[freq<f_hi] / np.sqrt(2.))
print(np.mean(asd_diff[(freq>1) & (freq<5)]) / np.sqrt(2.))
plt.grid()
plt.axis([1e-2, 73, 200, 50000])
plt.gca().set_yticklabels([])

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig('fig_lowf_noise_summary.pdf')
