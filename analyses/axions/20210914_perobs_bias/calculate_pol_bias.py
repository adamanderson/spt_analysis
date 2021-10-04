from spt3g import autoprocessing
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob
from scipy.signal import lombscargle

load_from_sims = False

field_list = ['ra0hdec-{}'.format(dec) for dec in ['44.75','52.25','59.75','67.25']]
band_list = [90, 150, 220]
data_path = '/sptlocal/user/kferguson/axion_angles_final_more_realizations'
angle_data = {}


if load_from_sims:
    for field in field_list:
        angle_data[field] = {}
        for band in band_list:
            angle_data[field][band] = {}

            fnames = np.sort(glob(os.path.join(data_path, '*{}*{}GHz.pkl'.format(field, band))))
            for fname in fnames:
                print(fname)
                obsid = int(fname.split('_')[-2])
                with open(fname, 'rb') as f:
                    angles = pickle.load(f)

                angle_data[field][band][obsid] = {}
                angle_data[field][band][obsid]['mean'] = np.mean(angles['noise'])
                angle_data[field][band][obsid]['median'] = np.median(angles['noise'])
                angle_data[field][band][obsid]['std'] = np.std(angles['noise'])
                angle_data[field][band][obsid]['N'] = len(angles['noise'])
    
    with open('angle_summary_stats.pkl', 'wb') as f:
        pickle.dump(angle_data, f)

else:
    with open('angle_summary_stats.pkl', 'rb') as f:
        angle_data = pickle.load(f)


for jfield, field in enumerate(field_list):
    for band in band_list:
        time = np.array([obsid for obsid in angle_data[field][band]])
        angle_bias = np.array([angle_data[field][band][obsid]['mean'] \
                            for obsid in angle_data[field][band]]) * 180/np.pi
        bias_unc = np.array([angle_data[field][band][obsid]['std'] / np.sqrt(angle_data[field][band][obsid]['N']) \
                            for obsid in angle_data[field][band]]) * 180/np.pi
        plt.figure()
        plt.errorbar(time, angle_bias, yerr=bias_unc, linestyle='none')
        chi2 = np.sum(angle_bias**2 / bias_unc**2)
        ndf = len(angle_bias)
        plt.title('{}, {} GHz: $\chi^2$ / ndf = {:.1f} / {}'.format(field, band, chi2, ndf))
        plt.xlabel('time')
        plt.ylabel('angle [deg]')
        plt.tight_layout()
        plt.savefig('angle_bias_{}_{}GHz.png'.format(field, band), dpi=200)
        plt.gca().set_xlim([8.0e7, 8.4e7])
        plt.savefig('angle_bias_zoom_{}_{}GHz.png'.format(field, band), dpi=200)
        plt.close()

        ls_freq = np.linspace(0.0005, 2, 4000)
        pgram = lombscargle(time / (24*3600), angle_bias, ls_freq)
        plt.figure()
        plt.plot(ls_freq, pgram)
        plt.xlabel('frequency [1/d]')
        plt.ylabel('Lomb-Scargle periodogram')
        plt.title('{}, {} GHz'.format(field, band))
        plt.tight_layout()
        plt.savefig('angle_bias_lsgram_{}_{}GHz.png'.format(field, band), dpi=200)
        plt.close()

