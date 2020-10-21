import pickle
import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, maps
from scipy.stats import norm
import os.path

coadd_dir = '/sptlocal/user/kferguson/full_daniel_maps/'

with open('weights_var_factors_backup.pkl', 'rb') as f:
    data_kstest = pickle.load(f)

for field in data_kstest['ks_result']:
    for obsid in data_kstest['ks_result'][field]:
        for band in data_kstest['ks_result'][field][obsid]:
            if data_kstest['ks_result'][field][obsid][band].pvalue < 0.01:
                map_fname = '{}_{}GHz_{}_map.g3.gz'.format(field, band, obsid)
                map_frame = list(core.G3File(os.path.join(coadd_dir, map_fname)))[0]
                maps.RemoveWeights(map_frame)

                q_arr = np.array(map_frame['Q'])
                q_weight = np.array(map_frame['Wpol'].QQ)
                q_arr_finite = q_arr[np.isfinite(q_arr)]
                q_weight_finite = q_weight[np.isfinite(q_arr)]

                factor = q_arr_finite * np.sqrt(q_weight_finite)

                plt.figure(1)
                plt.hist(factor / np.std(factor), bins=np.linspace(-4, 4, 101),
                         density=True)
                plt.plot(np.linspace(-4, 4, 101), norm.pdf(np.linspace(-4, 4, 101)))
                plt.xlim([-10,10])
                plt.xlabel('$[Q_i \\times \sqrt{W_{qqi}}] / $std')
                plt.ylabel('pixels')
                plt.title('{} {} {} GHz\n(KS p-value = {:.2e})'.format(field, obsid, band,
                                                                       data_kstest['ks_result'][field][obsid][band].pvalue))
                plt.tight_layout()
                plt.savefig('figures/pixel_value_hist_{}_{}_{}.png'.format(field, obsid, band), dpi=150)

                plt.gca().set_yscale('log')
                plt.savefig('figures/pixel_value_hist_{}_{}_{}_log.png'.format(field, obsid, band), dpi=150)
                plt.clf()
