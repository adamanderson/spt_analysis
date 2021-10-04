from glob import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from spt3g import autoprocessing
from scipy.signal import lombscargle

data_path = '/sptlocal/user/kferguson'

# get subfields corresponding to each obsid
scanify = autoprocessing.ScanifyDatabase(read_only=True)


# plot averaged angles by month with some kind of point-source masking
field_list = ['ra0hdec-{}'.format(dec) for dec in ['44.75','52.25','59.75','67.25']]
for field in field_list:
    angles = {}

    for band in [90, 150]:
        # fnames = np.sort(glob('/sptgrid/user/kferguson/axion_angles_mask_focus_quasar/'
        #                       'angles_{}_*_{}GHz.pkl'.format(field, band)))
        # for fname in fnames:
        #     print(fname)
        #     with open(fname, 'rb') as f:
        #         obsid = int(os.path.basename(fname).split('_')[2])
        #         if obsid not in angles:
        #             angles[obsid] = {}
        #         d = pickle.load(f)
        #         angles[obsid][band] = d['obs']

        with open(os.path.join(data_path, 'collated_final_angles_more_realizations_' + '%sGHz.pkl'%band),'rb') as f:
            d = pickle.load(f)
        ids = np.sort(list(d.keys()))

        subfields = {}
        for obsid in ids:
            subfields[obsid] = scanify.get_entries('.*', obsid, match_regex=True)[0][0]

        for obs in ids:
            if subfields[obs] == field:
                if obs not in angles:
                    angles[obs] = {}
                angles[obs][band] = d[obs]['angle']
    
    angle_differences = np.array([angles[obsid][90] - angles[obsid][150] for obsid in angles
                                  if 90 in angles[obsid] and 150 in angles[obsid]])
    times = np.array([obsid for obsid in angles
                      if 90 in angles[obsid] and 150 in angles[obsid]])

    plt.figure()
    plt.plot(times, angle_differences*180/np.pi, linestyle='none', marker='o')
    plt.xlabel('time')
    plt.ylabel('pol angle [deg]')
    # plt.legend()
    plt.title('{}'.format(field))
    plt.tight_layout()
    plt.savefig('pol_angle_90minus150_{}.png'.format(field))
    plt.close()

    ls_freq = np.linspace(0.0005, 2, 4000)
    pgram = lombscargle(times / (24*3600), angle_differences*180/np.pi, ls_freq)
    plt.figure()
    plt.plot(ls_freq, pgram)
    plt.xlabel('frequency [1/d]')
    plt.ylabel('Lomb-Scargle periodogram')
    plt.title('{}'.format(field))
    plt.tight_layout()
    plt.savefig('pol_angle_90minus150_lsgram_{}.png'.format(field), dpi=200)
    plt.close()
