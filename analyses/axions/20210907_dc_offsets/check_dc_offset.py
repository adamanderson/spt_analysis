from spt3g import autoprocessing
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob
from spt3g.std_processing import time_to_obsid


scanify = autoprocessing.ScanifyDatabase(read_only=True)
field_list = ['ra0hdec-{}'.format(dec) for dec in ['44.75','52.25','59.75','67.25']]
data_path = '/sptlocal/user/kferguson'

# get obsids
with open(os.path.join(data_path, 'collated_final_angles_more_realizations_90GHz.pkl'), 'rb') as f:
    angle_pk = pickle.load(f)
ids = sorted(list(angle_pk.keys()))

# get subfields corresponding to each obsid
subfields = {}
for obsid in ids:
    subfields[obsid] = scanify.get_entries('.*', obsid, match_regex=True)[0][0]

# load the angle data
time_edges = [0, 7.25e7, 8.75e7, 1e10] #[0, 81388800, 1e10]
for jtime in range(len(time_edges) - 1):
    angle_data = {}
    data_dist = {}
    for jband, band in enumerate([90,150]):
        with open(os.path.join(data_path, 'collated_final_angles_more_realizations_' + '%sGHz.pkl'%band),'rb') as f:
            angle_data[band] = pickle.load(f)

        data_dist[band] = {'ra0hdec-%s'%dec: [] for dec in ['44.75','52.25','59.75','67.25']}
        for obs in ids:
            if obs > time_edges[jtime] and obs < time_edges[jtime+1]:
                sub = subfields[obs]
                data_dist[band][sub].append(angle_data[band][obs]['angle'] / angle_data[band][obs]['unc'])

        plt.figure(jband+1, figsize=(8,8))
        for jfield, field in enumerate(list(data_dist[band].keys())):
            mean_norm_angle = np.mean(data_dist[band][field])
            std_norm_angle = np.std(data_dist[band][field])
            n_obs = len(data_dist[band][field])

            plt.subplot(2,2,jfield+1)
            plt.hist(data_dist[band][field],
                    bins=np.linspace(-5,5,31),
                    histtype='step',
                    density=True,
                    label='time period {} ({:.5f} $\pm$ {:.5f})'.format(jtime+1, mean_norm_angle,
                                                                        std_norm_angle / np.sqrt(n_obs)))
            plt.title('{} {} GHz'.format(field, band))
            plt.xlabel('angle [deg]')
            plt.legend()
        plt.tight_layout()
        plt.savefig('dc_offset_check_{}GHz.png'.format(band, jtime), dpi=200)
        plt.close()

# plot averaged angles by month
time_edges = [time_to_obsid('2019{:02}01_000000'.format(month)) for month in np.arange(3,12)]
band = 90
field_list = ['ra0hdec-52.25'] #['ra0hdec-%s'%dec for dec in ['44.75','52.25','59.75','67.25']]
for band in [90, 150]: #[90, 150]:
    with open(os.path.join(data_path, 'collated_final_angles_more_realizations_' + '%sGHz.pkl'%band),'rb') as f:
        angle_data = pickle.load(f)

    for jfield, field in enumerate(field_list):
        plt.figure()
        obsids = []
        angles = []
        for obs in ids:
            if subfields[obs] == field:
                angles.append(angle_data[obs]['angle'])
                obsids.append(obs)
        angles = np.array(angles)
        obsids = np.array(obsids)

        pol_angle = np.zeros(len(time_edges) - 1)
        pol_error = np.zeros(len(pol_angle))
        for jtime in range(len(time_edges) - 1):
            angles_this_time = angles[(obsids>time_edges[jtime]) & (obsids<time_edges[jtime+1])]
            pol_angle[jtime] = np.mean(angles_this_time)
            pol_error[jtime] = np.std(angles_this_time) / np.sqrt(len(angles_this_time))
        plt.errorbar(np.arange(len(pol_angle)),
                     pol_angle*180/np.pi, yerr=pol_error*180/np.pi,
                     linestyle='none', marker='o',
                     label='{}'.format(field))
        plt.xlabel('month')
        plt.ylabel('pol angle [deg]')
        plt.legend()
        plt.title('{}'.format(field))
        plt.tight_layout()
        plt.savefig('pol_angle_bymonth_{}_{}.png'.format(field, band))
        plt.close()

# plot averaged angles by month with some kind of point-source masking
for band in [90, 150]:
    for field in ['ra0hdec-52.25']:
        fnames = np.sort(glob('/sptgrid/user/kferguson/axion_angles_mask_focus_quasar/'
                              'angles_{}_*_{}GHz.pkl'.format(field, band)))
        obsids_new = []
        angles_new = []
        for fname in fnames:
            print(fname)
            with open(fname, 'rb') as f:
                obsids_new.append(int(os.path.basename(fname).split('_')[2]))
                d = pickle.load(f)
                angles_new.append(d['obs'])
        obsids_new = np.array(obsids_new)
        angles_new = np.array(angles_new)

        pol_angle = np.zeros(len(time_edges) - 1)
        pol_error = np.zeros(len(pol_angle))
        for jtime in range(len(time_edges) - 1):
            angles_this_time = angles_new[(obsids_new>time_edges[jtime]) & (obsids_new<time_edges[jtime+1])]
            pol_angle[jtime] = np.mean(angles_this_time)
            pol_error[jtime] = np.std(angles_this_time) / np.sqrt(len(angles_this_time))

        plt.figure()
        plt.errorbar(np.arange(len(pol_angle)),
                    pol_angle*180/np.pi, yerr=pol_error*180/np.pi,
                    linestyle='none', marker='o',
                    label='{}'.format(field))
        plt.xlabel('month')
        plt.ylabel('pol angle [deg]')
        plt.legend()
        plt.title('{}'.format(field))
        plt.tight_layout()
        plt.savefig('pol_angle_withmask_bymonth_{}_{}.png'.format(field, band))
        plt.close()
