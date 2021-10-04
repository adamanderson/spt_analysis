import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
from spt3g import autoprocessing


# get obsids
scanify = autoprocessing.ScanifyDatabase(read_only=True)
data_path = '/sptlocal/user/kferguson'
with open(os.path.join(data_path, 'collated_final_angles_more_realizations_90GHz.pkl'), 'rb') as f:
    angle_pk = pickle.load(f)
ids = np.sort(list(angle_pk.keys()))

# get subfields corresponding to each obsid
subfields = {}
for obsid in ids:
    subfields[obsid] = scanify.get_entries('.*', obsid, match_regex=True)[0][0]
subfields_plot = np.array([subfields[obsid] for obsid in ids])

# plot phases
phase_list = {'ra0hdec-44.75': 108.4702*np.pi/180,
              'ra0hdec-52.25': 142.2063*np.pi/180,
              'ra0hdec-59.75': 154.9772*np.pi/180,
              'ra0hdec-67.25': 152.8479*np.pi/180}
freq_list  = {'ra0hdec-44.75': 1.1960/(24*60*60),
              'ra0hdec-52.25': 1.1960/(24*60*60),
              'ra0hdec-59.75': 1.2410/(24*60*60),
              'ra0hdec-67.25': 1.2410/(24*60*60)}

for field in phase_list.keys():
    plt.figure()
    zs = np.zeros(len(ids[subfields_plot == field]))
    plt.plot(ids[subfields_plot == field], zs, 'o')
    plot_times = np.linspace(np.min(ids), np.max(ids), 1000000)
    plt.plot(plot_times, np.sin(2*np.pi*freq_list[field]*plot_times + phase_list[field]))
    plt.xlim([8e7, 8.1e7])
    plt.title(field)
    plt.savefig('phase_check_{}.png'.format(field))
    plt.close()


plt.figure(figsize=(16,12))
obs_times = ids[(subfields_plot == 'ra0hdec-44.75')]
plt.plot(obs_times, np.sin(2*np.pi*1.2410/(24*60*60)*obs_times + 154.4122*np.pi/180), 'o')
obs_times = ids[(subfields_plot == 'ra0hdec-52.25')]
plt.plot(obs_times, np.sin(2*np.pi*1.2410/(24*60*60)*obs_times + 154.4122*np.pi/180), 'o')
# plt.xlim([7.0e7, 9.0e7])
plt.title(field)
plt.savefig('phase_check_ra0hdec-44.75_ra0hdec-52.25.png')
plt.close()


plt.figure(figsize=(16,12))
obs_times = ids[(subfields_plot == 'ra0hdec-59.75')]
plt.plot(obs_times, np.sin(2*np.pi*1.2410/(24*60*60)*obs_times + 154.4122*np.pi/180), 'o')
obs_times = ids[(subfields_plot == 'ra0hdec-67.25')]
plt.plot(obs_times, np.sin(2*np.pi*1.2410/(24*60*60)*obs_times + 154.4122*np.pi/180), 'o')
# plt.xlim([7.0e7, 9.0e7])
plt.title(field)
plt.savefig('phase_check_ra0hdec-59.75_ra0hdec-67.25.png')
plt.close()