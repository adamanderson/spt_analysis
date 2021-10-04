from spt3g import core, maps, autoprocessing
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
from datetime import datetime

n_realizations = 10
seed = 234
scanify = autoprocessing.ScanifyDatabase(read_only=True)
data_path = '/sptlocal/user/kferguson'

# generate timestamp for filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# get obsids
with open(os.path.join(data_path, 'collated_final_angles_more_realizations_90GHz.pkl'), 'rb') as f:
    angle_pk = pickle.load(f)
ids = sorted(list(angle_pk.keys()))

# get subfields corresponding to each obsid
subfields = {}
for obsid in ids:
    subfields[obsid] = scanify.get_entries('.*', obsid, match_regex=True)[0][0]

# load the angle data
angle_data = {}
data_dist = {}
for jband, band in enumerate([90,150]):
    with open(os.path.join(data_path, 'collated_final_angles_more_realizations_' + '%sGHz.pkl'%band),'rb') as f:
        angle_data[band] = pickle.load(f)

    data_dist[band] = {'ra0hdec-%s'%dec: [] for dec in ['44.75','52.25','59.75','67.25']}
    for obs in ids:
        sub = subfields[obs]
        data_dist[band][sub].append(angle_data[band][obs]['angle'] / angle_data[band][obs]['unc'])

# actually generate the data
sim_data = {}
for j_realization in range(n_realizations):
    sim_data[j_realization] = {}
    for jband, band in enumerate([90,150]):
        for obsid in ids:
            sub = subfields[obsid]
            ind = np.random.randint(0, high=len(data_dist[band][sub]))
            sim_data[j_realization][obsid] = {}
            sim_data[j_realization][obsid]['unc'] = angle_data[band][obsid]['unc']
            sim_data[j_realization][obsid]['angle'] = angle_data[band][obsid]['unc'] * data_dist[band][sub][ind]

with open('sim_angles_{}.pkl'.format(timestamp), 'wb') as f:
    pickle.dump(sim_data, f)

    # plt.figure(jband+1)
    # times = np.array([obsid for obsid in sim_data.keys()])
    # angles = 180/np.pi*np.array([sim_data[obsid]['angle'] for obsid in sim_data.keys()])
    # plt.plot(times, angles, 'o')
    # plt.title('{} GHz; std = {:.2f}'.format(band, np.std(angles)))
