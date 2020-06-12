# Simple script to calculate the polarization angles from all observations in
# the 2018 data, after Kyle's coadding of the map jack-knife splits.

from glob import glob
import re
import os.path
import axion_utils
import pickle
from spt3g import core

# simulation
map_sims_dir = '/sptlocal/user/kferguson/mock_observed_cmb_maps_simstubs_from_data/'
map_sims_fnames = glob(os.path.join(map_sims_dir, '*'))

map_sims_fnames_dict = {}
for fname in map_sims_fnames:
    result = re.search('mock_observed_sim_(.*?)_(.*?)GHz', fname)
    obsid = int(result.group(1))
    band = int(result.group(2))
    
    if obsid not in map_sims_fnames_dict:
        map_sims_fnames_dict[obsid] = {}
        
    map_sims_fnames_dict[obsid][band] = fname

# data, full coadds
map_data_dir = '/sptlocal/user/kferguson/full_daniel_maps/'
map_data_fnames = glob(os.path.join(map_data_dir, '*g3.gz'))

map_data_fnames_dict = {}
for fname in map_data_fnames:
    print(fname)
    result = re.search('ra0hdec(.*?)_(.*?)GHz_(.*?)_map.g3.gz', fname)
    band = int(result.group(2))
    obsid = int(result.group(3))

    if obsid not in map_data_fnames_dict:
        map_data_fnames_dict[obsid] = {}

    map_data_fnames_dict[obsid][band] = fname


rho_fit_dict = {}
rho_err_dict = {}
fnames_test = list(map_data_fnames_dict.keys())
for obsid in fnames_test: #map_data_fnames_dict:
    if obsid in map_sims_fnames_dict:
        print(obsid)
        rho_fit_dict[obsid] = {}
        rho_err_dict[obsid] = {}
        for band in [90, 150, 220]:
            print(band)

            # load sims
            for fr in core.G3File(map_sims_fnames_dict[obsid][band]):
                if fr.type == core.G3FrameType.Map and fr['Id'] == '{}GHz'.format(band):
                    sim_frame = fr

            # load data
            data_frame = list(core.G3File(map_data_fnames_dict[obsid][band]))[0]

            rho, rho_err = axion_utils.calculate_rho(data_frame, sim_frame, freq_factor=8.7, return_err=True)
            rho_fit_dict[obsid][band] = rho
            rho_err_dict[obsid][band] = rho_err


with open('angle_fit_data.pkl', 'wb') as f:
    save_dict = {'angles_fit': rho_fit_dict, 'angles_err': rho_err_dict}
    pickle.dump(save_dict, f)
