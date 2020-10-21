import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration, maps
import os.path
from glob import glob
import axion_utils
import pickle


noise_dir = '/sptgrid/user/adama/signflip_noise_test4'
mockobs_dir = '/sptgrid/user/adama/nosignflip_mockobs_noiseless_0000'
obsid_list = 'grid_submit_noise_sims/ra0hdec-44.75_150GHz_obsids_test_big.txt'
obsids = np.loadtxt(obsid_list)

obs_bands = ['90GHz', '150GHz', '220GHz']
pol_angles = {}

for obsid in obsids:
    print('{:.0f}'.format(obsid))
    
    # load all template maps
    sim_fnames = glob(os.path.join(mockobs_dir, '*{:.0f}*.g3.gz'.format(obsid)))
    sim_frames = {}
    for sim_fname in sim_fnames:
        for fr in core.G3File(sim_fname):
            if 'Id' in fr and fr['Id'] in obs_bands and 'T' in fr:
                m = np.array(fr['T'])
                if len(m[m!=0])>0:
                    sim_frames[fr['Id']] = fr
    
    pol_angles[obsid] = {band: [] for band in obs_bands}
    noise_fnames = np.sort(glob(os.path.join(noise_dir, '*{:.0f}*.g3.gz'.format(obsid))))
    for jfname, noise_fname in enumerate(noise_fnames):
        print('Loading {}'.format(noise_fname))
        for fr in core.G3File(noise_fname):
            if 'Id' in fr and fr['Id'] in obs_bands:
                rho = axion_utils.calculate_rho(fr, sim_frames[fr['Id']], freq_factor=1)
                pol_angles[obsid][fr['Id']].append(rho)

with open('pol_angles.pkl', 'wb') as f:
    pickle.dump(pol_angles, f)
