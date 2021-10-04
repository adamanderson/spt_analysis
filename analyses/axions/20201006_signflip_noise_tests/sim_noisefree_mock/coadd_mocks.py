# simple script to coadd mock observations generated from multiple mock skies
import os
from glob import glob
import numpy as np

nskies = 5
freq = '150GHz'
mock_obs_dir = '/sptgrid/user/adama/20201006_signflip_noise_tests_2'
coadd_command = 'python /home/adama/SPT/spt3g_software/std_processing/combining_maps.py'

for jsky in range(nskies):
    fnames = np.sort(glob(os.path.join(mock_obs_dir,
                                       'noisefree-mock-sims*{}*{:04d}.g3.gz'.format(freq, jsky))))
    command_list = fnames.tolist()
    command_list.insert(0, coadd_command)
    command_list.append('-o noisefree_mock_sims_{}_{:04d}.g3.gz'.format(freq, jsky))
    command_list.append('--map-id {}'.format(freq))
    command = ' '.join(command_list)
    print(command)
    os.system(command)

