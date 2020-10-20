import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import os.path
import re

datadir = '/sptgrid/user/adama/20201006_signflip_noise_tests/'
fnames = glob('{}/calc-polangle*.pkl'.format(datadir))

mean_pol_angles = {}
std_pol_angles = {}
for fname in fnames:
    substrings = re.split('-|_',os.path.basename(fname))
    field = '{}-{}'.format(substrings[2], substrings[3])
    if field not in mean_pol_angles:
        mean_pol_angles[field] = []
        std_pol_angles[field] = []

    with open(fname, 'rb') as f:
        d = pickle.load(f)
    pol_angles = [x['combined_*150GHz']*180/np.pi for x in d]
    mean_pol_angles[field].append(np.mean(pol_angles))
    std_pol_angles[field].append(np.std(pol_angles))

    if True:
        plt.hist(pol_angles, bins=np.linspace(-0.3, 0.3, 51), histtype='step')
        #plt.title(fname)
        plt.xlabel('pol. angle [deg]')
        plt.tight_layout()
        plt.savefig('figures/{}.png'.format(os.path.splitext(os.path.basename(fname))[0]), dpi=200)
        plt.clf()

for field in mean_pol_angles:
    plt.figure(figsize=(6,4))
    plt.errorbar(np.arange(len(mean_pol_angles[field])), 
                 mean_pol_angles[field], std_pol_angles[field],
                 linestyle='none') #, marker='o')
    plt.xlabel('observation index')
    plt.ylabel('pol. angle [deg] in noise-free simulation')
    plt.title(field)
    plt.ylim([-0.2, 0.2])
    plt.tight_layout()
    plt.savefig('figures/pol_angle_{}_by_obsindex.png'.format(field), dpi=200)
    plt.clf()
