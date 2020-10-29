import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import os.path
import re
from scipy import stats

equal_flags_weights = True
datadir = '/sptgrid/user/adama/20201006_signflip_noise_tests/' #sim_noisefull_signflip_perscan/'
figs_dir = 'figures'
fnames = glob('{}/calc-polangle-noisefull_ra0hdec*.pkl'.format(datadir))
fig_stub_name = 'pol_angle_noisefull'

mean_pol_angles = {}
std_pol_angles = {}
jobs = {}
nsims = {}
for fname in fnames:
    substrings = re.split('-|_',os.path.basename(fname))
    field = '{}-{}'.format(substrings[-4], substrings[-3])
    if field not in mean_pol_angles:
        mean_pol_angles[field] = []
        std_pol_angles[field] = []
        jobs[field] = 0

    with open(fname, 'rb') as f:
        d = pickle.load(f)
    pol_angles = [x['combined_*150GHz']*180/np.pi for x in d]
    mean_pol_angles[field].append(np.mean(pol_angles))
    std_pol_angles[field].append(np.std(pol_angles))
    nsims[field] = len(pol_angles)

    if False:
        plt.hist(pol_angles, bins=np.linspace(-7, 7, 51), histtype='step')
        plt.title('{} {}'.format(field, re.split('\.|_',os.path.basename(fname))[-2]))
        plt.xlabel('pol. angle [deg]')
        plt.tight_layout()
        plt.savefig('{}/{}_{}.png'.format(figs_dir, os.path.splitext(os.path.basename(fname))[0][:-9], jobs[field]), dpi=200)
        plt.clf()
    
    jobs[field] += 1

for field in mean_pol_angles:
    plt.figure(figsize=(6,4))
    plt.errorbar(np.arange(len(mean_pol_angles[field])), 
                 mean_pol_angles[field],
                 np.asarray(std_pol_angles[field]) / np.sqrt(nsims[field]),
                 linestyle='none', marker='o')
    chi2 = np.sum(np.asarray(mean_pol_angles[field])**2 / (np.asarray(std_pol_angles[field])**2 / nsims[field]))
    ndf = len(mean_pol_angles[field])
    pval = stats.chi2.cdf(chi2, ndf)
    plt.xlabel('observation index')
    plt.ylabel('pol. angle [deg] in noise-free simulation')
    plt.title('{}: $\chi^2$ / ndf = {:.1f} / {}, p = {:.3f}'.format(field, chi2, ndf, pval))
    plt.ylim([-0.5, 0.5])
    plt.tight_layout()
    plt.savefig('{}/{}_{}_by_obsindex.png'.format(figs_dir, fig_stub_name, field), dpi=200)
    plt.clf()

for field in mean_pol_angles:
    plt.figure(figsize=(6,4))
    plt.hist(std_pol_angles[field], bins=np.linspace(0,4,31), histtype='step')
    plt.xlabel('std of pol. angle [deg]')
    plt.title('{}: median = {:.2f} deg'.format(field, np.median(std_pol_angles[field])))
    plt.xlim([0,4])
    plt.tight_layout()
    plt.savefig('{}/{}_{}_pol_angle_std.png'.format(figs_dir, fig_stub_name, field), dpi=200)
    plt.clf()

    plt.figure(figsize=(6,4))
    plt.hist(np.asarray(mean_pol_angles[field]) / (np.asarray(std_pol_angles[field]) / np.sqrt(nsims[field])),
             bins=np.linspace(-4,4,31), histtype='step')
    plt.xlabel('std of pol. angle [deg]')
    plt.title('{}: median = {:.2f} deg'.format(field, np.median(std_pol_angles[field])))
    plt.xlim([-4, 4])
    plt.tight_layout()
    plt.savefig('{}/{}_{}_pol_angle_sigmas.png'.format(figs_dir, fig_stub_name, field), dpi=200)
    plt.clf()
