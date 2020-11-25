import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import os.path
import re
from scipy import stats

nskies = 5
equal_flags_weights = True
datadir = '/sptgrid/user/adama/20201006_signflip_noise_tests_2/' #sim_noisefull_signflip_perscan/'
simstub_path = '/sptgrid/user/kferguson/axion_perscan_maps_2019/simstub_ra0*_150GHz*g3.gz'
figs_dir = 'figures'
simnames = ['noisefree'] #['noisefull', 'noisefree']


# build complete list / indexing of obsids that were processed
simstub_fnames = glob(simstub_path)
obsids_all = {}
for fname in simstub_fnames:
    field = re.search('simstub_(.+?)_', fname).group(1)
    if field not in obsids_all:
        obsids_all[field] = []

    obsid = int(re.search('GHz_(.+?).g3.gz', fname).group(1))
    obsids_all[field].append(obsid)

for field in obsids_all:
    obsids_all[field] = np.sort(obsids_all[field])


# analyze grid output
for jsky in range(nskies):
    fnames_to_glob = [#'{}/calc_polangle_noisefull_ra0hdec*.pkl'.format(datadir),
                      '{}/calc_polangle_ra0hdec*{:04d}.pkl'.format(datadir, jsky)]

    mean_pol_angles = {}
    mean_pol_angles_byobsid = {}
    std_pol_angles_byobsid = {}
    for fname_to_glob, simname in zip(fnames_to_glob, simnames):
        fnames = np.sort(glob(fname_to_glob))
        fig_stub_name = 'pol_angle_{}'.format(simname)

        mean_pol_angles[simname] = {}
        mean_pol_angles_byobsid[simname] = {}
        std_pol_angles_byobsid[simname] = {}
        jobs = {}
        nsims = {}
        for fname in fnames:
            if 'noisefull' in fname:
                substrings = re.split('-|_',os.path.basename(fname))
                field = '{}-{}'.format(substrings[-4], substrings[-3])
                substrings = re.split('_|\.',os.path.basename(fname))
                obsid = int(substrings[-2])
            else:
                substrings = re.split('-|_',os.path.basename(fname))
                field = '{}-{}'.format(substrings[-5], substrings[-4])
                substrings = re.split('_|\.',os.path.basename(fname))
                obsid = int(substrings[-3])

            if field not in mean_pol_angles[simname]:
                mean_pol_angles[simname][field] = []
                mean_pol_angles_byobsid[simname][field] = {}
                std_pol_angles_byobsid[simname][field] = {}
                jobs[field] = 0

            with open(fname, 'rb') as f:
                d = pickle.load(f)
            pol_angles = [x['combined_*150GHz']*180/np.pi for x in d]
            mean_pol_angles[simname][field].append(np.mean(pol_angles))
            mean_pol_angles_byobsid[simname][field][obsid] = np.mean(pol_angles)
            std_pol_angles_byobsid[simname][field][obsid] = np.std(pol_angles)
            nsims[field] = len(pol_angles)

            if False:
                plt.hist(pol_angles, bins=np.linspace(-7, 7, 51), histtype='step')
                plt.title('{} {}'.format(field, re.split('\.|_',os.path.basename(fname))[-2]))
                plt.xlabel('pol. angle [deg]')
                plt.tight_layout()
                plt.savefig('{}/{}_{}.png'.format(figs_dir, os.path.splitext(os.path.basename(fname))[0][:-9], jobs[field]), dpi=200)
                plt.close()

            jobs[field] += 1

        for field in mean_pol_angles_byobsid[simname]:
            plt.figure(figsize=(6,4))
            indexes_to_plot         = [np.arange(len(obsids_all[field]))[obsids_all[field]==obsid][0] \
                                       for obsid in mean_pol_angles_byobsid[simname][field]]
            mean_pol_angles_to_plot = [mean_pol_angles_byobsid[simname][field][obsid] \
                                       for obsid in mean_pol_angles_byobsid[simname][field]]
            std_pol_angles_to_plot  = [std_pol_angles_byobsid[simname][field][obsid] \
                                       for obsid in std_pol_angles_byobsid[simname][field]]
            std_err_to_plot = std_pol_angles_to_plot / np.sqrt(nsims[field])
            plt.errorbar(indexes_to_plot, 
                         mean_pol_angles_to_plot,
                         std_err_to_plot,
                         linestyle='none', marker='o')
            chi2 = np.sum(np.asarray(mean_pol_angles_to_plot)**2 / (np.asarray(std_err_to_plot)**2 / nsims[field]))
            ndf = len(mean_pol_angles[simname][field])
            pval = 1 - stats.chi2.cdf(chi2, ndf)
            plt.xlabel('observation index')
            plt.ylabel('pol. angle [deg] in noise-free simulation')
            plt.title('{}: $\chi^2$ / ndf = {:.1f} / {}, p = {:.3f}'.format(field, chi2, ndf, pval))
            plt.ylim([-0.5, 0.5])
            plt.tight_layout()
            plt.savefig('{}/{}_{}_{:04d}_by_obsindex.png'.format(figs_dir, fig_stub_name, field, jsky), dpi=200)
            plt.close()


            plt.figure(figsize=(6,4))
            plt.hist(std_pol_angles_to_plot, bins=np.linspace(0,4,31), histtype='step')
            plt.xlabel('std of pol. angle [deg]')
            plt.title('{}: median = {:.2f} deg'.format(field, np.median(std_pol_angles_to_plot)))
            plt.xlim([0,4])
            plt.tight_layout()
            plt.savefig('{}/{}_{}_{:04d}_pol_angle_std.png'.format(figs_dir, fig_stub_name, field, jsky), dpi=200)
            plt.close()


            plt.figure(figsize=(6,4))
            plt.hist(np.asarray(mean_pol_angles_to_plot) / np.asarray(std_err_to_plot),
                     bins=np.linspace(-4,4,31), histtype='step')
            plt.xlabel('mean pol. angle / $\sigma$(mean pol. angle) [$\sigma$]')
            plt.title('{}: $\sigma$ from 0deg pol. angle'.format(field))
            plt.xlim([-4, 4])
            plt.tight_layout()
            plt.savefig('{}/{}_{}_{:04d}_pol_angle_sigmas.png'.format(figs_dir, fig_stub_name, field, jsky), dpi=200)
            plt.close()


    # for field in mean_pol_angles_byobsid[simname]:
    #     plt.figure(figsize=(5,4))
    #     obsids_noisefree = np.array([k for k in mean_pol_angles_byobsid['noisefree'][field]])
    #     obsids_noisefull = np.array([k for k in mean_pol_angles_byobsid['noisefull'][field]])
    #     obsids_common = np.intersect1d(obsids_noisefree, obsids_noisefull)
    #     polangle_noisefree = np.array([mean_pol_angles_byobsid['noisefree'][field][obsid] for obsid in obsids_common])
    #     polangle_noisefull = np.array([mean_pol_angles_byobsid['noisefull'][field][obsid] for obsid in obsids_common])

    #     plt.plot(polangle_noisefree, polangle_noisefull, '.')
    #     plt.xlabel('noisefree pol. angle [deg]')
    #     plt.ylabel('noisefull pol. angle [deg]')
    #     plt.axis([-0.2, 0.2, -0.2, 0.2])
    #     rho, p = stats.pearsonr(polangle_noisefree, polangle_noisefull)

    #     plt.title('{} - $\\rho = {:.3f}$, $p = {:.2e}$'.format(field, rho, p)) #, $p = {:.2e}$
    #     plt.gca().set_aspect('equal')
    #     plt.tight_layout()
    #     plt.savefig('{}/pol_angle_{}_{:04d}_noisefreeVfull_pol_angle.png'.format(figs_dir, field, jsky), dpi=200)
    #     plt.close()

plt.plot()
