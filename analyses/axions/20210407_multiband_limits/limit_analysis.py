import numpy as np 
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy.stats import chi2

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='mode')

parser_sim = subparser.add_parser('bgconsistency')
parser_sim.add_argument('bkganglefile', action='store', type=str,
                        help='Name of file with bkg-only simluations.')
parser_sim.add_argument('dataanglefile', action='store', type=str,
                        help='Name of file with data (or signal-injected'
                        ' simulations).')
parser_sim.add_argument('--plot', action='store_true',
                        help='Make a plot showing the p-values as a function '
                        'of frequency.')

args = parser.parse_args()

if args.mode == 'bgconsistency':
    with open(args.bkganglefile, 'rb') as f:
        sim_data = pickle.load(f)

    with open(args.dataanglefile, 'rb') as f:
        data = pickle.load(f)

    # calculate delta chi2
    delta_chi2 = {}
    for freq, chi2_bestfit in sim_data['chi2(A=A_fit)'].items():
        delta_chi2[freq] = np.array(sim_data['chi2(A=0)']) - np.array(chi2_bestfit)
    freqs = np.array([freq for freq in delta_chi2])

    # distribution of test statistics
    max_delta_chi2 = np.zeros(len(delta_chi2[freqs[0]]))
    for j_realization in range(len(delta_chi2[freqs[0]])):
        delta_chi2_one_realization = np.array([delta_chi2[freq][j_realization] for freq in delta_chi2])
        max_delta_chi2[j_realization] = np.max(delta_chi2_one_realization)

    if args.plot:
        # distribution of delta chi2
        delta_chi2_up2sigma = np.array([np.percentile(delta_chi2[freq], 97.5) for freq in delta_chi2])
        delta_chi2_up1sigma = np.array([np.percentile(delta_chi2[freq], 84) for freq in delta_chi2])
        delta_chi2_down1sigma = np.array([np.percentile(delta_chi2[freq], 16) for freq in delta_chi2])
        delta_chi2_down2sigma = np.array([np.percentile(delta_chi2[freq], 2.5) for freq in delta_chi2])
        delta_chi2_median = np.array([np.median(delta_chi2[freq]) for freq in delta_chi2])
        
        plt.figure(1)
        plt.fill_between(freqs, delta_chi2_down2sigma, delta_chi2_up2sigma,
                        color=(0.95,0.95,0), label='$\pm 2\sigma$')
        plt.fill_between(freqs, delta_chi2_down1sigma, delta_chi2_up1sigma,
                        color=(0,0.9,0), label='$\pm 1\sigma$')
        plt.semilogx(freqs, delta_chi2_median, 'k--',
                    label='median $\Delta \chi^2$')
        # plot nominal quantiles
        plt.plot(freqs, chi2.ppf(0.025, df=2) * np.ones(len(freqs)), color='b', linewidth=0.5, linestyle='--')
        plt.plot(freqs, chi2.ppf(0.16, df=2) * np.ones(len(freqs)), color='b', linewidth=0.5, linestyle='--')
        plt.plot(freqs, chi2.ppf(0.50, df=2) * np.ones(len(freqs)), color='b', linewidth=0.5, linestyle='--')
        plt.plot(freqs, chi2.ppf(0.84, df=2) * np.ones(len(freqs)), color='b', linewidth=0.5, linestyle='--')
        plt.plot(freqs, chi2.ppf(0.975, df=2) * np.ones(len(freqs)), color='b', linewidth=0.5, linestyle='--')
        plt.xlabel('frequency [d$^{-1}$]')
        plt.ylabel('$\Delta \chi^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('delta_chi2.png', dpi=200)
        
        # pdf of test statistics
        plt.figure(2)
        plt.hist(max_delta_chi2, bins=26)
        plt.xlabel('$q_i$')
        plt.ylabel('# realizations')
        plt.tight_layout()
        plt.savefig('test_statistic_pdf.png', dpi=200)

        # p-value of test statistics
        max_delta_chi2_pdf, bin_edges = np.histogram(max_delta_chi2, bins=26)
        plt.figure(3)
        plt.plot(bin_edges[:-1], np.cumsum(max_delta_chi2_pdf) / len(max_delta_chi2))
        plt.xlabel('$q_i$')
        plt.ylabel('cdf(q_i)')
        plt.tight_layout()
        plt.savefig('test_statistic_cdf.png', dpi=200)
        