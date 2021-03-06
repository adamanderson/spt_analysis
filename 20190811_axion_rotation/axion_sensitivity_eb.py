import numpy as np
import matplotlib.pyplot as plt
import camb
import argparse as ap
from scipy.optimize import minimize, newton
from scipy.stats import chi2, sigmaclip
from timedomainfitting import *

P0 = ap.ArgumentParser(description='Script for estimating the sensitivity of '
                       'the axion rotation analysis using the EB estimator.',
                       formatter_class=ap.ArgumentDefaultsHelpFormatter)
S0 = P0.add_subparsers(dest='mode', metavar='MODE', title='subcommands',
                       help='Function to perform. For help, call: '
                       '%(prog)s %(metavar)s -h')

SP0 = S0.add_parser('singlemap', help='Simulate a single map and estimation of '
                   'its polarization angle.',
                   formatter_class=ap.ArgumentDefaultsHelpFormatter)
SP0.add_argument('--noise-per-obs', action='store', type=float, default=150,
                 help='Noise level per map in units of [uK-arcmin].')
SP0.add_argument('--n-sims', action='store', type=int, default=2000,
                 help='Number of simulated Cls to generate.')
SP0.add_argument('--field-size', action='store', type=int, default=500,
                 help='Size of field in square degrees.')
SP0.add_argument('--lmin', action='store', type=int, default=100,
                 help='lmin to use for analysis.')
SP0.add_argument('--lmax', action='store', type=int, default=2000,
                 help='lmax to use for analysis.')
SP0.add_argument('--cosmic-variance', action='store_true', 
                 help='Include cosmic variance..')

SP1 = S0.add_parser('timeseries', help='Simulate data as a function of time '
                    'and optionally fit it to a sinusoid.',
                    formatter_class=ap.ArgumentDefaultsHelpFormatter)
SP1.add_argument('--noise-per-obs', action='store', type=float, default=150,
                 help='Noise level per map in units of [uK-arcmin].')
SP1.add_argument('--field-size', action='store', type=int, default=500,
                 help='Size of field in square degrees.')
SP1.add_argument('--n-bundles', action='store', type=int, default=1000,
                 help='Number observations to simulate.')
SP1.add_argument('--n-sims', action='store', type=int, default=1000,
                 help='Number of simulations to use for covariance estimation.')
SP1.add_argument('--obs-per-bundle', action='store', type=int, default=1,
                 help='Coadd N observations into bundles, and then fit for '
                 'a polarization angle for each bundle.')
SP1.add_argument('--lmin', action='store', type=int, default=100,
                 help='lmin to use for analysis.')
SP1.add_argument('--lmax', action='store', type=int, default=3000,
                 help='lmax to use for analysis.')
SP1.add_argument('--time-domain-fit', action='store_true',
                 help='Do profile-likelihood fit of time-domain data.')
SP1.add_argument('--n-expectation-sims', action='store', type=int, default=1,
                 help='Instead of simulating a single realization of the '
                 'time-domain data, simulate many of them and compute '
                 'the expectation of the fit parameters.')
SP1.add_argument('--simulate-limits', action='store_true',
                 help='Simulate the +/-1sigma upper and lower limits on the '
                 'oscillation amplitude using the profile likelihood.')
args = P0.parse_args()


# Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

# Calculate results for these parameters
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
totCL = powers['total']
unlensedCL = powers['unlensed_scalar']
ls = np.arange(totCL.shape[0])
ells = ls[1:]
Cl_theory = {'TT': unlensedCL[1:,0],
             'EE': unlensedCL[1:,1],
             'BB': unlensedCL[1:,2],
             'TE': unlensedCL[1:,3]}


def generate_Cl(spectrum, ells, Cl_theory, f_sky, lmin, lmax, map_noise_T,
                beam_fwhm, cosmic_variance=False):
    '''
    Generate a Gaussian realization of the Cl's from a cosmology.

    Parameters
    ----------
    spectrum : str
        2-point spectrum for which to generate realization: 'TT', 'TE', etc.
    ells : numpy array
        Array of multipole number ells corresponding to the theory Cl's below.
    Cl_theory : Python dict
        Dictionary with theory Cl's, indexed by 'TT', 'TE', 'EE', 'BB'.
    f_sky : float
        Sky fraction.
    lmin : float
        Minimum ell to simulate.
    lmax : float
        Maximum ell to simulate.
    map_noise_T : float
        Temperature map noise level.
    beam_fwhm : float
        FWHM of the beam.

    Returns
    -------
    l_range : numpy array
        Multipoles corresponding to the Cl realization.
    Cl_realization : numpy array
        Realization of the Cl's given the map noise level passed as an argument.
    '''

    ells_cut = (ells>lmin) & (ells<lmax)
    l_range = ells[ells_cut]
    
    noise_weight_T = (map_noise_T * np.pi / 10800)**2
    noise_weight_P = (map_noise_T*np.sqrt(2.) * np.pi / 10800)**2
    sigma_beam = beam_fwhm / np.sqrt(8*np.log(2.)) * np.pi / 10800
    beam_weight = np.exp(-1.*l_range**2. * sigma_beam**2.)
    
    if cosmic_variance:
        if spectrum == 'TE':
            var_factor = (Cl_theory['TE'][ells_cut]**2 + \
                          (Cl_theory['TT'][ells_cut] + noise_weight_T * beam_weight) * \
                          (Cl_theory['EE'][ells_cut] + noise_weight_P * beam_weight))
            Cl_mean = Cl_theory['TE'][ells_cut]
        elif spectrum == 'EB':
            var_factor = ((Cl_theory['BB'][ells_cut] + noise_weight_P * beam_weight) * \
                          (Cl_theory['EE'][ells_cut] + noise_weight_P * beam_weight))
            Cl_mean = np.zeros(Cl_theory['TT'][ells_cut].shape)
        elif spectrum == 'TB':
            var_factor = ((Cl_theory['TT'][ells_cut] + noise_weight_T * beam_weight) * \
                          (Cl_theory['BB'][ells_cut] + noise_weight_P * beam_weight))
            Cl_mean = np.zeros(Cl_theory['TT'][ells_cut].shape)
        else:
            var_factor = (Cl_theory[spectrum][ells_cut] + noise_weight_P * beam_weight)**2
            Cl_mean = Cl_theory[spectrum][ells_cut]
    else:
        if spectrum == 'TE':
            var_factor = (noise_weight_T * beam_weight) * (noise_weight_P * beam_weight)
            Cl_mean = Cl_theory['TE'][ells_cut]
        elif spectrum == 'EB':
            var_factor = (noise_weight_P * beam_weight) * (noise_weight_P * beam_weight)
            Cl_mean = np.zeros(Cl_theory['TT'][ells_cut].shape)
        elif spectrum == 'TB':
            var_factor = (noise_weight_T * beam_weight) * (noise_weight_P * beam_weight)
            Cl_mean = np.zeros(Cl_theory['TT'][ells_cut].shape)
        else:
            var_factor = (noise_weight_P * beam_weight)**2
            Cl_mean = Cl_theory[spectrum][ells_cut]

    Cl_sigma = np.sqrt(2 / ((2*l_range + 1)*f_sky) * var_factor)
    
    # generate a gaussian realization of all the Cl's, then
    # average them all into a single bin over the ell range
    # specified in the argument
    Cl_realization = np.random.normal(loc=Cl_mean, scale=Cl_sigma)
    
    return l_range, Cl_realization


def generate_Cl_averaged(spectrum, ells, Cl_theory, f_sky, lmin, lmax,
                         map_noise_T, beam_fwhm):
    '''
    Generate Cl's averaged over [lmin, lmax]. This is the equivalent of 
    "binning" the bandpowers into a single bin.

    Parameters
    ----------
    spectrum : str
        2-point spectrum for which to generate realization: 'TT', 'TE', etc.
    ells : numpy array
        Array of multipole number ells corresponding to the theory Cl's below.
    Cl_theory : Python dict
        Dictionary with theory Cl's, indexed by 'TT', 'TE', 'EE', 'BB'.
    f_sky : float
        Sky fraction.
    lmin : float
        Minimum ell to simulate.
    lmax : float
        Maximum ell to simulate.
    map_noise_T : float
        Temperature map noise level.
    beam_fwhm : float
        FWHM of the beam.

    Returns
    -------
    Cl_averaged : float
        Cl's averaged over the full ell range.
    '''
    spectra_types = ['TT', 'TE', 'TB', 'EE', 'EB', 'BB']
    if spectrum not in spectra_types:
        raise ValueError('{} is not a valid spectrum type.'.format(spectrum))

    l_range, Cl_realization = generate_Cl(spectrum, ells, Cl_theory, f_sky,
                                          lmin, lmax, map_noise_T, beam_fwhm)
    Cl_averaged = np.mean(Cl_realization)
    return Cl_averaged


def pol_chi2(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB):
    '''
    Chi-square of the global polarization angle for a map with angular
    multipoles:
        (Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB),
    using the Dl_TB and Dl_EB quantities:
        D^{TB}_\ell = C^{TB}_\ell cos(2 alpha) - C^{TE}_\ell sin(2 alpha)
        D^{EB}_\ell = C^{EB}_\ell cos(4 alpha) - (1/2) ( C^{EE}_\ell - C^{BB}_\ell ) sin(4 alpha).
    In standard cosmology, D^{TB}_\ell = D^{EB}_\ell = 0.

    Parameters
    ----------
    x : float or numpy array
        Polarization angles at which to evaluate chi-square.
    Cl_TB : numpy array
        TB multipoles.
    Cl_TE : numpy array
        TE multipoles.
    Cl_EB : numpy array
        EB multipoles.
    Cl_EE : numpy array
        EE multipoles.
    Cl_BB : numpy array
        BB multipoles.
    var_Dl_TB : numpy array
        Variance of the Dl_TB, used to normalize the chi-square.
    var_Dl_TE : numpy array
        Variance of the Dl_TE, used to normalize the chi-square.

    Returns
    -------
    chi2 : float 
        The chi-square statistic.
    '''

    if not isinstance(x, np.ndarray):
        alpha_array = np.array([x])
    else:
        alpha_array = x
    chi2_array = []
    for alpha in alpha_array:
        Dl_TB = Cl_TB * np.cos(2*alpha) - Cl_TE * np.sin(2*alpha)
        Dl_EB = Cl_EB * np.cos(4*alpha) - (1./2.) * (Cl_EE - Cl_BB) * np.sin(4*alpha)
        data = np.array([Dl_TB, Dl_EB]) 
        chi2_array.append(Dl_TB**2 / var_Dl_TB + Dl_EB**2 / var_Dl_EB)
    chi2_array = np.array(chi2_array)
    return chi2_array


def delta_chi2_plus(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB,
                    var_Dl_EB, func_min, offset):
    return pol_chi2(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB) - \
        func_min - offset


def neg2logL_profiled_plus(A, period, times, angle, angle_error, func_min,
                           offset):
    fval, profiled_phase = neg2logL_profiled(A, period, times, angle, angle_error) - \
                           func_min - offset
    return fval


# simulated maps for either the single-map mode or for
# covariance estimation in the time=series mode
Cl_random = {}
for spectrum in ['TB', 'TE', 'EB', 'EE', 'BB']:
    Cl_random[spectrum] = np.zeros(args.n_sims)

if args.mode == 'singlemap':
    # generate the simulated spectra
    for jsim in range(args.n_sims):
        for spectrum in Cl_random:
            Cl_random[spectrum][jsim] = generate_Cl_averaged(spectrum, ells,
                                                             Cl_theory,
                                                             f_sky=args.field_size/41000,
                                                             lmin=args.lmin, lmax=args.lmax,
                                                             map_noise_T=args.noise_per_obs,
                                                             beam_fwhm=1.2)

    chi2_sims = np.zeros(args.n_sims)
    delta_chi2_sims = np.zeros(args.n_sims)
    angle_fit = np.zeros(args.n_sims)
    angle_up1sigma = np.zeros(args.n_sims)
    angle_down1sigma = np.zeros(args.n_sims)

    # minimize the chi-square and estimate the polarization angle and uncertainty
    for jsim in range(args.n_sims):
        out = minimize(pol_chi2, 0, args=(Cl_random['TB'][jsim],
                                          Cl_random['TE'][jsim],
                                          Cl_random['EB'][jsim],
                                          Cl_random['EE'][jsim],
                                          Cl_random['BB'][jsim],
                                          np.var(Cl_random['TB']),
                                          np.var(Cl_random['EB'])),
                       method='Powell')
        angle_fit[jsim] = out.x
        chi2_sims[jsim] = out.fun

        try:
            delta_chi2_sims[jsim]  = pol_chi2(0,
                                              Cl_random['TB'][jsim],
                                              Cl_random['TE'][jsim],
                                              Cl_random['EB'][jsim],
                                              Cl_random['EE'][jsim],
                                              Cl_random['BB'][jsim],
                                              np.var(Cl_random['TB']),
                                              np.var(Cl_random['EB'])) - chi2_sims[jsim]
            angle_up1sigma[jsim]   = newton(delta_chi2_plus,
                                            x0=angle_fit[jsim] + 0.01,
                                            args=(Cl_random['TB'][jsim],
                                                  Cl_random['TE'][jsim],
                                                  Cl_random['EB'][jsim],
                                                  Cl_random['EE'][jsim],
                                                  Cl_random['BB'][jsim],
                                                  np.var(Cl_random['TB']),
                                                  np.var(Cl_random['EB']),
                                                  chi2_sims[jsim], 1))
            angle_down1sigma[jsim] = newton(delta_chi2_plus,
                                            x0=angle_fit[jsim] - 0.01,
                                            args=(Cl_random['TB'][jsim],
                                                  Cl_random['TE'][jsim],
                                                  Cl_random['EB'][jsim],
                                                  Cl_random['EE'][jsim],
                                                  Cl_random['BB'][jsim],
                                                  np.var(Cl_random['TB']),
                                                  np.var(Cl_random['EB']),
                                                  chi2_sims[jsim], 1))
        except RuntimeError:
            print('Failed to estimate error bar. Skipping simulation.')

    plt.figure(1)
    hist_range = 10 #np.std(angle_up1sigma) * 4 * 180/np.pi
    _ = plt.hist(angle_up1sigma * 180/np.pi, histtype='step',
                 bins=np.linspace(-1*hist_range,hist_range,51),
                 label='$1\sigma$ upper limit')
    _ = plt.hist(angle_down1sigma * 180/np.pi, histtype='step',
                 bins=np.linspace(-1*hist_range,hist_range,51),
                 label='$1\sigma$ lower limit')
    _ = plt.hist((angle_up1sigma - angle_down1sigma) * 180/np.pi, histtype='step',
                 bins=np.linspace(-1*hist_range,hist_range,51),
                 label='$1\sigma$ error bar')
    plt.legend()
    plt.xlabel('polarization angle error [deg]')
    plt.ylabel('number of simulations')
    plt.tight_layout()
    plt.savefig('angle_hist.png', dpi=150)

elif args.mode == 'timeseries':
    chi2_sims         = np.zeros((args.n_bundles, args.n_expectation_sims))
    delta_chi2_sims   = np.zeros((args.n_bundles, args.n_expectation_sims))
    angle_fit         = np.zeros((args.n_bundles, args.n_expectation_sims))
    angle_up1sigma    = np.zeros((args.n_bundles, args.n_expectation_sims))
    angle_down1sigma  = np.zeros((args.n_bundles, args.n_expectation_sims))
    angle_error       = np.zeros((args.n_bundles, args.n_expectation_sims))
    
    periods_to_test       = np.arange(4, 80)
    amplitude_best_fit    = np.zeros((len(periods_to_test), args.n_expectation_sims))
    amplitude_up1sigma    = np.zeros((len(periods_to_test), args.n_expectation_sims))
    amplitude_down1sigma  = np.zeros((len(periods_to_test), args.n_expectation_sims))

    Cl_per_bundle = {}
    for spectrum in ['TB', 'TE', 'EB', 'EE', 'BB']:
        Cl_per_bundle[spectrum] = {}
        for jbundle in range(args.n_bundles):
            Cl_per_bundle[spectrum][jbundle] = np.zeros(args.n_expectation_sims)
    noise_per_bundle = args.noise_per_obs / np.sqrt(args.obs_per_bundle)

    # generate simulations to compute the covariance
    for jsim in range(args.n_sims):
        for spectrum in Cl_random:
            Cl_random[spectrum][jsim] = generate_Cl_averaged(spectrum, ells,
                                                             Cl_theory,
                                                             f_sky=args.field_size/41000,
                                                             lmin=args.lmin, lmax=args.lmax,
                                                             map_noise_T=noise_per_bundle,
                                                             beam_fwhm=1.2)


    # generate the simulated spectra
    for jbundle in range(args.n_bundles):
        for jsim in range(args.n_expectation_sims):
            for spectrum in Cl_per_bundle:
                Cl_per_bundle[spectrum][jbundle, jsim] = generate_Cl_averaged(spectrum, ells,
                                                                              Cl_theory,
                                                                              f_sky=args.field_size/41000,
                                                                              lmin=args.lmin, lmax=args.lmax,
                                                                              map_noise_T=noise_per_bundle,
                                                                              beam_fwhm=1.2)
            out = minimize(pol_chi2, 0, args=(Cl_per_bundle['TB'][jbundle, jsim],
                                              Cl_per_bundle['TE'][jbundle, jsim],
                                              Cl_per_bundle['EB'][jbundle, jsim],
                                              Cl_per_bundle['EE'][jbundle, jsim],
                                              Cl_per_bundle['BB'][jbundle, jsim],
                                              np.var(Cl_random['TB']),
                                              np.var(Cl_random['EB'])), method='Powell')
            angle_fit[jbundle, jsim] = out.x
            chi2_sims[jbundle, jsim] = out.fun

            angle_up1sigma[jbundle, jsim]   = newton(delta_chi2_plus,
                                               x0=angle_fit[jbundle, jsim] + 0.01,
                                               args=(Cl_per_bundle['TB'][jbundle, jsim],
                                                     Cl_per_bundle['TE'][jbundle, jsim],
                                                     Cl_per_bundle['EB'][jbundle, jsim],
                                                     Cl_per_bundle['EE'][jbundle, jsim],
                                                     Cl_per_bundle['BB'][jbundle, jsim],
                                                     np.var(Cl_random['TB']),
                                                     np.var(Cl_random['EB']),
                                                     chi2_sims[jbundle, jsim], 1.0))
            angle_down1sigma[jbundle, jsim] = newton(delta_chi2_plus,
                                               x0=angle_fit[jbundle, jsim] - 0.01,
                                               args=(Cl_per_bundle['TB'][jbundle, jsim],
                                                     Cl_per_bundle['TE'][jbundle, jsim],
                                                     Cl_per_bundle['EB'][jbundle, jsim],
                                                     Cl_per_bundle['EE'][jbundle, jsim],
                                                     Cl_per_bundle['BB'][jbundle, jsim],
                                                     np.var(Cl_random['TB']),
                                                     np.var(Cl_random['EB']),
                                                     chi2_sims[jbundle, jsim], 1.0))

            # symmetrize the error
            angle_error[jbundle, jsim] = (angle_up1sigma[jbundle, jsim] - \
                                          angle_down1sigma[jbundle, jsim]) / 2
            times = np.arange(args.n_bundles)


    if args.time_domain_fit:
        for jperiod, period in enumerate(periods_to_test):
            amplitudes = np.linspace(0, 2e-2)
            for jsim in range(args.n_expectation_sims):
                neg2logL_global_fval, neg2logL_global_params = neg2logL_global_fit(period, times,
                                                                                   angle_fit[:, jsim],
                                                                                   angle_error[:, jsim])
                amplitude_best_fit[jperiod, jsim]    = neg2logL_global_params[0]
                if args.simulate_limits:
                    amplitude_up1sigma[jperiod, jsim]    = newton(neg2logL_profiled_plus,
                                                                x0=amplitude_best_fit[jperiod, jsim]+0.1,
                                                                args=(period, times,
                                                                      angle_fit[:, jsim],
                                                                      angle_error[:, jsim],
                                                                      neg2logL_global_fval,
                                                                      1.0))
                    amplitude_down1sigma[jperiod, jsim]  = newton(neg2logL_profiled_plus,
                                                                  x0=amplitude_best_fit[jperiod, jsim]-0.1,
                                                                  args=(period, times,
                                                                        angle_fit[:, jsim],
                                                                        angle_error[:, jsim],
                                                                        neg2logL_global_fval,
                                                                        1.0))

    ### PLOTTING ###
    if args.time_domain_fit:
        plt.figure(1)
        plt.plot(periods_to_test, amplitude_best_fit[:,0] * 180 / np.pi,
                 color='C0', label='observed amplitude')
        plt.plot(periods_to_test, np.median(amplitude_best_fit, axis=1) * 180 / np.pi,
                 '--', color='C1', label='median expected amplitude')
        plt.plot(periods_to_test, np.percentile(amplitude_best_fit, 15.9, axis=1) * 180 / np.pi,
                 ':', color='C1', label='+/-1$\sigma$ range of amplitude')
        plt.plot(periods_to_test, np.percentile(amplitude_best_fit, 84.1, axis=1) * 180 / np.pi,
                 ':', color='C1')

        plt.xlabel('oscillation period [# of bundles]')
        plt.ylabel('oscillation amplitude [deg]')
        plt.title('best-fit oscillation amplitude (single realization)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('bestfit_A_vs_period.png', dpi=150)
        
        if args.simulate_limits:
            plt.figure(10)
            plt.plot(periods_to_test, amplitude_up1sigma[:,0] * 180 / np.pi,
                     color='C0', label='observed +1$\sigma$ limit')
            plt.plot(periods_to_test, np.median(amplitude_up1sigma, axis=1) * 180 / np.pi,
                     '--', color='C1', label='median +1$\sigma$ limit')
            plt.plot(periods_to_test, np.percentile(amplitude_up1sigma, 15.9, axis=1) * 180 / np.pi,
                     ':', color='C1', label='+/-1$\sigma$ range of limit')
            plt.plot(periods_to_test, np.percentile(amplitude_up1sigma, 84.1, axis=1) * 180 / np.pi,
                     ':', color='C1')

            plt.xlabel('oscillation period [# of bundles]')
            plt.ylabel('oscillation amplitude [deg]')
            plt.title('best-fit oscillation amplitude (single realization)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('up1sigma_A_vs_period.png', dpi=150)

            plt.figure(11)
            plt.plot(periods_to_test, amplitude_down1sigma[:,0] * 180 / np.pi,
                     color='C0', label='observed -1$\sigma$ limit')
            plt.plot(periods_to_test, np.median(amplitude_down1sigma, axis=1) * 180 / np.pi,
                     '--', color='C1', label='median -1$\sigma$ limit')
            plt.plot(periods_to_test, np.percentile(amplitude_down1sigma, 15.9, axis=1) * 180 / np.pi,
                     ':', color='C1', label='+/-1$\sigma$ range of limit')
            plt.plot(periods_to_test, np.percentile(amplitude_down1sigma, 84.1, axis=1) * 180 / np.pi,
                     ':', color='C1')

            plt.xlabel('oscillation period [# of bundles]')
            plt.ylabel('oscillation amplitude [deg]')
            plt.title('best-fit oscillation amplitude (single realization)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('down1sigma_A_vs_period.png', dpi=150)

    plt.figure(2)
    _ = plt.errorbar(np.arange(len(angle_fit[:,0])), angle_fit[:,0] * 180/np.pi,
                     yerr=angle_error[:,0] * 180/np.pi,
                     linestyle='None', marker='o', markersize=3)
    plt.xlabel('bundle index')
    plt.ylabel('rotation angle [deg]')
    plt.title('simulated global rotation angle\n'
              '{} observations / bundle, {:.1f} uK-arcmin per bundle\n'\
              .format(args.obs_per_bundle, noise_per_bundle))
    plt.tight_layout()
    plt.savefig('angle_vs_bundle.png', dpi=150)


