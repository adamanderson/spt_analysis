import numpy as np
import matplotlib.pyplot as plt
import camb
import argparse as ap
from scipy.optimize import minimize, newton
from scipy.stats import chi2, sigmaclip

P0 = ap.ArgumentParser(description='Script for estimating the sensitivity of '
                       'the axion rotation analysis using the EB estimator.',
                       formatter_class=ap.ArgumentDefaultsHelpFormatter)
S0 = P0.add_subparsers(dest='mode', metavar='MODE', title='subcommands',
                       help='Function to perform. For help, call: '
                       '%(prog)s %(metavar)s -h')
SP0 = S0.add_parser('singlemap', help='Simulate a single map and estimation of '
                   'its polarization angle.',
                   formatter_class=ap.ArgumentDefaultsHelpFormatter)
SP0.add_argument('--noise-per-map', action='store', type=float, default=150,
                 help='Noise level per map in units of [uK-arcmin].')
SP0.add_argument('--n-sims', action='store', type=int, default=2000,
                 help='Number of simulated Cls to generate.')
SP0.add_argument('--field-size', action='store', type=int, default=500,
                 help='Size of field in square degrees.')
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
                beam_fwhm):
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

def delta_chi2_plus1(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB, func_min):
    return pol_chi2(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB) - func_min - 1


nsim_ul = 2000


if args.mode == 'singlemap':
    delta_chi2_sims = np.zeros(nsim_ul)
    angle_fit = np.zeros(nsim_ul)
    angle_up1sigma = np.zeros(nsim_ul)
    angle_down1sigma = np.zeros(nsim_ul)

    Cl_random = {}
    for spectrum in ['TB', 'TE', 'EB', 'EE', 'BB']:
        Cl_random[spectrum] = np.zeros(nsim_ul)
    Dl_sims = np.zeros(nsim_ul)


    # generate the simulated spectra
    for jsim in range(args.n_sims):
        for spectrum in Cl_random:
            Cl_random[spectrum][jsim] = generate_Cl_averaged(spectrum, ells,
                                                             Cl_theory,
                                                             f_sky=args.field_size/41000,
                                                             lmin=100, lmax=3000,
                                                             map_noise_T=args.noise_per_map,
                                                             beam_fwhm=1.2)

    # minimize the chi-square and estimate the polarizationg angle and uncertainty
    for jsim in range(args.n_sims):
        out = minimize(pol_chi2, 0, args=(Cl_random['TB'][jsim], Cl_random['TE'][jsim],
                                          Cl_random['EB'][jsim], Cl_random['EE'][jsim],
                                          Cl_random['BB'][jsim],
                                          np.var(Cl_random['TB']), np.var(Cl_random['EB'])),
                       method='Powell')
        angle_fit[jsim] = out.x
        Dl_sims[jsim] = out.fun

        try:
            delta_chi2_sims[jsim]  = pol_chi2(0, Cl_random['TB'][jsim], Cl_random['TE'][jsim],
                                              Cl_random['EB'][jsim], Cl_random['EE'][jsim],
                                              Cl_random['BB'][jsim],
                                              np.var(Cl_random['TB']), np.var(Cl_random['EB'])) - Dl_sims[jsim]
            angle_up1sigma[jsim]   = newton(delta_chi2_plus1, x0=angle_fit[jsim] + 0.01,
                                            args=(Cl_random['TB'][jsim], Cl_random['TE'][jsim],
                                                  Cl_random['EB'][jsim], Cl_random['EE'][jsim], Cl_random['BB'][jsim],
                                                  np.var(Cl_random['TB']), np.var(Cl_random['EB']), Dl_sims[jsim]))
            angle_down1sigma[jsim] = newton(delta_chi2_plus1, x0=angle_fit[jsim] - 0.01,
                                            args=(Cl_random['TB'][jsim], Cl_random['TE'][jsim],
                                                  Cl_random['EB'][jsim], Cl_random['EE'][jsim], Cl_random['BB'][jsim],
                                                  np.var(Cl_random['TB']), np.var(Cl_random['EB']), Dl_sims[jsim]))
        except RuntimeError:
            print('Failed to estimate error bar. Skipping simulation.')
    plt.figure(1)
    hist_range = np.std(angle_up1sigma) * 4 * 180/np.pi
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
