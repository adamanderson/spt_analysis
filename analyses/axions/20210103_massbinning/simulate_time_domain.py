import argparse
import numpy as np
import pickle
import timefittools
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument('nsims', action='store', type=int,
                    help='Number of realizations of fake data to simulate.')
parser.add_argument('fmin', action='store', type=float,
                    help='Minimum frequency to fit for in units of inverse '
                    'days.')
parser.add_argument('fmax', action='store', type=float,
                    help='Maximum frequency to fit for in units of inverse '
                    'days.')
parser.add_argument('nfreqs', action='store', type=int,
                    help='Number of frequencies to fit for between fmin and '
                    'fmax.')
parser.add_argument('signalamp', action='store', type=float,
                    help='Amplitude of signal to inject in time domain '
                    'simulations, in units of degrees polariation rotation.')
parser.add_argument('signalfreq', action='store', type=float,
                    help='Frequency of signal to inject in time domain '
                    'simulations, in units of inverse days.')
parser.add_argument('--signal-phase', action='store', type=float, default=None,
                    help='Phase of signal to inject in time domain '
                    'simulations. Default value of `None` results in uniform '
                    'random sampling of the phase.')
parser.add_argument('--pol-error', action='store', type=float, default=1.5,
                    help='Uncertainty on polarization error per observation, '
                    'in deg.')
parser.add_argument('--duration', action='store', type=float, default=270,
                    help='Duration of data to simulate, in days.')
parser.add_argument('--obs', action='store', default=500,
                    help='Number of observations to simulate per realization '
                    'or the name of a pickle file of obsids to load.')
parser.add_argument('--nseasons', action='store', type=int, default=1,
                    help='Number of years. Takes the observations times in a '
                    'pickle file and concatenates N copies of them, offset by '
                    'one calendar year.')
parser.add_argument('--upper-limit-cl', action='store', type=float,
                    default=None,
                    help='Confidence level for upper limit. If `None`, then '
                    'do not compute the upper limit.')
parser.add_argument('--outfile', action='store', default='sim_results.pkl',
                    help='Name of file in which to store simulation output.')
args = parser.parse_args()

out_dict = {'results':{}}

# interpret arguments `--obs` as either an integer number of observations or
# the name of a pickle file with obsids, based on context
try:
    num_obs = int(args.obs)
    times = np.linspace(0, 60*60*24*args.duration, num_obs)
except:
    with open(args.obs, 'rb') as f:
        times = pickle.load(f)
        for jseason in np.arange(args.nseasons-1):
            times = np.append(times, times + 60*60*24*365)
        num_obs = len(times)

frequencies = np.linspace(args.fmin, args.fmax, args.nfreqs)


for jsim in np.arange(args.nsims):
    print(jsim)
    if args.signal_phase is None:
        signal_phase = 2*np.pi*np.random.rand()
    else:
        signal_phase = args.signal_phase

    pol_angles = args.signalamp * np.sin(2*np.pi*args.signalfreq / (60*60*24) * times + \
                                         signal_phase)
    pol_angles += np.random.normal(loc=0, scale=args.pol_error, size=num_obs)
    pol_angle_errs = np.array([args.pol_error for jobs in np.arange(num_obs)])

    data = {'times': times,
            'angles': pol_angles,
            'errs': pol_angle_errs}
    model = timefittools.TimeDomainModel(data)

    Afits = {}
    Acls = {}
    A_test = np.linspace(0, 3, 20)
    for freq in frequencies:
        def posterior(A):
            if A < 0:
                return 99999
            else:
                return -2*np.log(model.posterior_marginal(A, freq / (60*60*24)))
        result = minimize(posterior, args.signalamp, method='Powell')
        Afits[freq] = result['x']

        if args.upper_limit_cl is not None:
            Acls[freq] = model.upper_limit_bayesian(freq / (60*60*24), args.upper_limit_cl)
        else:
            Acls[freq] = None

    out_dict['results'][jsim] = {'A_upperlimit': Acls,
                                 'min_chi2': model.min_chi2,
                                 'A_fit': Afits,
                                 'data': data}

with open(args.outfile, 'wb') as f:
    pickle.dump(out_dict, f)
