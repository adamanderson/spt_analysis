import argparse
import numpy as np
import pickle
import timefittools

parser = argparse.ArgumentParser()
parser.add_argument('nsims', action='store', type=int,
                    help='Number of realizations of fake data to simulate.')
parser.add_argument('fmin', action='store', type=float,
                    help='Minimum frequency to simulate.')
parser.add_argument('fmax', action='store', type=float,
                    help='Maximum frequency to simulate.')
parser.add_argument('nfreqs', action='store', type=int,
                    help='Number of frequencies to simulate between fmin and '
                    'fmax.')
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
                    default=0.95,
                    help='Confidence level for upper limit.')
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

frequencies = np.logspace(np.log10(args.fmin), np.log10(args.fmax), args.nfreqs)

for jsim in np.arange(args.nsims):
    pol_angles = np.random.normal(loc=0, scale=args.pol_error, size=num_obs)
    pol_angle_errs = np.array([args.pol_error for jobs in np.arange(num_obs)])

    data = {'times': times,
            'angles': pol_angles,
            'errs': pol_angle_errs}
    min_chi2 = timefittools.minimize_chi2(data)
    Acls = {freq: timefittools.upper_limit_bayesian(freq / (60*60*24),
                                                    data, args.upper_limit_cl,
                                                    [1e-3, 1], min_chi2) \
            for freq in frequencies}

    out_dict['results'][jsim] = {'upper limit': Acls,
                                 'data': data}

with open(args.outfile, 'wb') as f:
    pickle.dump(out_dict, f)
