from camb import model, initialpower
import argparse as ap
from scipy.optimize import minimize
import pickle

def neg2LogL(x, cls_data):
    H0 = x[0]
    ombh2 = x[1]
    omch2 = x[2]
    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings,
    # with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=0.0666)
    camb.set_params(lmax=5000)
    pars.InitPower.set_params(As=2.141e-9, ns=0.9683, r=0)

    # calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=powers['total']

    ells_theory = np.arange(totCL.shape[0])
    ells_theory_binned = np.intersect1d(ells_theory, cls_data['ell'])
    ells_data_tt = cls_data['ell'][np.isin(cls_data['ell'], ells_theory_binned)]
    dls_theory_tt_binned = totCL[:,0][np.isin(ells_theory, ells_theory_binned)]
    cls_data_tt_binned = cls_data['TT'][np.isin(cls_data['ell'], ells_theory_binned)]
    residual = (dls_theory_tt_binned - \
                cls_data_tt_binned * ells_theory_binned * \
                (ells_theory_binned+1) / (2*np.pi))
    chi2 = np.sum(residual**2)

    return chi2


# parser the arguments
parser = ap.ArgumentParser(description='Fit CLs from a simulated map to '
                           'cosmology with a simple likelihood using CAMB.')
parser.add_argument('clfile', type='string',
                    help='Name of the file that contains the Cls to fit.')
parser.add_argument('outfile', type='string',
                    help='Name of the output file where fit parameters are '
                    'written.')
args = parser.parse_args()


# load the Cls
with open(args.clfile, 'r') as f:
    simdata = pickle.load(f)

# minimize the -2 log-likelihood
res = minimize(neg2LogL, [67.87, 0.022277, 0.11843], args=(cls), method='nelder-mead',
               options={'xtol': 1e-6, 'disp': True})
simdata['fit_result'] = res

with open(args.outfile, 'wb') as f:
    pickle.dump(simdata, f)
