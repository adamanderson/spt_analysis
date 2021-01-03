import argparse
import pickle
from ksztools import *
from pathlib import Path
import os
import matplotlib.pyplot as plt
from glob import glob

# construct interpolated noise curves
s4_ilc_noise = np.loadtxt('ilc_residuals_s4.csv', delimiter=',')
spt4_ilc_noise = np.load('cmb_ilc_residuals_90-150-220-225-286-345_reducedradiopower4.0.npy').item()

s4_ilc_interp = interp1d(s4_ilc_noise[:,0],
                         s4_ilc_noise[:,1])
ell_plot = np.linspace(1,10000,10000)
spt4_ilc_interp = interp1d(ell_plot,
                           ell_plot*(ell_plot+1)*spt4_ilc_noise['cl_ilc_residual'] / (2*np.pi))

ell_interp = np.linspace(500, 7000)
noise_coeffs_s4 = np.polyfit(ell_interp, s4_ilc_interp(ell_interp), deg=5)
noise_coeffs_spt4 = np.polyfit(ell_interp, spt4_ilc_interp(ell_interp), deg=5)

noise_curve_s4 = np.poly1d(noise_coeffs_s4)
noise_curve_spt4 = np.poly1d(noise_coeffs_spt4)

# construct interpolated noise curves
basepath = Path().absolute()
expt_names = ['s4deepv3r025', 's4deepv3r025plusspt4HF', 's4wide',
              'sobaseline', 'spt3g', 'spt4_C3']
noise_curves = {}
noise_curves_poly = {}
for expt in expt_names:
      fname = glob(os.path.join(basepath, 'ilc_residuals', expt, '*.npz'))[0]
      noise_dict = np.load(fname, allow_pickle = 1, encoding = 'latin1')['arr_0'].item()
      noise_curves[expt] = noise_dict['all_cl_ilc_residual']['TT']
      noise_ell = np.arange(len(noise_curves[expt]))
      Cl2Dl_factor = noise_ell[noise_ell>100]*(noise_ell[noise_ell>100]+1)/(2*np.pi)
      noise_coeffs = np.polyfit(noise_ell[noise_ell>100], Cl2Dl_factor * noise_curves[expt][noise_ell>100], deg=5)
      noise_curves_poly[expt] = np.poly1d(noise_coeffs)

      plt.semilogy(noise_ell, noise_ell*(noise_ell+1)/(2*np.pi) * noise_curves[expt], label=expt)
      plt.semilogy(noise_ell, noise_curves_poly[expt](noise_ell), 'k--')
plt.xlim([100,8000])
plt.legend()
plt.tight_layout()
plt.savefig('noise_curves.png', dpi=200)


# experimental parameters
expt_params = {'SPT4-simple':    {'beam_fwhm': 1.2,
                                  'noise': 3,
                                  'fsky': 0.036,
                                  'noise_curve': noise_curve_spt4,
                                  'use_DlkSZ_reion_derivs': True},
               'S4-wide-simple': {'beam_fwhm': 1.4,
                                  'noise': 4,
                                  'fsky': 0.4,
                                  'noise_curve': noise_curve_s4,
                                  'use_DlkSZ_reion_derivs': True},
               'default':        {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.7,
                                  'noise_curve': None,
                                  'use_DlkSZ_reion_derivs': True},
               's4deepv3r025':        {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.04,
                                  'noise_curve': noise_curves_poly['s4deepv3r025'],
                                  'use_DlkSZ_reion_derivs': True},
               's4deepv3r025plusspt4HF':        {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.03,
                                  'noise_curve': noise_curves_poly['s4deepv3r025plusspt4HF'],
                                  'use_DlkSZ_reion_derivs': True},
               's4wide':        {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.4,
                                  'noise_curve': noise_curves_poly['s4wide'],
                                  'use_DlkSZ_reion_derivs': True},
               'sobaseline':        {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.4,
                                  'noise_curve': noise_curves_poly['sobaseline'],
                                  'use_DlkSZ_reion_derivs': True},
               'spt3g':     {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.036,
                                  'noise_curve': noise_curves_poly['spt3g'],
                                  'use_DlkSZ_reion_derivs': True},
               'spt4_C3':        {'beam_fwhm': 1.0,
                                  'noise': 1.0,
                                  'fsky': 0.036,
                                  'noise_curve': noise_curves_poly['spt4_C3'],
                                  'use_DlkSZ_reion_derivs': True},
              }

parser = argparse.ArgumentParser()
parser.add_argument('outfile', action='store', type=str,
                    help='Pattern for filename to use for fisher matrix output.')
parser.add_argument('experiment', choices=list(expt_params.keys()),  type=str,
                    help='Experimental configuration to simulate.')
parser.add_argument('nbins', action='store', type=int,
                    help='Number of bins to use for 4pt kSZ spectra.')
args = parser.parse_args()


# fiducial forecast values
Cshot_matrix = np.zeros((args.nbins, args.nbins))

x0 = {'tau': 0.060, 'delta_z_re': 1.2, 'A_late': 1, 'alpha_late': 0,
      'shot_noise': Cshot_matrix}
dx = {'tau': 0.0002, 'delta_z_re': 0.05, 'A_late': 0.05, 'alpha_late': 0.05,
      'shot_noise': np.ones((args.nbins, args.nbins))}
prior = {'tau': None, 'delta_z_re': None, 'A_late': None, 'alpha_late': None,
         'shot_noise': None}
ell_bin_edges = np.linspace(2000, 7000, args.nbins+1)

# General forecast parameters
model_4pt_Nbin = kSZModel(tau=x0['tau'],
                          delta_z_re=x0['delta_z_re'],
                          A_late=x0['A_late'],
                          alpha_late=x0['alpha_late'],
                          beam_fwhm_arcmin=expt_params[args.experiment]['beam_fwhm'],
                          noise_uKarcmin=expt_params[args.experiment]['noise'],
                          ell_bin_edges=ell_bin_edges,
                          Cshot=Cshot_matrix,
                          noise_curve_interp=expt_params[args.experiment]['noise_curve'],
                          use_DlkSZ_reion_derivs=expt_params[args.experiment]['use_DlkSZ_reion_derivs'])
fisher_4pt = Fisher4Point(model_4pt_Nbin, fsky=expt_params[args.experiment]['fsky'])
nlkk, clkk, deriv_clkk, fisher_matrix = fisher_4pt.calc_fisher(x0, dx, prior)

result = {'NLKK': nlkk,
          'CLKK': clkk,
          'deriv_CLKK': deriv_clkk,
          'fisher_matrix': fisher_matrix}

out_fname = '{}_{}_nbins={:d}.pkl'.format(args.outfile, args.experiment, args.nbins)
with open(out_fname, 'wb') as f:
    pickle.dump(result, f)
