import numpy as np
from spt3g import core, calibration, dfmux
from scipy.optimize import curve_fit
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('asdintrans',
                    help='Name of file containing ASDs to fit for in-transition '
                    'data.')
parser.add_argument('asdhorizon',
                    help='Name of file containing ASDs to fit for horizon data.')
parser.add_argument('outfile',
                    help='Name of output file to save.')
args = parser.parse_args()

normal_data = list(core.G3File(args.asdintrans))
horizon_data = list(core.G3File(args.asdhorizon))

f_min = 0.01
f_max = 60

# noise model
def noise_model_full(x, readout, A, alpha, photon, phonon, tau):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha)) + (photon**2 + phonon**2) / (1 + 2*np.pi*(x*tau)))
def full_readout_model(x, readout, A, alpha):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha)))

# fits to horizon data
freq = np.array(horizon_data[1]["ASD"]['frequency']) / core.G3Units.Hz
fit_params_horizon = {}

jbolo = 0
for bolo in horizon_data[1]["ASD"].keys():
    asd = np.array(horizon_data[1]["ASD"][bolo])
    if np.all(np.isfinite(asd)):
        try:
            par_normal, cov = curve_fit(full_readout_model,
                                        freq[(freq>f_min) & (freq<f_max)],
                                        asd[(freq>f_min) & (freq<f_max)],
                                        bounds=([0, 0, 0],
                                                [np.inf, np.inf, np.inf]),
                                        p0=[10, 1, 1])
            fit_params_horizon[bolo] = par_normal
        except RuntimeError as err:
            print('RuntimeError: {}'.format(err))

    jbolo += 1
    if jbolo % 100 == 0:
        print(jbolo)


# fits to in-transition data
freq = np.array(normal_data[1]["ASD"]['frequency']) / core.G3Units.Hz
fit_params_normal_nocal = {}

jbolo = 0
for bolo in normal_data[1]["ASD"].keys():
    if bolo in fit_params_horizon and \
       bolo in normal_data[1]['VBiasRMS'].keys() and \
       bolo in horizon_data[1]['VBiasRMS'].keys():
        asd = np.array(normal_data[1]["ASD"][bolo])
        if np.all(np.isfinite(asd)) and \
           horizon_data[1]['VBiasRMS'][bolo] != 0:
            vbias_ratio = normal_data[1]['VBiasRMS'][bolo] / \
                          horizon_data[1]['VBiasRMS'][bolo]

            # fit model with *no* power recalibration
            def noise_model_fixed(x, A, alpha, photon, tau):
                return noise_model_full(x, fit_params_horizon[bolo][0] * vbias_ratio,
                                        A, alpha, photon, 0, tau)

            try:
                par_normal, cov = curve_fit(noise_model_fixed,
                                            freq[(freq>f_min) & (freq<f_max)],
                                            asd[(freq>f_min) & (freq<f_max)],
                                            bounds=([0, 0, 0, 0],
                                            [np.inf, np.inf, np.inf, np.inf]),
                                            p0=[10, 1, 10, 0.01])
                fit_params_normal_nocal[bolo] = par_normal
            except RuntimeError as err:
                print('RuntimeError: {}'.format(err))

    jbolo += 1
    if jbolo % 100 == 0:
        print(jbolo)


# save data
outdata = {}
outdata['horizon'] = fit_params_horizon
outdata['in-transition'] = fit_params_normal_nocal
with open(args.outfile, 'wb') as f:
    pickle.dump(outdata, f)
