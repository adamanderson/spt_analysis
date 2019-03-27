from spt3g import core, mapmaker, coordinateutils, mapspectra
from spt3g.mapmaker.mapmakerutils import load_spt3g_map
from spt3g.mapspectra.map_analysis import calculateCls
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import pickle
import os.path
import camb
from scipy.optimize import minimize


def knox_errors(ells, cls, fsky, noise):
    sigma_beam = 1.2 / np.sqrt(8*np.log(2)) * np.pi / 10800.
    dcls = np.sqrt(2. / ((2.*ells + 1) * fsky)) * \
                (cls + (noise * np.pi / 10800)**2. * np.exp(ells**2 * sigma_beam**2))
    return dcls


def neg2LogL(x, cls_data):
    camb_index = {'TT':0, 'EE':1, 'BB':3, 'TE': 2}
    fsky = 1500 / (4*np.pi / ((2*np.pi / 360)**2))
    Tnoise = {150: 2.2} # uK arcmin
    
    H0 = x[0]
    ombh2 = x[1]
    omch2 = x[2]

    if H0 > 80 or H0 < 50 or ombh2 > 0.03 or \
       ombh2 < 0.012 or omch2 > 0.15 or omch2 < 0.07:
        return 1e9

    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings,
    # with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=0.0666)
    camb.set_params(lmax=5000)
    pars.InitPower.set_params(As=2.141e-9, ns=0.9683, r=0)

    #calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=powers['total']

    ells_theory = np.arange(totCL.shape[0])
    ells_theory_binned = np.intersect1d(ells_theory, cls_data['ell'])
    ells_data = cls_data['ell'][np.isin(cls_data['ell'], ells_theory_binned)]
    chi2 = 0
    for spectrum in ['TT', 'EE']:
        dls_theory_binned = totCL[:,camb_index[spectrum]] \
                                 [np.isin(ells_theory, ells_theory_binned)]
        cls_theory_binned = dls_theory_binned / (ells_theory_binned*(ells_theory_binned+1) / (2*np.pi))
        cls_data_binned = cls_data[spectrum][np.isin(cls_data['ell'], ells_theory_binned)]
        residual = cls_theory_binned - cls_data_binned 
        
        if spectrum == 'TT':
            noise = Tnoise[150]
        else:
            noise = Tnoise[150] * np.sqrt(2.)
        cl_cov = knox_errors(ells_theory_binned, cls_theory_binned, fsky, noise)**2
        
        chi2 += np.sum(residual**2. / cl_cov)
    
    return chi2


parser = ap.ArgumentParser(description='Simulate effect of bolometer gain '
                           'variation as a function of field elevation.')
parser.add_argument('simskies', nargs='*',
                    help='FITS files with simulated skies')
parser.add_argument('outfile', type=str,
                    help='Name of output filename with Cls of simulated maps.')
parser.add_argument('--nskies', type=int, default=1,
                    help='Number of skies to simulate. Must be less than the '
                    'number of files specified in `simskies` argument.')
parser.add_argument('--linear-bias-mag', type=float, default=[0.0], nargs='*',
                    help='Bias magnitude in units of percent gain variation per '
                    'degree of declination.')
parser.add_argument('--ncalstares', type=int, default=1,
                    help='Number of calibrator stares that are assumed to '
                    'occur in the field observation. The bias is assumed to be '
                    'reset at the elevation of each calibrator stare.')
parser.add_argument('--fit-cosmology', action='store_true',
                    help='Fit cosmology to the simulated sky.')
parser.add_argument('--res', type=float, default=2.0,
                    help='Resolution of simulated in map in arcmin. Note that '
                    'there is no check to ensure that the resolution of the '
                    'constructed map is smaller than the resolution of the '
                    'source simulated skies.')
parser.add_argument('--ra-center', type=float, default=0,
                    help='Center of map RA in deg.')
parser.add_argument('--dec-center', type=float, default=-60,
                    help='Center of map dec in deg.')
parser.add_argument('--ra-pixels', type=int, default=4000,
                    help='Width of the map in number of RA pixels.')
parser.add_argument('--dec-pixels', type=int, default=1500,
                    help='Width of the map in number of dec pixels.')
parser.add_argument('--norm-to-unbiased', action='store_true',
                    help='Normalize spectra TT, EE, and BB separately to '
                    'the power spectrum with zero bias. Note that this '
                    'requires that a simulation with no bias is run '
                    'first.')
parser.add_argument('--save-maps', action='store_true',
                    help='Save the biased and unbiased maps that are generated '
                    'by the script.')
args = parser.parse_args()

# check arguments
if args.norm_to_unbiased and args.linear_bias_mag[0] != 0.0:
    raise ValueError('Argument `--norm-to-unbiased` is set, but the first '
                     'simulation is not a map with zero bias. Set first '
                     'argument to `--linear-bias-mag` to 0.0.')


for jsky in range(args.nskies):
    Cls = {}
    for bias_mag in args.linear_bias_mag:
        print(args.norm_to_unbiased)
        print(bias_mag)
        Cls[bias_mag] = {}
        print('Simulating sky #{}'.format(jsky))

        # load the simulated map
        sim_map = load_spt3g_map(args.simskies[jsky])

        # make the flat sky map
        print('Making flat sky map...')
        Tmap = coordinateutils.FlatSkyMap(args.ra_pixels, args.dec_pixels,
                                          args.res*core.G3Units.arcmin,
                                          proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                          alpha_center=args.ra_center*core.G3Units.deg,
                                          delta_center=args.dec_center*core.G3Units.deg)
        Qmap = coordinateutils.FlatSkyMap(args.ra_pixels, args.dec_pixels,
                                          args.res*core.G3Units.arcmin,
                                          proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                          alpha_center=args.ra_center*core.G3Units.deg,
                                          delta_center=args.dec_center*core.G3Units.deg)
        Umap = coordinateutils.FlatSkyMap(args.ra_pixels, args.dec_pixels,
                                          args.res*core.G3Units.arcmin,
                                          proj=coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea,
                                          alpha_center=args.ra_center*core.G3Units.deg,
                                          delta_center=args.dec_center*core.G3Units.deg)

        max_dec = -33.5
        min_dec = -78.5
        cal_decs = np.linspace(max_dec, min_dec, args.ncalstares+1)[:-1]
        for xpx in range(args.ra_pixels):
            for ypx in range(args.dec_pixels):
                coords = Tmap.pixel_to_angle(xpx, ypx)
                npx = sim_map['T'].angle_to_pixel(coords[0], coords[1])
                if npx < sim_map['T'].shape[0]:
                    pixel_dec = coords[1]/core.G3Units.deg
                    Tmap[ypx, xpx] = sim_map['T'][npx] * \
                                     (1 + bias_mag/100 * \
                                      (np.min(cal_decs[cal_decs>pixel_dec]) - pixel_dec))
                    Qmap[ypx, xpx] = sim_map['Q'][npx] * \
                                     (1 + bias_mag/100 * \
                                      (np.min(cal_decs[cal_decs>pixel_dec]) - pixel_dec))
                    Umap[ypx, xpx] = sim_map['U'][npx] * \
                                     (1 + bias_mag/100 * \
                                      (np.min(cal_decs[cal_decs>pixel_dec]) - pixel_dec))

        # make the weights map
        weights = core.G3SkyMapWeights(Tmap, weight_type=core.WeightType.Wunpol)
        for xpx in range(weights.shape[0]):
            for ypx in range(weights.shape[1]):
                if Tmap[xpx, ypx] != 0:
                    weights[xpx, ypx] = np.eye(3)

        # build the map frame
        map_fr = core.G3Frame(core.G3FrameType.Map)
        map_fr['T'] = Tmap
        map_fr['Q'] = Qmap
        map_fr['U'] = Umap
        map_fr['Wpol'] = weights

        # make the apodization mask
        apod = mapspectra.apodmask.makeBorderApodization(
            map_fr['Wpol'], apod_type='cos',
            radius_arcmin=15.,zero_border_arcmin=10,
            smooth_weights_arcmin=5)

        # calculate the Cls
        print('Calculating Cls...')
        Cls[bias_mag]['cls'] = calculateCls(map_fr, apod_mask=apod)
        if args.norm_to_unbiased and bias_mag != 0.0:
            Cls[bias_mag]['cal'] = dict()
            Cls[bias_mag]['cls_normalized'] = dict()
            Cls[bias_mag]['cls_normalized']['ell'] = Cls[bias_mag]['cls']['ell']
            for spectrum in ['TT', 'EE', 'BB']:
                ell_range_nobias = (Cls[0.0]['cls']['ell'] > 500) & \
                                   (Cls[0.0]['cls']['ell'] < 2000)
                ell_range_bias   = (Cls[bias_mag]['cls']['ell'] > 500) & \
                                   (Cls[bias_mag]['cls']['ell'] < 2000)
                cal_factor = np.mean(Cls[0.0]['cls'][spectrum][ell_range_nobias]) / \
                             np.mean(Cls[bias_mag]['cls'][spectrum][ell_range_bias])
                Cls[bias_mag]['cal'][spectrum] = cal_factor
                Cls[bias_mag]['cls_normalized'][spectrum] = Cls[bias_mag]['cls'][spectrum] * cal_factor
                                                            

        # save the maps to a G3 file
        if args.save_maps:
            fname_stub = 'linearbias_{:.1f}percentPerDeg'.format(bias_mag)
            w = core.G3Writer('{}_{}.g3'.format(fname_stub,
                                                os.path.splitext(os.path.basename(args.simskies[jsky]))[0]))
            w.Process(map_fr)
            w.Process(core.G3Frame(core.G3FrameType.EndProcessing))

        if args.fit_cosmology:
            if args.norm_to_unbiased and bias_mag != 0:
                res = minimize(neg2LogL, [67.87, 0.022277, 0.11843],
                               args=(Cls[bias_mag]['cls_normalized']),
                               method='powell',
                               options={'xtol': 1e-6, 'disp': True})
            else:
                res = minimize(neg2LogL, [67.87, 0.022277, 0.11843],
                               args=(Cls[bias_mag]['cls']),
                               method='powell',
                               options={'xtol': 1e-6, 'disp': True})
            Cls[bias_mag]['fit'] = res

        # write output data file after every bias simulation
        # with open('sim_cl_{}.pkl'.format(os.path.splitext(os.path.basename(args.simskies[jsky]))[0]), 'wb') as f:
        #     pickle.dump(Cls, f)
        with open(args.outfile, 'wb') as f:
            pickle.dump(Cls, f)

