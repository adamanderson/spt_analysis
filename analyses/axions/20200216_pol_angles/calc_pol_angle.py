import argparse as ap
from spt3g import core, mapmaker, calibration
from spt3g.mapmaker.mapmakerutils import remove_weight
from scipy.optimize import minimize, newton
import numpy as np
import os.path
import pickle

def calc_chi2(pol_rotation, obs_maps, coadd_maps, subfield=None):
    '''
    Calculate the chi2 as a function of global polarization rotation.

    Parameters
    ----------
    pol_rotation : float or array_like
        Global polarization rotation angle to assume in model.
    obs_maps : G3Frame
        Frame containing T, Q, U, and Wpol entries with maps for a single
        observation.
    coadd_maps : G3Frame
        Frame containing T, Q, U, and Wpol coadd maps.
    subfield : array_like or None (default=None)
        Array specifying the range in RA and dec over which to evaluate the
        chi2: [RA_min, RA_max, dec_min, dec_max]
        If `None`, then uses the entire map.

    Returns
    -------
    chi2_map : float
        Chi-square calculated over the entire map, at the model parameters.
    '''
    if type(pol_rotation) is not np.ndarray:
        pol_rotation = np.array([pol_rotation])
        
    chi2_map = np.zeros(len(pol_rotation))

    obs_noweight = core.G3Frame(core.G3FrameType.Map)
    T_noweight, Q_noweight, U_noweight = remove_weight(obs_maps['T'], obs_maps['Q'],
                                                       obs_maps['U'], obs_maps['Wpol'])
    obs_noweight['T'] = T_noweight
    obs_noweight['Q'] = Q_noweight
    obs_noweight['U'] = U_noweight

    coadd_noweight = core.G3Frame(core.G3FrameType.Map)
    T_coadd_noweight, Q_coadd_noweight, U_coadd_noweight = remove_weight(coadd_maps['T'], coadd_maps['Q'],
                                                                         coadd_maps['U'], coadd_maps['Wpol'])
    coadd_noweight['T'] = T_coadd_noweight
    coadd_noweight['Q'] = Q_coadd_noweight
    coadd_noweight['U'] = U_coadd_noweight
        
    for jpol, pol_rot in enumerate(pol_rotation):
        npixels = len(obs_noweight['Q'])
        print(npixels)
        delta_f = 3
        chi2_per_pixel = np.zeros((obs_noweight['Q'].shape[0],
                                   obs_noweight['Q'].shape[1]))
        for ipixel in range(chi2_per_pixel.shape[0]):
            for jpixel in range(chi2_per_pixel.shape[1]):
                if np.isfinite(obs_noweight['T'][ipixel, jpixel]):
                    pix_coords = obs_noweight['T'].pixel_to_angle(ipixel, jpixel)
                    if subfield is None or \
                       (pix_coords[0] > subfield[0] and pix_coords[0] < subfield[1] and \
                        pix_coords[1] > subfield[2] and pix_coords[1] < subfield[3]):
                        weights = obs_maps['Wpol'][ipixel, jpixel]
                        tqu = np.array([obs_noweight['T'][ipixel, jpixel] - coadd_noweight['T'][ipixel, jpixel],
                                        obs_noweight['Q'][ipixel, jpixel] - coadd_noweight['Q'][ipixel, jpixel] + \
                                            pol_rot*coadd_noweight['U'][ipixel, jpixel],
                                        obs_noweight['U'][ipixel, jpixel] - coadd_noweight['U'][ipixel, jpixel] - \
                                            pol_rot*coadd_noweight['Q'][ipixel, jpixel]])
                        chi2_per_pixel[ipixel, jpixel] = np.matmul(np.matmul(tqu.transpose(), weights/(delta_f**2)), tqu)
                        chi2_map[jpol] += chi2_per_pixel[ipixel, jpixel]
    return chi2_map


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Estimates the polarization angle for '
                               'a field observation.',
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument('obsfile', action='store', type=str,
                        help='Name of the observation file with the map to analyze.')
    parser.add_argument('coaddfile', action='store', type=str,
                        help='Name of the file with the map coadds.')
    parser.add_argument('--frequency', action='store', type=str,
                        choices=['90', '150', '220'], default='150',
                        help='Observing frequency of maps to analyze. Each '
                        'frequency is analyzed separately in the current scheme.')
    parser.add_argument('--results-filename', action='store',
                        default='fit_results.pkl',
                        help='Name file in which to save fit results.')
    parser.add_argument('--subfield-coords', action='store', type=float, nargs=4,
                        default=None,
                        help='Coordinates [in deg] of the subfield to which to '
                        'restrict the analysis: [RA_min, RA_max, dec_min, dec_max]')
    args = parser.parse_args()

    # Check that output pickle file won't overwrite existing file.
    if os.path.exists(args.results_filename):
        raise FileExistsError('Pickle file with results already exists.')

    # Convert angular arguments to G3 units system.
    if args.subfield_coords is not None:
        for jcoord in range(len(args.subfield_coords)):
            args.subfield_coords[jcoord] = args.subfield_coords[jcoord] * core.G3Units.deg

    # Read the data with a pipeline.
    class MapExtractor(object):
        def __init__(self, frequency=None):
            self.observation_maps = core.G3Frame(core.G3FrameType.Map)
            self.delta_f_weights = None
            self.map_frequency = frequency
        def __call__(self, frame):
            if frame.type == core.G3FrameType.PipelineInfo and \
               "weight_high_freq" in frame.keys() and \
               "weight_low_freq" in frame.keys():
                self.delta_f_weights = (frame["weight_high_freq"] - frame["weight_low_freq"]) / core.G3Units.sec
            elif frame.type == core.G3FrameType.Map and \
                 (self.map_frequency is None or self.map_frequency in frame['Id']):
                self.observation_maps['T'] = frame['T']
                self.observation_maps['Q'] = frame['Q']
                self.observation_maps['U'] = frame['U']
                self.observation_maps['Wpol'] = frame['Wpol']

    if not os.path.exists(args.obsfile):
        raise FileNotFoundError('Observation file {} does not exist.'.format(args.obsfile))

    # individual observation
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=args.obsfile)
    map_extractor = MapExtractor(frequency=args.frequency)
    pipe.Add(map_extractor)
    pipe.Run()
    obs_maps = map_extractor.observation_maps
    delta_f_weights = map_extractor.delta_f_weights

    # coadded observation
    pipe_coadd = core.G3Pipeline()
    pipe_coadd.Add(core.G3Reader, filename=args.coaddfile)
    map_extractor_coadd = MapExtractor()
    pipe_coadd.Add(map_extractor_coadd)
    pipe_coadd.Run()
    coadd_maps = map_extractor_coadd.observation_maps


    # Do the minimization.
    chi2_min_result = minimize(calc_chi2, 0.01, method='Powell',
                               args=(obs_maps, coadd_maps, args.subfield_coords),
                               options={'ftol':1e-6})
    fit_angle = float(chi2_min_result.x)
    fit_chi2 = float(chi2_min_result.fun)

    # Do a scan to evaluate the chi2 on a grid of points for diagnostic purposes
    # with a width of +/- 10 deg of the minimum value.
    scan_angles = np.linspace(fit_angle - 10*np.pi/180,
                              fit_angle + 10*np.pi/180, 20)
    scan_chi2 = calc_chi2(scan_angles, obs_maps, coadd_maps, args.subfield_coords)

    # The likelihood is extremely parabolic, so we can fit the scan data and work
    # in the parabolic approximation.
    polycoeffs = np.polyfit(scan_angles, scan_chi2, deg=2)

    # Calculate confidence intervals.
    poly_min_result = minimize(lambda x: np.polyval(polycoeffs, x), fit_angle,
                               method='Powell', options={'ftol':1e-6})
    chi2_poly_min = float(poly_min_result.fun)
    angle_poly_min = float(poly_min_result.x)

    def delta_chi2_minus_1(x, p, chi2_0):
        return np.polyval(p, x) - chi2_0 - 1

    # Since we are assuming a parabolic likelihood, the intervals are formally
    # symmetric, but this is cheap so we'll compute it twice anyway.
    err_down1sigma = newton(delta_chi2_minus_1, x0=angle_poly_min - 1*np.pi/180,
                            args=(polycoeffs, chi2_poly_min))
    err_up1sigma   = newton(delta_chi2_minus_1, x0=angle_poly_min + 1*np.pi/180,
                            args=(polycoeffs, chi2_poly_min))

    # Package up results and save.
    results = {'minization_results': chi2_min_result,
               'fit_angle': fit_angle,
               'fit_chi2': fit_chi2,
               'scan_angles': scan_angles,
               'scan_chi2': scan_chi2,
               'quadratic_approx_coeffs': polycoeffs,
               'angle_interval': [err_down1sigma, err_up1sigma]}
    with open(args.results_filename, 'wb') as f:
        pickle.dump(results, f)
