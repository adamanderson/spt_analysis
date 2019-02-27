from spt3g import core, mapmaker, coordinateutils, mapspectra
from spt3g.mapmaker.mapmakerutils import load_spt3g_map
from spt3g.mapspectra.map_analysis import calculateCls
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import pickle
import os.path

parser = ap.ArgumentParser(description='Simulate effect of bolometer gain '
                           'variation as a function of field elevation.')
parser.add_argument('simskies', nargs='*',
                    help='FITS files with simulated skies')
parser.add_argument('--nskies', type=int, default=1,
                    help='Number of skies to simulate. Must be less than the '
                    'number of files specified in `simskies` argument.')
parser.add_argument('--bias-type', choices=['none', 'linear'], default='none',
                    help='Type of bias to inject into the map.')
parser.add_argument('--linear-bias-mag', type=float, default=1.0,
                    help='Bias magnitude in units of % gain variation per '
                    'degree of declination.')
parser.add_argument('--ncalstares', type=int, default=1,
                    help='Number of calibrator stares that are assumed to '
                    'occur in the field observation. The bias is assumed to be '
                    'reset at the elevation of each calibrator stare.')
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
args = parser.parse_args()

Cls = {}

for jsky in range(args.nskies):
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
                if args.bias_type == 'none':
                    Tmap[ypx, xpx] = sim_map['T'][npx]
                    Qmap[ypx, xpx] = sim_map['Q'][npx]
                    Umap[ypx, xpx] = sim_map['U'][npx]
                if args.bias_type == 'linear':
                    pixel_dec = coords[1]/core.G3Units.deg
                    Tmap[ypx, xpx] = sim_map['T'][npx] * (1 + args.linear_bias_mag/100 * (np.min(cal_decs[cal_decs>pixel_dec]) - pixel_dec))
                    Qmap[ypx, xpx] = sim_map['Q'][npx] * (1 + args.linear_bias_mag/100 * (np.min(cal_decs[cal_decs>pixel_dec]) - pixel_dec))
                    Umap[ypx, xpx] = sim_map['U'][npx] * (1 + args.linear_bias_mag/100 * (np.min(cal_decs[cal_decs>pixel_dec]) - pixel_dec))

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
    Cls[args.simskies[jsky]] = calculateCls(map_fr, apod_mask=apod)

    if args.bias_type == 'linear':
        fname_stub = '{}_{}'.format(args.bias_type, args.linear_bias_mag)
    else:
        fname_stub = args.bias_type
    w = core.G3Writer('{}_{}.g3'.format(fname_stub,
                                        os.path.splitext(os.path.basename(args.simskies[jsky]))[0]))
    w.Process(map_fr)
    w.Process(core.G3Frame(core.G3FrameType.EndProcessing))


if args.bias_type == 'linear':
    fname_stub = '{}_{}'.format(args.bias_type, args.linear_bias_mag)
else:
    fname_stub = args.bias_type

with open('sim_cl_bias_{}.pkl'.format(fname_stub), 'wb') as f:
    pickle.dump(Cls, f)

