import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from spt3g import core, calibration, mapmaker
from spt3g.mapmaker.mapmakerutils import remove_weight
from spt3g.mapspectra.map_analysis import apply_weight
from glob import glob
import pickle

parser = ap.ArgumentParser(description='Estimates the pixel-pixel covariance '
                           'for a 1 deg^2 cutout of a single observation.',
                           formatter_class=ap.ArgumentDefaultsHelpFormatter)
parser.add_argument('coaddfile', action='store', type=str,
                    help='Name of the file with the map coadds.')
parser.add_argument('obsfile', action='store', type=str,
                    help='Name of the observation file with the map to analyze.')
parser.add_argument('--results-filename', action='store',
                    default='covariance.pkl',
                    help='Name file in which to save fit results.')
args = parser.parse_args()

coadddata = list(core.G3File(args.coaddfile))[0]
coadd_noweight = core.G3Frame(core.G3FrameType.Map)
coadd_noweight['T'], coadd_noweight['Q'], coadd_noweight['U'] = remove_weight(coadddata['T'], coadddata['Q'],
                                                                              coadddata['U'], coadddata['Wpol'])
coadd_noweight['Wpol'] = coadddata['Wpol']

mapdata = list(core.G3File(args.obsfile))[-1]

map_noweight = core.G3Frame(core.G3FrameType.Map)
map_noweight['T'], map_noweight['Q'], map_noweight['U']       = remove_weight(mapdata['T'], mapdata['Q'],
                                                                              mapdata['U'], mapdata['Wpol'])
map_noweight['Wpol'] = mapdata['Wpol']

cutout_dim = 1*core.G3Units.deg
cutout_size = int(cutout_dim / map_noweight['Q'].res)
if cutout_size % 2:
    cutout_size += 1

# unweighted coadd from unweighted observation, then add weights back in
map_subtracted = core.G3Frame(core.G3FrameType.Map)
for stokes in ['T', 'Q', 'U']:
    map_subtracted[stokes] = map_noweight[stokes] - coadd_noweight[stokes]
map_subtracted['Wpol'] = map_noweight['Wpol']
#map_subtracted_weight = apply_weight(map_subtracted)

cutout_dict = {}
for stokes in ['T', 'Q', 'U']:
    stokes_arr = np.array(map_subtracted[stokes])

    cutout_running = np.zeros((cutout_size, cutout_size))

    for i in range(stokes_arr.shape[0] - cutout_size):
        for j in range(stokes_arr.shape[1] - cutout_size):
            cutout = stokes_arr[i:(i+cutout_size), j:(j+cutout_size)]
            cutout_center = cutout[int((cutout_size-1) / 2),
                                   int((cutout_size-1) / 2)]
            if np.all(np.isfinite(cutout)):
                cutout_running += cutout * cutout_center

    cutout_dict[stokes] = cutout_running 
cutout_dict['filename'] = args.obsfile

with open(args.results_filename, 'wb') as f:
    pickle.dump(cutout_dict, f)
