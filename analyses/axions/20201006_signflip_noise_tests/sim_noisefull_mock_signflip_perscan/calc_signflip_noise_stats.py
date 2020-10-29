from spt3g import core, maps
from spt3g.util.framecombiner import MapFrameCombiner
import argparse
import os.path
import numpy as np
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import axion_utils
import pickle
import gc

parser = argparse.ArgumentParser()
parser.add_argument('fname', type=str,
                    help='Name of file containing per-scan maps.')
parser.add_argument('fnamecoadd', type=str,
                    help='Name file containing coadd template to use for pol '
                         'angle calculation.')
parser.add_argument('mapstoadd', nargs='*', default=[],
                    help='List of files containing maps to add to the sign-'
                    'flipped coadd. This could be a mock-observed noise-free '
                    'CMB sky, for example.')
parser.add_argument('--nmaps', type=int, default=100,
                    help='Number of sign-flip noise realizations to generate.')
parser.add_argument('--mapid', default=None,
                    help='Id of maps to coadd (wildcards allowed; uses Python '
                         'fnmatch.fnmatch)')
parser.add_argument('-o', '--output', default='output.pkl',
                    help='Name of output file.')
parser.add_argument('--save-maps', action='store_true',
                    help='Save each sign-flip coadded noise realization.')
parser.add_argument('--plot-maps', action='store_true',
                    help='Plot each sign-flip coadded noise realization.')
parser.add_argument('--verbose', action='store_true',
                    help='Add core.Dumps')
parser.add_argument('--seed', type=int, default=np.random.randint(100000),
                    help='Seed for random sign generation. Defaults to a '
                    'random integer.')
parser.add_argument('--keep-mock-cmb', action='store_true',
                    help='Sums maps up in such a way that the resulting map '
                    'consists of a mock observed CMB + the residual from '
                    'differencing L-R pairs. This is useful only for mock '
                    'observations in which we are trying to characterize the '
                    'bias due to imperfect L-R differencing.')
args = parser.parse_args()


# initialize random number generator
rng = np.random.RandomState(args.seed)

pol_angles = []


# extract coadded template for pol angle calculation
coadd_template_frame = None
for frame in core.G3File(args.fnamecoadd):
    if 'Id' in frame and fnmatch(frame['Id'], args.mapid):
        if coadd_template_frame is not None:
            raise KeyError('Multiple frames in coadd template file match map '
                           'ID pattern!')
        else:
            coadd_template_frame = frame
if coadd_template_frame is None:
    raise ValueError('No frame found in coadd template file that matches map '
                     'ID pattern!')

# extract maps to add to signflip coadd
map_frames_to_add = []
for fname in args.mapstoadd:
    map_found = False
    for frame in core.G3File(fname):
        if 'Id' in frame and fnmatch(frame['Id'], args.mapid):
            # if map_found is True:
            #     raise KeyError('Multiple frames in coadd template file match map '
            #                    'ID pattern!')
            # else:
            map_frames_to_add.append(frame)
                # map_found = True
    # if map_found is False:
    #     raise ValueError('No frame found in coadd template file that matches map '
    #                      'ID pattern!')

# pipeline module for calculating polarization angles
class CalcPolAngles(object):
    def __init__(self, coadd_frame, map_id):
        self.coadd_template_frame = coadd_frame
        self.pol_angles = {}
        self.map_id = map_id

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map and 'Id' in frame and \
            fnmatch(frame['Id'], 'combined_*') and \
            fnmatch(frame['Id'], self.map_id):
                self.pol_angles[frame['Id']] = axion_utils.calculate_rho(frame, self.coadd_template_frame, freq_factor=1)


# simple module for flipping signs of maps
class FlipSign(object):
    def __init__(self, rng, offset):
        self.rng = rng
        self.scan_num = 0
        self.signs = []
        self.offset = offset
        
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map and 'Id' in frame and \
           fnmatch(frame['Id'], args.mapid) and 'ScanNumber' in frame:
            if frame['ScanNumber'] % 2 == 0:
                self.signs.append(rng.choice([-1., 1.]))
            else:
                self.signs.append(-1*self.signs[-1])

            flipped_frame = core.G3Frame(core.G3FrameType.Map)
            for key in frame:
                if key in ['T', 'Q', 'U']:
                    flipped_frame[key] = float(self.signs[-1] + self.offset) * frame[key]
                else:
                    flipped_frame[key] = frame[key]
            del frame
            return flipped_frame

        if frame.type == core.G3FrameType.EndProcessing:
            output_frame = core.G3Frame()
            output_frame['signs'] = core.G3VectorDouble(self.signs)
            return [output_frame, frame]
           

class AddMockObsSky(object):
    '''
    Module to add mock-observed sky components to the sign-flip noise, while
    correctly handling weights.
    '''
    def __init__(self, frames, map_id):
        '''
        Constructor
        
        Parameters
        ----------
        frames : list of G3Frame
            Frames that each contain a set of single-observation, mock-observed
            maps in T, Q, U, which should be added in to the noise generated
            from sign-flip per-scan maps.
        map_id : str
            Add the supplied map frames to combined map frames that match this 
            string.
        '''
        self.map_frames = frames
        self.map_id = map_id

        # remove the weights of each map that we want to add in
        for frame in self.map_frames:
            if frame.type == core.G3FrameType.Map and 'Id' in frame and \
               fnmatch(frame['Id'], self.map_id) and \
               len(self.map_frames) > 0:
                # first temporarily remove the weights from the coadded map
                maps.RemoveWeights(frame)

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map and 'Id' in frame and \
           fnmatch(frame['Id'], 'combined_*') and \
           fnmatch(frame['Id'], self.map_id) and \
           len(self.map_frames) > 0:
            # first temporarily remove the weights from the coadded map
            maps.RemoveWeights(frame)
            
            for map_frame in self.map_frames:
                # then add the map in 
                for pol in ['T', 'Q', 'U']:
                    temp_map = frame.pop(pol)
                    temp_map = temp_map + map_frame[pol]
                    frame[pol] = temp_map

            # reapply the map weights to the frame with the additional components added in
            maps.ApplyWeights(frame)


# loop over noise iterations
for jmap in np.arange(args.nmaps):
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=args.fname)
    
    if args.keep_mock_cmb:
        pipe.Add(FlipSign, rng=rng, offset=1.)
    else:
        pipe.Add(FlipSign, rng=rng, offset=0.)

    # coadd maps with signflip
    pipe.Add(MapFrameCombiner, fr_id=args.mapid)

    # add signal map
    pipe.Add(AddMockObsSky, frames=map_frames_to_add, map_id=args.mapid)

    # discard maps that are not the coadded map
    pipe.Add(lambda frame: 'Id' not in frame or \
             fnmatch(frame['Id'], 'combined_*'))

    if args.plot_maps:
        # plot the maps
        def plot_map(frame):
            if frame.type == core.G3FrameType.Map and 'Id' in frame and \
               fnmatch(frame['Id'], 'combined_*'):
                maps.RemoveWeights(frame)

                for pol in ['T', 'Q', 'U']:
                    f = plt.figure(figsize=(12,8))
                    if pol == 'T':
                        vmag = 50
                    else:
                        vmag = 10
                    plt.imshow(frame[pol] / (core.G3Units.microkelvin),
                               vmin=-1*vmag, vmax=vmag)
                    plt.colorbar()
                    plt.title(pol)
                    plt.tight_layout()
                    plt.savefig('{}_map_{}_{}.png'.format(os.path.basename(args.output).split('.')[0], jmap, pol),
                                dpi=200)
                    plt.close(f)
                maps.ApplyWeights(frame)

        pipe.Add(plot_map)

    if args.save_maps:
        out_fname = '{}_{}.g3.gz'.format(os.path.basename(args.output).split('.')[0], jmap)
        pipe.Add(core.G3Writer, filename=out_fname)

    # calculate statistics
    pol_angle_calculator = CalcPolAngles(coadd_frame=coadd_template_frame, 
                                         map_id=args.mapid)
    pipe.Add(pol_angle_calculator)
    # pipe.Add(CalcPolAngles, coadd_frame=coadd_template_frame, map_id=args.mapid)

    if args.verbose:
        pipe.Add(core.Dump)

    pipe.Run(profile=True)
    gc.collect()

    pol_angles.append(pol_angle_calculator.pol_angles)
    del pol_angle_calculator

with open(args.output, 'wb') as f:
    pickle.dump(pol_angles, f)
