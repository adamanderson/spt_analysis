"""
Various utilities for coadding maps
"""
import os
from glob import glob
import argparse
from spt3g import core, util
from fnmatch import fnmatch
from spt3g import maps
import numpy as np
import pickle

class FlipSignsForCoadd(object):
    def __init__(self, seed):
        self.seed = seed
        self.signs = []
        self.last_scan_number = -1
        self.rng = np.random.RandomState(self.seed)

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map:
            if 'ScanNumber' in frame:
                if frame['ScanNumber'] == self.last_scan_number:
                    return []
                else:
                    self.last_scan_number = frame['ScanNumber']
            
            if 'ScanNumber' not in frame or \
                frame['ScanNumber'] == 0:
                self.signs.append(self.rng.choice([-1, 1]))
                print(self.signs[-1])

            if self.signs[-1] == -1.:
                negative_frame = core.G3Frame(core.G3FrameType.Map)
                negative_frame['Id'] = frame['Id']
                negative_frame['Wpol'] = frame['Wpol']

                for stokes in ['T', 'Q', 'U']:
                    negative_frame[stokes] = -1 * frame[stokes]

                return negative_frame
            else:
                return frame

class CutObsids(object):
    def __init__(self, obslist):
        '''
        Skip over maps from observation IDs that are in a list of obsids to
        cut.

        Parameters
        ----------
        obslist : list
            List of obsids to cut.
        '''
        self.obslist = obslist
        self.obsid = 0

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Observation and \
           "ObservationID" in frame:
            self.obsid = frame["ObservationID"]
        
        if frame.type == core.G3FrameType.Map and \
           self.obsid in self.obslist:
            return None 
            

def coadd_maps(maps_in, map_id=None, output_file=None, obslist_to_cut=None):
    """
    Coadd `maps_in` into a single map.
    
    Parameters
    ----------
    maps_in: str or list of str
        A filepath, directory, or lists of either that
        points to maps stored in .g3(.gz) files.
        
    map_id: str or list of str
        Add maps that have an Id key matching the pattern, or one of the 
        patterns in the list. Maps matching separate patterns are added to
        separate coadds. Understands Unix shell-style wildcards (e.g. *, ?).
        
    output_file: str
        If specified, save the output map to this path.
        
    Returns
    -------
    G3Frame containing the coadded map
    """
    if not isinstance(maps_in, list):
        maps_in = [maps_in]
    if not all([isinstance(mp, str) for mp in maps_in]):
        raise TypeError("All inputs must be strings")
        
    if map_id is not None:
        if not isinstance(map_id, list):
            map_id = [map_id]
        if not all([isinstance(mid, str) for mid in map_id]):
            raise TypeError("All inputs must be strings")


    maps_to_add = []
    for pth in maps_in:
        if os.path.isdir(pth):
            now_maps = glob(os.path.join(pth,'*.g3*'))
            maps_to_add += now_maps
        elif os.path.isfile(pth):
            maps_to_add.append(pth)
        else:
            raise OSError(
                "%s is not an existing file or directory."%pth)
    
    map_out = _GrabMapFrame()
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=maps_to_add)
    pipe.Add(core.Dump)
    if obslist_to_cut is not None:
        pipe.Add(CutObsids, obslist=obslist_to_cut)
    pipe.Add(lambda frame: frame.type == core.G3FrameType.Map)
    pipe.Add(lambda frame: 'ScanNumber' not in frame or \
             frame['ScanNumber']<70)
    if args.sign_flip:
        pipe.Add(FlipSignsForCoadd, seed=42)
    if not map_id:
        pipe.Add(util.framecombiner.MapFrameCombiner, fr_id=None)
    else:
        for id in map_id:
            pipe.Add(util.framecombiner.MapFrameCombiner, fr_id=id)
    pipe.Add(lambda frame: 'Id' not in frame or \
             fnmatch(frame['Id'], 'combined_*'))
    if output_file is not None:
        parent = os.path.dirname(output_file)
        if parent != "" and not os.path.exists(parent):
            os.makedirs(parent)
        pipe.Add(core.G3Writer, filename=output_file)    
    pipe.Add(map_out)
    pipe.Run()
    
    return map_out.map
    
class _GrabMapFrame():
    def __init__(self):
        self.map = None
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map:
            self.map = frame
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('maps_in', nargs='+')
    parser.add_argument('--output-file', '-o', default='map_coadd.g3',
                        help='output filename')
    parser.add_argument('--map-id', nargs='+', default=None,
                        help='Strings used to filter map ids.')
    parser.add_argument('--sign-flip', action='store_true',
                        help='Coadd observations with sign-flips in order '
                        'to extract noise spectra.')
    parser.add_argument('--obslist-to-cut', type=str, default=None,
                        help='Pickle file with list of observations to cut.')
    args = parser.parse_args()
    
    if args.obslist_to_cut is not None:
        obslist_to_cut = []
        with open(args.obslist_to_cut, 'rb') as f:
            obsdict_to_cut = pickle.load(f)
            for field in obsdict_to_cut:
                obslist_to_cut = np.hstack([obslist_to_cut, obsdict_to_cut[field]])

    coadd_maps(args.maps_in, map_id=args.map_id, output_file=args.output_file,
               obslist_to_cut=obslist_to_cut)
