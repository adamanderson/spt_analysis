from spt3g import core, maps
from glob import glob
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('inpath', 
                    help='Path containing files to coadd.')
parser.add_argument('outpath',
                    help='Path to which to write output files.')
parser.add_argument('action', choices=['add', 'difference'],
                    help='Add or difference the L and R maps.')
args = parser.parse_args()

infiles = glob(os.path.join(args.inpath, '*.g3.gz'))


class LRMapCombiner(object):
    def __init__(self, bands):
        self.bands = bands
        self.map_cache = {band: {} for band in self.bands}
        self.added = {band: False for band in self.bands}
        
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map:
            for band in self.bands:
                if band in frame['Id']:
                    if self.added[band]:
                        self.map_cache[band] = {}

                    if len(self.map_cache[band]) == 0:
                        self.map_cache[band]['Id'] = band
                        for mapkey in ['T', 'Q', 'U', 'Wpol']:
                            self.map_cache[band][mapkey] = frame[mapkey]
                        return []
                    else:
                        combined_frame = core.G3Frame(core.G3FrameType.Map)
                        for mapkey in ['T', 'Q', 'U', 'Wpol']:
                            combined_frame[mapkey] = self.map_cache[band][mapkey] + frame[mapkey]
                        combined_frame['Id'] = band

                        self.added[band] = True
                        return combined_frame

for jfile, infile in enumerate(infiles):
    print('Processing {} ({} of {})'.format(os.path.basename(infile), jfile, len(infiles)))

    outfile = os.path.basename(infile).rstrip('.g3.gz') + '_lrcoadd.g3.gz'
    outfile = os.path.join(args.outpath, outfile)

    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=infile)
    pipe.Add(LRMapCombiner, bands=['90GHz', '150GHz', '220GHz'])
    pipe.Add(core.G3Writer, filename=outfile)
    pipe.Run()
