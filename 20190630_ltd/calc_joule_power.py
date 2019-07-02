from spt3g import core, std_processing, calibration, todfilter, mapmaker, dfmux
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('infiles', nargs='+', help='Input files.')
parser.add_argument('-o', '--output', default='output.g3',
                    help='Name of output file.')
args = parser.parse_args()

def cleanup(frame, to_save=[]):
    if frame.type == core.G3FrameType.Scan:
        for key in frame:
            if key not in to_save:
                frame.pop(key)
        return frame
    elif frame.type == core.G3FrameType.EndProcessing:
        return frame
    elif frame.type == core.G3FrameType.Calibration:
        return frame
    else:
        return []

def calc_tod_median(frame, Input='TimestreamsWatts', Output='AvgPower'):
    if frame.type == core.G3FrameType.Scan:
        frame[Output] = core.G3MapDouble()
        for bolo in frame[Input].keys():
            frame[Output][bolo] = np.median(frame[Input][bolo][np.isfinite(frame[Input][bolo])])

pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=args.infiles)
pipe.Add(dfmux.unittransforms.ConvertTimestreamUnits,
         Input='RawTimestreams_I', Output='TimestreamsWatts')
pipe.Add(calc_tod_median)
pipe.Add(cleanup, to_save='AvgPower')
pipe.Add(core.G3Writer, filename=args.output)

pipe.Run(profile=True)
