from spt3g import core, std_processing, calibration
from spt3g.std_processing.gainmatching import match_gains, difference_pairs, sum_pairs
import os.path

datapath = '/spt/data/bolodata/downsampled/noise/68609192/'

def cleanup(frame):
    if 'GainMatchCoeff' in frame.keys():
        newframe = core.G3Frame()
        newframe['GainMatchCoeff'] = frame['GainMatchCoeff']
        return newframe

pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=[os.path.join(datapath, 'offline_calibration.g3'),
                                  os.path.join(datapath, '0000.g3')])
pipe.Add(std_processing.flagsegments.FieldFlaggingPreKcmbConversion,
         flag_key = 'Flags', ts_key = 'RawTimestreams_I')
pipe.Add(std_processing.CalibrateRawTimestreams,
        output = 'CalTimestreams')
pipe.Add(core.Delete, keys = ['RawTimestreams_I', 'RawTimestreams_Q', 'TimestreamsWatts'])
pipe.Add(std_processing.flagsegments.FieldFlaggingPostKcmbConversion,
         flag_key = 'Flags', ts_key = 'CalTimestreams')
pipe.Add(match_gains, ts_key = 'CalTimestreams', flag_key='Flags',
         gain_match_key = 'GainMatchCoeff', freq_range=[0.1, 1.0])
pipe.Add(cleanup)
# pipe.Add(difference_pairs, ts_key = 'CalTimestreams',
#          gain_match_key = 'GainMatchCoeff',
#          pair_diff_key='PairDiffTimestreams')
# pipe.Add(sum_pairs, ts_key = 'CalTimestreams',
#          gain_match_key = 'GainMatchCoeff',
#          pair_diff_key='PairSumTimestreams')
pipe.Add(core.G3Writer, filename='gain_match_test.g3')
pipe.Run(profile=True)
